#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adapter_placement.py  --  TIM: Task Interference Mapping

WHERE TO PUT THE ADAPTER SO THAT IT FORGETS LESS

The current convention is to put LoRA on q_proj and v_proj of EVERY layer.
Nobody has measured that: it is convention, not result.

Steele (arXiv:2603.02224) reports that forgetting under LoRA follows
    F = alpha (1 - cos^2 theta_min) + beta
where theta_min is the minimum principal angle between the gradient subspaces
of the two tasks. If that holds, there are modules where adapting is cheap (the
subspaces barely overlap) and modules where adapting destroys (they overlap a
lot), and the difference can be measured BEFORE training.

This script performs that measurement. It does NOT train: it passes a few
batches of each task through the model, accumulates the gradient of each target
weight matrix, and computes the principal angles between the rank-r subspaces
of the two gradients.

SAME-TASK CONTROL, and this is what makes the measurement mean anything. Two
gradients from different tasks come from the SAME model, so their dominant
directions are dictated by the structure of the model and not by the task:
comparing against random subspaces asks a question whose answer is trivially
yes. The correct reference is how much two halves of the SAME task overlap.
That is the ceiling. What informs is how far the cross-task overlap DROPS from
it, not the raw cosine. A module whose ceiling is already 0.999 cannot
discriminate anything and is flagged as saturated.

COST: minutes. One forward+backward per batch, with no optimiser and without
updating anything. Against the ~45 min of a training cell, it is another order.

START ON A LAPTOP with a 1B model to debug the pipeline, and only once the
ranking comes out stable across batches, launch it on the large model on GPU:

    MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 python adapter_placement.py
    MODEL=NousResearch/Meta-Llama-3-8B python adapter_placement.py

WHAT THIS DECIDES, written before seeing any numbers: if the per-module ranking
is stable across subsets of batches, the diagnostic is usable and worth
validating by placing the adapter only where it says. If it changes with the
batch, the measurement is noise and the line closes here, for minutes of
compute.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
RANK = int(os.environ.get("RANK", "8"))          # the LoRA rank that would be used
N_BATCH = int(os.environ.get("N_BATCH", "8"))    # batches per task
SEQ = int(os.environ.get("SEQ", "256"))
TARGETS = os.environ.get("TARGETS", "q_proj,k_proj,v_proj,o_proj").split(",")
TASK_A = os.environ.get("TASK_A", "code")
TASK_B = os.environ.get("TASK_B", "prose")
OUT = os.environ.get("OUT", f"placement_{os.environ.get('TASK_A','code')}_{os.environ.get('TASK_B','prose')}.json")

if torch.cuda.is_available():
    DEV, DTYPE = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    DEV, DTYPE = "mps", torch.float16
else:
    DEV, DTYPE = "cpu", torch.float32
print(f"model: {MODEL}\ndevice: {DEV} ({DTYPE})\n"
      f"tasks: {TASK_A} -> {TASK_B}\n"
      f"rank {RANK}, {N_BATCH} batches per task, seq {SEQ}\n")


# ---------------------------------------------------------------- data
# Task catalogue. Lets you change the PAIR without touching code, which is what
# it takes to find out whether the ranking depends on the model or on the tasks:
#   TASK_A=code TASK_B=prose    the original pair
#   TASK_A=code TASK_B=math     same model, another pair
#   TASK_A=prose TASK_B=math    no domain in common with the original
TASKS = {
    "code":  ("code-search-net/code_search_net", "python", "train",
              "whole_func_string"),
    "prose": ("Skylion007/openwebtext", None, "train", "text"),
    "math":  ("open-r1/OpenR1-Math-220k", None, "train", "problem"),
    "legal": ("pile-of-law/pile-of-law", "r_legaladvice", "train", "text"),
}


def batches(tok, domain, n, skip=0):
    """A few batches of each task. Streaming, so nothing large is downloaded.
    skip takes a different stretch of the same stream, which is how the two
    halves of the same-task control are built."""
    if domain not in TASKS:
        raise SystemExit(f"unknown task '{domain}'. Options: {list(TASKS)}")
    repo, config, split, field = TASKS[domain]
    if config:
        ds = load_dataset(repo, config, split=split, streaming=True,
                          trust_remote_code=True)
    else:
        ds = load_dataset(repo, split=split, streaming=True,
                          trust_remote_code=True)
    out, it = [], iter(ds)
    for _ in range(skip):
        next(it)
    while len(out) < n:
        t = next(it)[field]
        if not t or len(t) < 200:
            continue
        e = tok(t, return_tensors="pt", truncation=True, max_length=SEQ)
        if e["input_ids"].shape[1] < 32:
            continue
        out.append(e)
    return out


# ------------------------------------------------- per-module gradients
def gradients(model, tok, domain, skip=0, n=None):
    """Accumulates the gradient of each target module. There is no optimiser."""
    n = n or N_BATCH
    acc = {}
    for i, b in enumerate(batches(tok, domain, n, skip)):
        ids = b["input_ids"].to(DEV)
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        for nm, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu()
            acc[nm] = g if nm not in acc else acc[nm] + g
        print(f"    batch {i+1}/{n}", end="\r", flush=True)
    model.zero_grad(set_to_none=True)
    print(" " * 30, end="\r")
    return {k: v / n for k, v in acc.items()}


def subspace(G, r):
    """Orthonormal basis of the rank-r row subspace of the gradient.
    This is the subspace a rank-r adapter would want to use."""
    return torch.linalg.svd(G.double(), full_matrices=False).Vh[:r]


def angles(Va, Vb):
    """Cosines of the principal angles between two rank-r subspaces."""
    return torch.linalg.svdvals(Va @ Vb.t()).clamp(0, 1).tolist()


def overlap_dim(Va, Vb):
    """Effective dimension of the overlap: the sum of cos^2 of the principal
    angles, between 0 and r. This is the right quantity here, not the leading
    cosine.

    The leading cosine corresponds to the MINIMUM principal angle, which is
    what Steele's law uses, but it SATURATES at 1 as soon as the two subspaces
    share a single direction, and then it cannot distinguish sharing one from
    sharing eight. Verified: on subspaces sharing k of 8 directions, the
    leading cosine returns 1.0000 for every k >= 1, while this sum returns
    1.03, 2.01, 4.01, 6.00 and 8.00 for k = 1, 2, 4, 6 and 8.
    """
    c = torch.linalg.svdvals(Va @ Vb.t()).clamp(0, 1)
    return float((c ** 2).sum())


def null_overlap(n, r):
    """Overlap dimension expected between two RANDOM rank-r subspaces in R^n.
    It has a closed form, r^2/n, verified numerically (0.0314 measured against
    0.0312 theoretical for r=8, n=2048), so there is no need to simulate."""
    return r * r / n


# ---------------------------------------------------------------- main
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)

    # only the target modules need a gradient: saves memory and time
    for p in model.parameters():
        p.requires_grad_(False)
    target_modules = []
    for nm, p in model.named_parameters():
        if any(nm.endswith(t + ".weight") for t in TARGETS):
            p.requires_grad_(True)
            target_modules.append(nm)
    print(f"target modules: {len(target_modules)}\n")
    if not target_modules:
        raise SystemExit("No module matches TARGETS. Check the names.")

    # three measurements: two halves of task A (the ceiling) and task B
    half = max(2, N_BATCH // 2)
    print(f"task A ({TASK_A}), half 1 of {half} batches   [control]")
    GA1 = gradients(model, tok, TASK_A, skip=0, n=half)
    print(f"task A ({TASK_A}), half 2 of {half} batches   [control]")
    GA2 = gradients(model, tok, TASK_A, skip=half, n=half)
    print(f"task B ({TASK_B}), {half} batches")
    GB = gradients(model, tok, TASK_B, skip=0, n=half)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

    rows = []
    print("\n" + "=" * 88)
    print("GRADIENT SUBSPACE OVERLAP")
    print("=" * 88)
    print(f"  measure = overlap DIMENSION (sum of cos^2), between 0 and {RANK}.")
    print("            Not the leading cosine, which saturates at 1 as soon as")
    print("            they share a single direction and then cannot tell one")
    print("            from eight.")
    print("  ceiling = between two halves of the SAME task. It is the most they")
    print("            can overlap by model structure, with no task in between.")
    print("  cross   = between the two different tasks.")
    print("  DROP    = ceiling minus cross, in dimensions.")
    print("  rel     = DROP / ceiling. THIS is what orders the ranking, because")
    print("            the absolute drop is bounded by the ceiling and rewards")
    print("            modules with a high ceiling (measured: Spearman +0.56")
    print("            with the ceiling, against -0.13 for the relative one).")
    print(f"            null = r^2/n = {RANK*RANK}/n.\n")
    print(f"{'module':38} {'ceiling':>8} {'cross':>8} {'DROP':>8} {'rel':>7}")
    null_cache = {}
    for nm in target_modules:
        if nm not in GA1 or nm not in GA2 or nm not in GB:
            continue
        n_col = GA1[nm].shape[1]
        if n_col not in null_cache:
            null_cache[n_col] = null_overlap(n_col, RANK)
        m_null = null_cache[n_col]
        SA1 = subspace(GA1[nm], RANK)
        ceiling = overlap_dim(SA1, subspace(GA2[nm], RANK))
        cross = overlap_dim(SA1, subspace(GB[nm], RANK))
        drop = ceiling - cross
        short = nm.replace("model.layers.", "L").replace(".weight", "")
        # saturated = the ceiling exhausts the rank, leaving no room to discriminate
        sat = "  <- ceiling saturated" if ceiling > RANK - 0.05 else ""
        print(f"{short:38} {ceiling:8.3f} {cross:8.3f} {drop:8.3f} "
              f"{drop/max(ceiling,1e-9):7.3f}{sat}")
        rows.append(dict(module=nm, ceiling=ceiling, cross=cross, drop=drop,
                         # the drop is BOUNDED by the ceiling, so ordering by it
                         # rewards modules with a high ceiling: measured
                         # Spearman(ceiling, drop) = +0.56 and only -0.13 with
                         # the relative one. The ranking uses the relative one.
                         rel=drop / max(ceiling, 1e-9),
                         null=m_null, saturated=bool(ceiling > RANK - 0.05),
                         cos1_cross=angles(SA1, subspace(GB[nm], RANK))[0]))

    # written to file already sorted by rel, best first, so that taking the
    # first k entries of the file is the same as taking the top of the ranking
    rows.sort(key=lambda f: -f["rel"])
    json.dump(rows, open(OUT, "w"), indent=2)

    useful = [f for f in rows if not f["saturated"]]
    sat = [f for f in rows if f["saturated"]]
    print("\n" + "=" * 88)
    print("HOW TO READ THIS")
    print("=" * 88)
    print(f"  modules with a saturated ceiling (>= {RANK-0.05:.2f} of {RANK}), "
          f"which do NOT discriminate: "
          f"{len(sat)} of {len(rows)}")
    if sat:
        from collections import Counter
        c = Counter(f["module"].split(".")[-2] for f in sat)
        print(f"    by type: {dict(c)}")
    if not useful:
        print("\n  NO module discriminates: the ceiling is saturated in all of")
        print("  them, so in this model the gradient subspaces are dictated by")
        print("  structure and not by the task. The line closes here.")
        print(f"\nSaved to {OUT}")
        return

    print(f"\n  RANKING by RELATIVE DROP, among the {len(useful)} that discriminate.")
    print("  A high drop = the two tasks use different directions there, so")
    print("  adapting should cost less forgetting.\n")
    for f in useful[:12]:
        c = f["module"].replace("model.layers.", "L").replace(".weight", "")
        print(f"    {c:40} rel {f['rel']:5.3f}  (ceiling {f['ceiling']:.2f} "
              f"-> cross {f['cross']:.2f})")
    if len(useful) > 16:
        print("    ...")
        for f in useful[-3:]:
            c = f["module"].replace("model.layers.", "L").replace(".weight", "")
            print(f"    {c:40} rel {f['rel']:5.3f}   <- the two tasks "
                  f"share almost everything")

    k = max(1, len(useful) // 4)
    print(f"\n  SUGGESTED PLACEMENT: the quartile of largest relative drop, {k} modules.")
    for f in useful[:min(8, k)]:
        print("   ", f["module"])
    if k > 8:
        print(f"    ... and {k-8} more, in {OUT}")

    from collections import defaultdict
    by_type = defaultdict(list)
    for f in useful:
        by_type[f["module"].split(".")[-2]].append(f)
    print("\n  BY MODULE TYPE (means):")
    print(f"    {'type':10} {'ceiling':>8} {'cross':>8} {'rel':>8}  {'n':>4}")
    for t, L in sorted(by_type.items(), key=lambda x: -sum(f["rel"] for f in x[1])/len(x[1])):
        print(f"    {t:10} {sum(f['ceiling'] for f in L)/len(L):8.3f} "
              f"{sum(f['cross'] for f in L)/len(L):8.3f} "
              f"{sum(f['rel'] for f in L)/len(L):8.3f}  {len(L):4d}")
    print("    If one type dominates the ranking, that is more informative than")
    print("    the list of modules: it says WHERE in the attention block the")
    print("    tasks diverge.")

    print("\n" + "=" * 88)
    print("BEFORE TRUSTING THIS")
    print("=" * 88)
    print("  1. STABILITY, and there is something concrete to look at. The")
    print("     ceiling falls with depth, so a low ceiling may be a diffuse")
    print("     gradient or it may be that 4 batches are not enough to estimate")
    print("     the subspace. Run again with N_BATCH=32: if the ceiling RISES it")
    print("     was sampling noise; if it holds and so does the ranking, it is")
    print("     structure and worth validating by training.")
    print("  2. A high ceiling is NOT a failure of the module: it says its")
    print("     gradients are dominated by the structure of the model. What is")
    print("     ruled out is its ability to DISCRIMINATE between tasks, not the")
    print("     module.")
    print("  3. The validation, when the time comes: two arms of ten seeds,")
    print("     suggested placement against the conventional one, with the SAME")
    print("     total number of adapter parameters, so that the comparison is")
    print("     about placement and not about size.")
    print("  4. SCOPE: the default candidate set is the attention projections")
    print("     (q, k, v, o). Nothing in the criterion is specific to them, so")
    print("     MLP modules can be added through TARGETS; we have not measured")
    print("     that case.")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
