#!/usr/bin/env python3
"""retention_placement.py -- reproduce the retention experiment of the paper.

The diagnostic in `adapter_placement.py` produces a ranking. This script turns
that ranking into the two placements the paper compares, checks them against
what the paper claims, and drives the training harness that measures retention.

    python experiments/retention_placement.py --check
        selects the two placements, verifies them against the paper, prints
        them, and stops. No GPU, no training.

    python experiments/retention_placement.py --emit
        additionally prints the exact commands for the ten paired seeds.

    python experiments/retention_placement.py --run --harness ../OmegaS-LLM
        runs them.

WHAT THIS SCRIPT NEEDS THAT THE REPOSITORY DOES NOT YET CONTAIN

1. The diagnostic output for the pair reported in the paper. Pass it with
   --ranking. `results/placement.json` holds the retention results, not the
   ranking, so it cannot be used here.
2. The training harness, which lives in the companion repository
   (github.com/BiomeMakers/OmegaS-LLM). It must support the LORA_MODULES
   environment variable for explicit placement; without it every arm falls
   back to the q_proj/v_proj default and the two arms become identical.
   The script checks for that and refuses to run otherwise, because the
   failure would otherwise be silent and would produce two arms that look
   like a null result.
"""

import argparse
import collections
import json
import os
import subprocess
import sys

SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

# What the paper says the suggested placement is. Used as a check, not as input.
PAPER_CLAIM = {"o_proj": 17, "q_proj": 10, "k_proj": 4, "v_proj": 0}
PAPER_N = 31


def module_type(name):
    for t in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if name.endswith(t):
            return t
    return name.rsplit(".", 1)[-1]


def suggested_from_ranking(path, n=PAPER_N):
    rows = json.load(open(path))
    useful = [r for r in rows if not r.get("saturated")]
    ranked = sorted(useful, key=lambda r: -r["rel"])
    picked = [r["module"] for r in ranked[:n]]
    return picked, len(rows), len(rows) - len(useful)


def conventional(picked, layers=None):
    """The default placement: q_proj and v_proj, matched in count to `picked`.

    The paper matches parameter count rather than layer coverage, so the
    number of modules is what has to agree.
    """
    n_layers_needed = (len(picked) + 1) // 2
    mods = []
    for i in range(n_layers_needed):
        mods += [f"model.layers.{i}.self_attn.q_proj",
                 f"model.layers.{i}.self_attn.v_proj"]
    return mods[:len(picked)]


def check(picked):
    counts = collections.Counter(module_type(m) for m in picked)
    print(f"  {len(picked)} modules selected")
    ok = True
    for t in ("o_proj", "q_proj", "k_proj", "v_proj"):
        got, want = counts.get(t, 0), PAPER_CLAIM[t]
        marca = "ok " if got == want else "!! "
        if got != want:
            ok = False
        print(f"  {marca}{t:8s} {got:>3}   paper says {want}")
    if not ok:
        print("\n  The selected set does NOT match the composition reported in\n"
              "  the paper. Either the ranking comes from a different task pair\n"
              "  or model, or the diagnostic is not deterministic across runs.\n"
              "  Do not report retention numbers from a set that disagrees\n"
              "  with the one the paper describes without saying so.")
    return ok


def harness_supports_explicit_placement(harness):
    p = os.path.join(harness, "experiments", "rerun_retention.py")
    if not os.path.exists(p):
        return False, f"not found: {p}"
    src = open(p, encoding="utf-8").read()
    if "LORA_MODULES" not in src:
        return False, ("the harness has no LORA_MODULES hook, so it would "
                       "silently use the q_proj/v_proj default for BOTH arms")
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default="placement_code_prose.json",
                    help="output of adapter_placement.py for the reported pair")
    ap.add_argument("--harness", default="../OmegaS-LLM")
    ap.add_argument("--used", default="results/placements_used.json",
                    help="the placements actually used in the paper")
    ap.add_argument("--n", type=int, default=PAPER_N)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    # The placements behind the published numbers are shipped in
    # results/placements_used.json, recovered from the queue script of the
    # original run. Use them unless a ranking is given explicitly.
    if not os.path.exists(a.ranking) and os.path.exists(a.used):
        d = json.load(open(a.used))
        picked, conv = d["suggested"], d["conventional"]
        print(f"using the published placements from {a.used}")
        print(f"  source: {d.get('source','')}")
        check(picked)
        if not (a.emit or a.run):
            for m in picked:
                print(f"    {m}")
            return 0
        return run_or_emit(picked, conv, a)

    if not os.path.exists(a.ranking):
        print(f"Ranking file not found: {a.ranking}\n\n"
              "Produce it first:\n"
              "    MODEL=meta-llama/Meta-Llama-3-8B TASK_A=code TASK_B=prose \\\n"
              "        python experiments/adapter_placement.py\n\n"
              "Note that results/placement.json is NOT this file: it holds the\n"
              "retention results, not the module ranking.")
        return 1

    picked, n_total, n_sat = suggested_from_ranking(a.ranking, a.n)
    conv = conventional(picked)

    print(f"ranking: {a.ranking}  ({n_total} modules, {n_sat} saturated)")
    print("\nSUGGESTED placement (top quartile by relative drop)")
    ok = check(picked)
    for m in picked:
        print(f"    {m}")
    print(f"\nCONVENTIONAL placement ({len(conv)} modules, q_proj and v_proj)")
    for m in conv[:4]:
        print(f"    {m}")
    print(f"    ... {len(conv) - 4} more")

    if len(conv) != len(picked):
        print("\n  WARNING: the two arms do not have the same number of "
              "modules, so capacity is not held fixed.")

    json.dump({"suggested": picked, "conventional": conv,
               "n": len(picked), "ranking_file": a.ranking,
               "matches_paper_composition": ok},
              open("placements_used.json", "w"), indent=2)
    print("\nWritten to placements_used.json")

    if a.check:
        return 0 if ok else 1

    return run_or_emit(picked, conv, a)


def run_or_emit(picked, conv, a):
    supported, why = harness_supports_explicit_placement(a.harness)
    if not supported:
        print(f"\nCannot run: {why}")
        print("Clone the companion repository next to this one, or pass "
              "--harness.")
        return 1
    cmds = []
    for arm, mods in (("suggested", picked), ("conventional", conv)):
        for s in SEEDS:
            cmds.append(
                f'LORA_MODULES="{",".join(mods)}" '
                f'python experiments/rerun_retention.py --cell --seed {s} '
                f'--arm none --out retention_{arm}_{s}.json')

    print(f"\n{len(cmds)} cells (2 placements x {len(SEEDS)} seeds). "
          "Both arms use --arm none: no penalty is involved, only placement.")
    if a.emit or not a.run:
        for c in cmds:
            print("  " + c)
        return 0

    for c in cmds:
        print("\n>>> " + c[:110] + " ...")
        subprocess.run(c, shell=True, cwd=a.harness, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
