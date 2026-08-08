# Reproducing the results in this repository

This file states exactly which numbers can be reproduced from what is here,
which cannot, and what is needed for the ones that cannot. It exists because
the two halves of the paper have very different reproduction costs, and the
repository was shipped with only one of them.

## The diagnostic: fully reproducible here

`experiments/adapter_placement.py` is the measurement. It needs a GPU and the
base model, it does not train anything, and it produces the module ranking.

    MODEL=meta-llama/Meta-Llama-3-8B TASK_A=code TASK_B=prose \
        python experiments/adapter_placement.py

This writes `placement_code_prose.json`: one row per attention projection with
its same-task ceiling, its cross-task overlap, the relative drop `rel`, and a
`saturated` flag. The suggested placement is the top quartile of the
non-saturated rows by `rel`.

Two claims of the paper are checkable from this output alone: that the leading
cosine saturates at 1.0 for every k while the overlap dimension does not, and
the composition of the suggested set on Llama-3-8B (17 `o_proj`, 10 `q_proj`,
4 `k_proj`, no `v_proj` at all).

## The retention experiment: NOT reproducible from this repository alone

The headline comparison, 0.7557 against 0.5405 in retention and a 36.8 per cent
relative gain in absolute capability over ten paired seeds, comes from training
runs. `results/placement.json` holds the per-seed outcomes of those runs, so
every statistic in the paper can be recomputed from it, but the runs themselves
cannot be repeated from this repository. Three things were missing from the
original release. The first is fixed below; the other two are stated rather
than left implicit.

**1. The exact module list used: now shipped.** It was not in the original
release. `results/placement.json` describes the suggested placement in prose
only. The two lists behind the published numbers are now in
`results/placements_used.json`, recovered from the queue script of the original
run and checked against the composition the paper reports: 17 `o_proj`, 10
`q_proj`, 4 `k_proj` and no `v_proj`, with both arms at exactly 31 modules so
that what varies is location and not capacity. A reader who re-runs the
diagnostic can compare their own ranking against that file.

**2. The training harness.** The two-phase fine-tuning and the HumanEval
evaluation were run with the harness of the companion repository,
github.com/BiomeMakers/OmegaS-LLM, not with code in this repository.

**3. Explicit-placement support in that harness.** The harness must accept a
`LORA_MODULES` environment variable listing full module names. Without it every
arm falls back to the `q_proj`/`v_proj` default, the two arms become identical,
and the experiment silently returns a null result rather than an error.
`experiments/retention_placement.py` refuses to run if the hook is absent, for
that reason.

## Driving the retention experiment once those are in place

    python experiments/retention_placement.py            # uses the published lists
    python experiments/retention_placement.py --ranking placement_code_prose.json --check
    python experiments/retention_placement.py --ranking placement_code_prose.json --emit
    python experiments/retention_placement.py --ranking placement_code_prose.json --run --harness ../OmegaS-LLM

The first form selects both placements, verifies the suggested one against the
composition reported in the paper, writes `placements_used.json` and stops. It
needs no GPU. The second prints the twenty commands. The third runs them.

Both arms run with no penalty. What varies between them is location, not
capacity and not regularisation.

## What the numbers mean, and their limits

Run-to-run standard deviation on the retention ratio was measured at 0.104 on
the same seed, the same configuration and the same hardware. Any mean
difference below roughly 0.066 is not distinguishable at ten seeds. That figure
is in `results/placement.json` and it is the reason the third arm in that file,
which adds a regulariser on top of the suggested placement, is reported as not
distinguishable rather than as an improvement.

Retention is a ratio, so an arm that learns less of the first task has its
ratio inflated. `results/placement.json` therefore records HumanEval after task
A for every cell, and the absolute figures should be read alongside the ratio.

The candidate set of the diagnostic is attention projections only. Nothing in
the method prevents applying it to MLP modules, and other work locates
important modules there, but this repository does not measure that case.
