# Reader control arm: pre-registered, 21 August 2026

This directory holds the per-seed results of a pre-registered control arm, and a
fresh run of the measured placement so that both arms come from the same session,
the same hardware and the same library versions.

## What is being compared

**Measured** (`pl_A2_s*.json`) is the placement TIM suggests: 17 `o_proj`,
10 `q_proj`, 4 `k_proj`, no `v_proj`. Same set as `results/placements_used.json`.

**Reader** (`pl_B_s*.json`) is the placement a reader of the paper could build
without running the diagnostic. The paper reports the type composition of the
suggested set but not its depth distribution, so this arm uses the published
composition (17/10/4) with the layers drawn at random, one draw per seed. The
exact lists were frozen before any result existed and are in
`frozen_module_lists.txt`.

The two arms share type composition, and therefore share the trainable parameter
budget exactly. The only thing that differs is which layers are adapted.

## Result

| | Retention (mean, 10 seeds) |
|---|---|
| Measured | 0.7216 |
| Reader | 0.3923 |

Paired difference +0.3293, sd 0.1455, standard error 0.0460, t = 7.16, and the
measured arm wins in 10 of 10 seeds. The pre-registered indistinguishability
threshold was 0.066 in mean retention at ten seeds.

Capability after the first task: 0.2732 for the measured arm, 0.3116 for the
reader's. The reader's placement learns the new task better and retains less, so
the difference is not a ratio flattered by an arm that learned less.

The reader's placement is also worse than the plain `q,v` default (0.5405),
by 0.148 in 8 of 10 seeds.

## Setup

Llama-3-8B, LoRA rank 8, plain LoRA with no penalty (`--arm none`), 31 adapted
modules in every arm, HumanEval pass@1 measured after each task. Seeds 42, 123,
456, 789, 1011, 2022, 3033, 4044, 5055, 6066. Run on 5x A100 SXM 80GB,
torch 2.8.0+cu128, peft 0.20.0, 21 August 2026.

## A reproducibility caveat worth reading

Per-seed values do not transfer across machines. Seed 42 of the measured arm
gives 0.5333 in the August session and 0.8049 here, with identical code and seed.
The distribution reproduces and the comparison between arms is unchanged, but
anyone rerunning this repository on other hardware should expect to match the
former and not the latter. This is why both arms above were re-run in one
session rather than compared across sessions.

## Files

- `pl_A2_s<seed>.json` and `.log`: measured placement, ten seeds
- `pl_B_s<seed>.json` and `.log`: reader placement, ten seeds
- `frozen_module_lists.txt`: the ten module lists of the reader arm, one per
  seed, generated with `random.Random(seed)` and `random.sample(range(32), n)`
  over the types in order `o_proj`, `q_proj`, `k_proj`
