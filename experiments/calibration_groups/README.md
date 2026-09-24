# Calibration groups: which parts of a calibration set a block-removal choice sacrifices

An application of the TIM logic (compare, module by module, how different sets of gradients respond) to depth
pruning of large language models. TIM was built to decide where to place an adapter; here the modules are whole
transformer blocks and the decision is which ones to remove.

## The question

Block-removal methods score blocks with a calibration set and average over all its samples. If the set mixes
content that depends on different blocks, the average can hide the content that a given removal choice damages
most. This experiment asks whether that content can be found without labels, from per-sample gradients alone,
and whether it has a name.

## Data

Per-sample gradient matrices `A` (2,048 samples × N blocks) published by Jansen, Rausch, Hashemi, Montero and Orús
for "LLM Compression by Block Removal with Constrained Binary Optimization" (arXiv:2602.00161): Zenodo
10.5281/zenodo.20125262, CC BY 4.0. They are not redistributed here; download `Amatrices.zip` and unzip it into
`Amatrices_final/`.

Their public code (configs/cbo_configs) shows that Llama-3.1-8B, Llama-3.3-70B and Qwen3-14B were calibrated on the
same samples: OpenHermes-2.5 train, shuffled with seed 42, first 2,048. `export_samples.py` reconstructs those
samples. `results/samples_and_groups.csv` lists, for each row of `A`, its OpenHermes index, source, length and the
group assigned in each model. Conversation text is not included.

## Method

`tim_calib.py`
- `descubrir_grupos`: k-means (k = 2 to 6, best silhouette) on each sample's importance profile, i.e. how its
  squared gradient is distributed across blocks, in Hellinger coordinates; stability by 80 % resampling.
- `perfil_bloques`: which blocks each group values more than the whole set.
- `dano`: the share of each group's importance removed by a given set of blocks, relative to the whole set, with a
  null from random groups of the same size.
- `prueba_sintetica`: three planted groups; the tool recovers them (adjusted Rand 0.85) and detects the damage to
  the small one (2.30 against a null 95th percentile of 1.17). A first version that clustered on gradient
  direction failed this test (adjusted Rand 0.00) and was replaced before any real result was read.

## Results

Rows of `A` align with the reconstructed samples: Spearman between gradient norm and sample length is −0.90,
−0.74 and −0.64 for the three models; misaligned rows would give values near zero.

Groups found without labels are associated with OpenHermes sources in all three models (Cramér's V 0.156, 0.566
and 0.255 against permutation nulls at the 99th percentile of about 0.14), and sample length does not explain them
(AUC 0.54 to 0.67).

The clearest case is Llama-3.3-70B. One group gathers mathematics and science (metamath and UnnaturalInstructions
at about twice their share, code at 0.4 of it); the other gathers code (glaive-code-assist at 1.7 times its share,
metamath at 0.1). The published lowest-energy removal of 16 of 80 blocks takes 1.23 times more of the mathematics
group's importance than of the whole set (null range 0.91 to 1.09). In the authors' own tables that same
configuration keeps MMLU at 80.7 while GSM8K falls to 60.6, against 74.9 for the block-influence baseline.

In three of four models the published selection damages one group by 22 to 29 % more than the whole set, above
the null. Groups are not shared across models (adjusted Rand between models 0.004 to 0.027): each model splits the
same content differently, so the tool describes what a given model does with a given calibration set.

## Scope

One clear case (Llama-3.3-70B), a weaker one (Qwen3-14B) and a marginal one (Llama-3.1-8B). The match with the
GSM8K drop is a single comparison, not a test. A confirmatory run needs new per-sample gradients on a calibration
set with known composition and a downstream evaluation of the pruned models.

## Files

`tim_calib.py`, `read_pt.py` (reads `.pt` tensors without torch), `report.py` (groups, block profiles and damage for
the four models), `sources_check.py` (alignment and source association), `export_samples.py`,
`results/samples_and_groups.csv`. Pre-registrations and the full log of both checks, including the ones that failed,
are kept with the project notes.

## Licence

Same as this repository: AGPL-3.0 for research and non-commercial use, commercial licence on request. The
Multiverse gradient matrices keep their own licence (CC BY 4.0); their code, which this folder does not include or
modify, is licensed for non-commercial research only.
