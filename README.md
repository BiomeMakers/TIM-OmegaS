# TIM

**Task Interference Mapping**

[![USPTO Patent Pending](https://img.shields.io/badge/USPTO-Patent%20Pending%2064%2F121%2C656-blue)](https://www.uspto.gov)

**LoRA tells you how to adapt. TIM tells you where.**

---

You fine-tune a model on a new task and it gets worse at what it already did.
You are probably putting the adapter in `q_proj` and `v_proj`, because that is
what everyone does and what the libraries default to.

**On the case we measured, that is the worst place to put it.** Moving the
adapter to the modules where your two tasks interfere least keeps **36.8% more**
of the model's original capability, with the same number of trainable
parameters and no change to your training loop.

TIM finds those modules in minutes and trains nothing.

```bash
pip install torch transformers datasets
MODEL=your/model TASK_A=code TASK_B=prose python experiments/adapter_placement.py
```

---

## What you get back

A ranked table and a module list you can paste straight into your config:

```
BY MODULE TYPE (means)          [abridged; Llama-3-8B, code -> prose]
  type        ceiling    cross      rel
  o_proj                         0.751
  q_proj                         0.683
  k_proj                         0.648
  v_proj                         0.584

SUGGESTED PLACEMENT: the quartile of largest relative drop, 31 modules
  model.layers.22.self_attn.o_proj
  model.layers.25.self_attn.o_proj
  ...
```

Higher `rel` means the two tasks share less there, which means adapting there
costs less forgetting.

## Using it

The module list drops into `peft` as `target_modules`:

```python
import json
from peft import LoraConfig, get_peft_model

ranked = sorted((x for x in json.load(open("placement_code_prose.json"))
                 if not x["saturated"]),
                key=lambda x: -x["rel"])         # highest relative drop first
suggested = [x["module"] for x in ranked][:31]   # the top quartile

config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=suggested,                    # instead of ["q_proj","v_proj"]
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, config)
```

The script already writes the file in ranking order, so the explicit `sorted`
is belt and braces. Rank before you slice either way: taking the first entries
of an unsorted file gives you modules in layer order, not the top of the
ranking.

`peft` accepts full module names, not just type suffixes, so the list works as
is. Nothing else in your pipeline changes.

### How many modules?

We used the top quartile, which on Llama-3-8B is 31 modules: one of the 128
attention projections had a saturated ceiling and was excluded, leaving 127. That was chosen to
match the parameter count of the conventional placement (`q_proj` and `v_proj`
across 16 layers), not because 25% is optimal. If you want a different budget,
take more or fewer from the top of the ranking; we have not measured where the
returns stop.

### Does it work with things other than LoRA?

Probably, and we have not measured it. The criterion uses nothing specific to
low-rank adaptation: it measures where two tasks interfere in the *base* model,
before any adapter exists. Any method that has to pick which modules to touch
(adapters of other kinds, prefix tuning, selective fine-tuning) faces the same
choice and could use the same measurement. What it cannot inform is a method
that touches everything, like full fine-tuning.

If you try it with something other than LoRA, we would like to hear.

### What about the MLP modules?

The default candidate set is the four attention projections, and that is what we
measured. Nothing in the criterion is specific to them, so you can add MLP
modules through `TARGETS` and rank those too. We have not done it, and other
placement work does report MLP modules mattering, so this is a real gap rather
than a settled question.

### What if the ranking comes out differently?

Then use your ranking, not ours. That is the point of running it. Across five
model and task combinations `o_proj` came first and `q_proj` second every time,
but the last two positions swapped in two of them and the size of the gap
between the first two varied by a factor of five across model families, and by
close to twenty across all five combinations. The ordering is yours to measure, not ours to prescribe.

### Two things to check before trusting a comparison

Both of these caught us:

- **Same number of adapted modules in both arms.** Otherwise you are comparing
  capacity, not placement.
- **Both arms should end the first task at similar capability.** If one learns
  the new task worse, its retention ratio is inflated by a ceiling effect and
  the comparison means nothing.

And one seed does not separate much: repeating an identical run in our setting
moves retention by 0.104 in standard deviation.

## Why it works

Steele ([arXiv:2603.02224](https://arxiv.org/abs/2603.02224)) reports that
forgetting under LoRA follows a law in the minimum principal angle between the
two tasks' gradient subspaces: the more the tasks share, the more the model
forgets. That relationship is measurable before any training, and it varies
across modules.

So: measure where they share least, adapt there.

Two details make the measurement work, and both are in the
[paper](paper/adapter_placement.pdf):

- The statistic is the **dimension** of overlap, the sum of squared cosines, not
  the leading cosine. The leading cosine saturates at one as soon as two
  subspaces share a single direction, so it cannot tell sharing one from
  sharing eight.
- The reference is a **same-task ceiling**, the overlap between two halves of the
  first task, not a random-subspace null. Two gradients from the same model
  share directions because of the model, not the task; what informs is the drop
  from that ceiling.

## How this compares to other placement methods

Adapter placement is an active question and several groups work on it. What
distinguishes them is **what they measure, when, and what they optimise for**:

| method | signal | timing | optimises |
|---|---|---|---|
| [LoRA ablation](https://arxiv.org/abs/2106.09685) | downstream accuracy | once, offline | new-task accuracy |
| [AdaLoRA](https://arxiv.org/abs/2303.10512) | adapter singular values | during training | new-task accuracy |
| [PLoP](https://arxiv.org/abs/2506.20629) | normalised feature norm | before training | new-task accuracy |
| [FLoE](https://arxiv.org/abs/2506.00495) | Fisher information | before training | new-task accuracy |
| [Aletheia](https://arxiv.org/abs/2604.15351) | gradient magnitude | before training | new-task accuracy |
| [DomLoRA](https://arxiv.org/abs/2605.06183) | dominant module | before training | new-task accuracy |
| [O-LoRA](https://arxiv.org/abs/2310.14152) | past-task subspace | during training | retention |
| [OPLoRA](https://arxiv.org/abs/2510.13003) | top-k singular dirs of W₀ | during training | retention |
| **TIM (this)** | **task-pair gradient overlap** | **before training** | **retention** |

**The placement methods above optimise for learning the new task well and
cheaply.** That is a different question from ours: a module that is ideal for
learning fast is not necessarily one where learning does least damage, and none
of them reports retention.

**The retention methods act during training, not on placement.** They modify the
update inside modules that were already chosen.

### OPLoRA is complementary, not competing

It is the closest work to this one: same problem, similar tooling. The
difference is what gets protected. OPLoRA protects the dominant singular
directions **of the pre-trained weights**, which do not depend on which tasks
you have. We measure where **the two tasks' gradients** overlap, which does.

And they act at different points: OPLoRA changes the update inside chosen
modules, this chooses which modules to instrument. **Doing both is coherent and nobody has tried it**: pick the placement with
TIM, then project updates within it.
If you do, we would like to hear.

### Independent agreement from Amazon

Amazon ran a
[benchmark sweep](https://www.amazon.science/blog/optimizing-lora-target-module-selection-for-efficient-fine-tuning)
of target-module selection on their Nova 2.0 Lite model and found `o_proj`
alone never fails outright, typically lands within a few points of the best
configuration, and is an attractive default.

TIM fills 17 of its 31 slots with `o_proj`, arriving there from task
overlap rather than from a sweep. Two different routes, same destination.

### Why there are no head-to-head numbers here

Because the comparison would not mean anything as things stand. OPLoRA
fine-tunes LLaMA-2 and Qwen2.5 on MetaMathQA and measures forgetting on ARC and
SIQA; PLoP and FLoE report downstream accuracy on their own suites; we measure
HumanEval retention on Llama-3-8B. Different quantities on different models.

A real comparison means re-running everything under one protocol. **That is the
single most useful thing anyone could contribute here**, and we have not done
it.

## What we measured

| placement | modules | retention | absolute | vs. default |
|---|---|---|---|---|
| default (`q`,`v`) | 31 | 54.1% | 0.151 | — |
| **measured** | 31 | **75.6%** | **0.206** | **9/10 seeds, Wilcoxon p=0.006** |

Llama-3-8B, code → prose, ten seeds, plain LoRA in both arms. Per-seed data in
`results/placement.json`.

## What we did not establish

**A second training validation.** We tried Mistral-7B and could not finish: that
base model scores near zero on HumanEval, so there is no denominator for
retention. This is the most useful thing anyone could add, and we have not done
it.

**Whether it stacks with a weight-space regulariser.** We have
a separate method ([arXiv:2608.03887](https://arxiv.org/abs/2608.03887), code at
[OmegaS-LLM](https://github.com/BiomeMakers/OmegaS-LLM)) that penalises
weight-graph structure during training. Adding it on top gives 6/10 and 7/10
seeds on the two measures, both below the noise floor. Positive direction, not
distinguishable. One lesson from that experiment does generalise: **a penalty
strength chosen for one configuration has to be re-derived for another.** We got
a negative result at the wrong strength and the sign flipped on recalibration.

## Run it on your model

**[Report what you get →](../../issues/new?template=replication.yml)**

A report that the ordering comes out differently is as useful to us as one that
confirms it. Five combinations is not enough to know where it stops holding.

## Contents

```
experiments/adapter_placement.py   the diagnostic
results/placement.json             per-seed data behind every number here
paper/adapter_placement.pdf        the writeup
```

## Licence

**Code:** AGPL-3.0 for non-commercial research and academic use, with a
commercial licence available separately. Same structure as the companion
repository [OmegaS-LLM](https://github.com/BiomeMakers/OmegaS-LLM). See
`LICENSE` and `COMMERCIAL-LICENSE.md`.

**Evaluating it inside a company is covered** and you do not need to ask. That
is an explicit exception, clause 1(d) of `LICENSE`: measuring is free, shipping
a model whose placement came from that measurement is not.

**Paper:** CC BY-NC-ND 4.0, the same licence as the companion paper
[arXiv:2608.03887](https://arxiv.org/abs/2608.03887).

**Patent status.** USPTO Patent Pending, Application No. 64/121,656 (filed 29
July 2026).

## Citation

```bibtex
@misc{acedo2026placement,
  title  = {TIM: Adapter Placement by Task Interference Mapping},
  author = {Acedo, Alberto},
  year   = {2026},
  note   = {Preprint. USPTO Patent Pending No. 64/121,656}
}
```
