# TIM Commercial License

*TIM: Task Interference Mapping*

**Copyright (C) 2026 Alberto Acedo / Biome Makers Inc.**
**USPTO Patent Pending, No. 64/121,656** (filed 29 July 2026)

---

## Evaluating it does not need a license

**If you want to run TIM on your own model to see whether the ordering holds,
that is free, it is covered, and you do not need to ask.** This applies inside a
for-profit organization. It is written into `LICENSE` as clause 1(d).

Run the diagnostic, see what comes out, decide whether it matters to you. We
would be glad to hear the result either way, and a report that the ordering
comes out differently is as useful to us as one that confirms it. There is a
reporting template in the repository issues.

What evaluation does not cover: placing an adapter chosen by TIM in a model you
deploy or offer commercially. That is the line, and it is below.

---

## Who Needs a Commercial License

The default license for this repository (AGPL-3.0, see `LICENSE`) permits free
use for **non-commercial academic research and education**, plus the evaluation
exception above.

You require a **separate commercial license** if any of the following apply:

- You integrate TIM into a product or service that generates revenue, directly
  or indirectly.
- You use a placement selected by TIM to train or fine-tune models that are
  deployed in a commercial production environment.
- You offer TIM (or a derivative) as part of a paid API, platform, or ML
  infrastructure service.
- You incorporate TIM into proprietary software and do not wish to release the
  source code of your modifications (as required by AGPL-3.0).

**In plain terms:** measuring is free. Shipping a model whose adapter placement
came from that measurement is not.

---

## What a Commercial License Includes

A commercial license grants you:

1. **Right to use** TIM in commercial products, services, and internal
   corporate systems without the source-disclosure obligation of AGPL-3.0.

2. **Patent license** covering the TIM method, identified in the executed
   agreement, for the scope of use defined therein.

3. **No copyleft obligation**: you are not required to release the source code
   of your products or modifications.

4. **Technical support**: terms and scope to be defined per agreement.

5. **Attribution flexibility**: negotiable terms for how TIM is credited in
   your product documentation.

---

## License Tiers

Commercial licenses are available under the following indicative tiers. Final
terms are subject to negotiation and a signed license agreement.

| Tier | Use Case | Model |
|---|---|---|
| **Startup** | Early-stage companies (<$1M ARR) | Annual fee + revenue share |
| **Scale-up** | Growth companies ($1M-$10M ARR) | Annual fee |
| **Enterprise** | Large organizations (>$10M ARR) | Annual fee + custom terms |
| **Research Partnership** | Corporate R&D with publication rights | Project-based |
| **OEM / Embedded** | Integration into third-party products for redistribution | Per-unit or revenue share |

If you also hold or are negotiating a commercial license for Omega-S
(`OmegaS-LLM`), say so: both methods sit within the same patent application and
can be covered by one agreement.

---

## Research Partnership Program

There is one thing we want more than a license fee, and it is stated plainly in
the paper: **a second training validation on a model we have not tried.** Our
attempt on Mistral-7B could not be completed because that base model scores near
zero on HumanEval, so retention has no denominator. A model with real capability
on the retained task, two arms of ten seeds, same number of adapted modules in
both, would settle it.

Organizations able to run that, or to compare TIM against other placement
methods under a single protocol, may qualify for a **Research Partnership
Agreement**. Under this model:

- Access to TIM under commercial terms is provided at reduced or zero cost
  during the research period.
- Results may be co-published, with co-authorship negotiated based on
  contribution.
- Participating organizations receive preferential terms on subsequent
  commercial licenses.

A negative result is a valid outcome of a partnership and does not affect its
terms.

---

## How to Obtain a License

To inquire about commercial licensing or the Research Partnership Program,
contact the author:

**Email:** acedo@biomemakers.com
**GitHub:** https://github.com/BiomeMakers/TIM-OmegaS

Please include in your inquiry:

- Organization name and size
- Intended use case
- Scale of deployment (approximate number of models, parameters, GPUs)
- Whether you are interested in a standard commercial license or the Research
  Partnership Program

---

## Frequently Asked Questions

**Q: Can I run TIM inside my company to see whether it helps, without a
commercial license?**
A: Yes. That is the evaluation exception in clause 1(d) of `LICENSE`, and it is
deliberate. You need a commercial license once a placement chosen by TIM goes
into a model you deploy or sell.

**Q: Can I publish a paper using TIM without a commercial license?**
A: Yes, provided you cite the original work. This holds for commercial
organizations too when the work is an evaluation: we would rather have the
replication than the fee.

**Q: Does the AGPL-3.0 copyleft obligation apply if I run TIM internally?**
A: AGPL-3.0's network copyleft applies when you provide a service over a
network. The diagnostic itself is normally run internally and offline, which is
covered by the evaluation exception either way.

**Q: The paper says the ordering may not transfer to my model. Am I paying for
something that might not work?**
A: That is exactly why evaluation is free. Measure first.

**Q: What happens to my commercial license if the patent is granted or
rejected?**
A: The software copyright and the AGPL-3.0/commercial dual-license structure are
independent of the patent outcome. The patent covers the *method*; the copyright
covers the *implementation*. Both licenses remain in force regardless of patent
status.

---

*This document does not constitute a license agreement. A valid commercial
license requires a signed written agreement between the licensee and Alberto
Acedo / Biome Makers Inc. (or the designated licensing entity). Jurisdiction: to
be specified in the license agreement.*
