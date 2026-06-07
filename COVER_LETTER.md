# Cover letter — *Nature Communications* submission

*Draft, 2026-06-07. Subject to revision once final figures are locked.*

---

Dear Editor,

We submit our manuscript **"Cross-disaster mapping is a calibration
problem, not a representation problem: four labels recover the full-pool
oracle"** for consideration as an Article in *Nature Communications*.

## What the paper does

For a decade the deep-learning disaster-mapping literature has interpreted
cross-event generalisation failure as a **representation** problem, and
responded with larger backbones, foundation embeddings and multi-modal
fusion. We test an alternative hypothesis — that the dominant cause of
cross-event F1 loss is **calibration drift** of the decision threshold,
with the underlying pixel ranking transferring largely intact — and find
that calibration drift explains the bulk of the deficit. We then quantify
how cheap the calibration fix is: under a strict leave-one-event-out
protocol (10 folds × 10 seeds = 100 paired pairs), a learned active-
calibration policy with a **four-chip label budget** reaches the
full-pool oracle ceiling on F1 (Δ = −0.002, paired *t*-p = 0.42) and
significantly outperforms zero-shot calibration (Δ = +0.015, p = 0.009),
CoreSet active learning (Δ = +0.008, p = 0.024) and a 3-seed ensemble-
uncertainty baseline (Δ = +0.010, p = 0.003).

## Why we believe it fits *Nature Communications*

1. **Discriminative scientific question.** We frame the work as a head-
   to-head test of two competing mechanistic hypotheses (H1
   representation drift vs. H2 calibration drift) and structure the
   Results section around three independent falsification tests. The
   paper's central deliverable is the *answer to the question*, not the
   method by which we ask it.
2. **Breadth of evidence.** The H2-dominates finding is supported across
   12 real events from two independent public benchmarks
   (Sen1Floods11 floods and xBD building damage), two backbones (a
   trained U-Net and a frozen Google AlphaEarth foundation model), and
   under three statistical protocols (within-event paired,
   leave-one-event-out 10-fold, and a leave-one-hazard-out cross-hazard
   protocol). To our knowledge this is the first apples-to-apples
   comparison of an Earth-observation foundation model against a
   same-input U-Net on cross-event flood mapping under leakage-free
   evaluation.
3. **Operational consequence.** The finding is directly actionable.
   Operational disaster-response services (Copernicus EMS Rapid Mapping
   typically delivers 1-to-3-day cycles) can deploy near-oracle
   calibration on a new event in *minutes* with four labels, with no
   model retraining. The end-to-end agent we ship runs at 0.031 s per
   chip on a single GPU and is fully reproducible (28 paper-grade
   figures, ~50 result JSONs, live auto-updating dashboard at
   `https://geodisaster-fm.pages.dev/`).
4. **Calibrated negative findings.** We present a deliberate panel of
   negative results — foundation embeddings on equal inputs are
   comparable but not superior to U-Net; a 3-seed ensemble-uncertainty
   chip-selection baseline is significantly worse than a learned
   policy; richer 10-d chip features hurt the learned policy; an
   earlier within-event protocol over-stated PPO gains and we correct
   the numbers in full — that strengthen, rather than dilute, the H2
   claim. We believe this transparency is exactly the editorial
   standard *Nature Communications* asks for.

## Why we did not submit to *Nature* flagship

We considered direct submission to *Nature*. The work reframes a
deep-learning sub-field (cross-disaster mapping) and quantifies an
operational deliverable, but the empirical scope is currently bounded
to the disaster-mapping community (two benchmarks, twelve events, two
backbones). We believe *Nature Communications* — whose breadth-of-
impact criterion encompasses the broadly significant advance to a
specific scientific community that this paper makes — is the natural
home, and that *Nature*-flagship would require a wider hazard /
backbone spread that we explicitly mark as future work in the
Limitations.

## Suggested reviewers (with non-conflict reasoning)

*To be added in consultation with the corresponding authors. Candidate
domains: cross-domain segmentation methodology, Earth-observation
foundation model evaluation, active learning for remote sensing,
post-hoc calibration of deep classifiers, operational disaster
response.*

## Conflicts

No competing interests to declare.

## Reproducibility

All code, data manifests, intermediate result JSONs, paper-grade
figures, and the live auto-updating agent dashboard are public:

* GitHub: <https://github.com/14H034160212/geodisaster-fm>
* Live dashboard: <https://geodisaster-fm.pages.dev/>
* Advisor-facing progress report: <https://geodisaster-fm.pages.dev/report.html>

The leave-one-event-out experiment can be reproduced end-to-end from
raw inputs by running `scripts/cache_all10_chips.py`,
`scripts/cache_ensemble_uncertainty.py`, then
`scripts/eval_ppo_meta_train_test.py` once per fold and
`scripts/aggregate_loeo.py` for the headline table.

We thank you for your consideration.

Sincerely,

Qiming Bao, Yanbing Bai (corresponding author)

*[Affiliations and contact details to be filled in at submission.]*
