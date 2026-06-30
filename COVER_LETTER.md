# Cover letter — *Nature Communications* submission

*Draft v2, 2026-06-13. Affiliations + suggested reviewers to be filled
before submission.*

---

Dear Editor,

We submit our manuscript **"Cross-disaster mapping is largely a
calibration problem, not a representation problem: four labels recover
99 % of the full-pool oracle, regardless of how those labels are
chosen"** for consideration as an Article in *Nature Communications*.

## What the paper does

For a decade the deep-learning disaster-mapping literature has
interpreted cross-event generalisation failure as a **representation**
problem, and responded with larger backbones, foundation embeddings and
multi-modal fusion. We test the alternative hypothesis — that the
dominant cause is **calibration drift** of the decision threshold, with
the pixel ranking transferring largely intact — with four independent
falsification experiments across three public benchmarks (floods,
building damage, wildfires; ≥ 18 real events) and four backbones (a
U-Net and three frozen foundation models: AlphaEarth, Prithvi, DOFA):

1. **Ranking transfers; thresholds do not.** 15 of 18 event-optimal
   thresholds differ from the 0.5 default; re-fitting one threshold
   recovers up to +0.235 F1 on a single event.
2. **Representation does not close the gap — three times.** No
   foundation model exceeds the from-scratch U-Net; Prithvi, pre-trained
   on the wildfire benchmark's own HLS modality, is slightly *worse* on
   every held-out fire season, and the calibration drift the three
   backbones exhibit rises as their task-match weakens.
3. **Four labels recover 99 % of the full-pool oracle** under a strict
   leave-one-event-out protocol (200 paired pairs), and the entire
   family of practical 4-chip selection methods (random, entropy,
   CoreSet, ensemble uncertainty, learned policy) spans only 0.017 F1.
4. **And the labels are necessary — the drift is not pure label shift.**
   The two standard zero-label corrections from the label-shift
   literature both fail under the same protocol: Saerens EM diverges
   (−0.61 F1 versus doing nothing) and BBSE, despite estimating the
   new-event priors nearly correctly, still produces a corrected
   threshold *worse than no correction* (−0.14 F1). The
   class-conditional score distributions themselves distort under
   cross-event transfer; four labels measure that distortion, which no
   zero-label method can see. This decomposes cross-disaster drift into
   prior shift (free to fix) plus score distortion (four labels to fix)
   — to our knowledge the first such decomposition on operational
   disaster benchmarks.

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
3. **Foreseeable real-world economic and societal benefit — not just a
   test-set gain.** The finding is directly actionable and the payoff is
   quantifiable in operational, not accuracy-point, terms. (i) *Annotation
   labour*, the dominant human cost in rapid mapping, falls roughly
   ten-fold: from the tens of expert-annotated chips per event that
   full-pool calibration needs to four chips, while retaining ≈ 99 % of
   the full-pool F1. (ii) *Time-to-first-map* collapses — operational
   services such as Copernicus EMS Rapid Mapping target 24 h to first
   delivery and 1-to-3-day actionable cycles, whereas our per-event
   learning is a single threshold fit from four chips (0.031 s per chip
   of machine time), moving the actionable map well inside the first-72-
   hour window in which evacuation and aid-allocation decisions are made.
   (iii) *Marginal coverage cost is near-zero and hazard-independent*:
   the perception model is frozen and reused, so each new event costs
   four labels and no retraining — demonstrated across floods, building
   damage and wildfire — which is the property that makes global,
   all-hazard coverage economically feasible rather than a bespoke
   per-event project. These savings feed decision-level outputs that
   matter operationally (flooded-area totals track analyst labels at
   r = 0.971; on the USA test event, 2.4 % of buildings and 6.6 % of
   road-kilometres flagged as affected — the exposure figures that size a
   response). The end-to-end agent we ship runs at 0.031 s per chip on a
   single GPU and is fully reproducible (28 paper-grade figures, ~50
   result JSONs, live auto-updating dashboard at
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
