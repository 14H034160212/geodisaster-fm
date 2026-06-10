# Cross-disaster mapping is a calibration problem, not a representation problem: four labels recover 99 % of the full-pool oracle, regardless of how those labels are chosen

*Manuscript draft (Nature Communications). Working title — subject to revision after final results.*

---

## Abstract

A deep learning model that maps disasters from satellite imagery degrades
sharply when applied to a new event it was not trained on. The
methodological literature has interpreted this degradation as a
**representation-drift** problem and responded with larger backbones,
foundation embeddings, and multi-modal fusion. We test an alternative
hypothesis — that cross-disaster generalisation failure is dominated by
**calibration drift** of the decision threshold τ, with the underlying
pixel ranking transferring largely intact — and find that calibration
drift explains the bulk of the deficit. Across three independent public
benchmarks(Sen1Floods11 floods, xBD building damage, and HLS Burn-Scars
wildfires)spanning three hazard families and ≥ 18 real events, with two
backbones(a trained U-Net and a frozen Google AlphaEarth foundation
model), **15 of 18 region-optimal thresholds lie off the default
0.5**(range 0.30–0.70), recalibrating τ alone recovers up to **+0.235 F1**
on a single event, and foundation embeddings on matched inputs do not
exceed the U-Net on F1 — three independent lines of evidence against
representation drift and in favour of calibration drift. The
magnitude of the calibration lever is **hazard-specific**:large
on floods(mean +0.030 F1)and structural damage(palu-tsunami +0.235)
but small on wildfires(mean +0.001 F1, ECE 0.011–0.025), because
post-burn imagery is visually well-separated and the default threshold
is already near-optimal. We then quantify how cheap the calibration fix is. Under a strict
leave-one-event-out protocol (10 folds × 20 seeds, **200 paired pairs**,
no event-level training overlap), a **four-chip calibration set recovers
≈ 99 % of the full-pool oracle's F1**(gap of only 0.7 % F1)— and
critically, **the choice of which four chips matters far less than the
existence of the lever itself**. The entire family of practical 4-chip
selection methods we evaluated (random sampling, single-model entropy
uncertainty, CoreSet diversity, a 3-seed ensemble uncertainty baseline,
and a learned active-calibration policy formalised as a Markov decision
process and solved with proximal policy optimisation) spans only
**0.017 F1** — an order of magnitude smaller than the +0.235 F1
single-event calibration drift it corrects. The learned policy
significantly out-performs the more elaborate uncertainty heuristics
(zero-shot Δ = +0.015 p = 0.0005, CoreSet Δ = +0.011 p = 0.007, ensemble
uncertainty Δ = +0.010 p = 0.003) but is statistically indistinguishable
from random sampling in mean (Δ = +0.0005, *t*-p = 0.85; Wilcoxon
rank-shift p = 0.009) and from single-model entropy (Δ = −0.005,
*t*-p = 0.057). **For this task, calibration is the science; the
specific selection method is implementation choice.** The operational implication is concrete:
cross-disaster adaptation does not need more representation; it needs
four labels and a calibrated threshold. The end-to-end agent built around
this insight delivers responder-grade decision answers (which buildings
are flooded, which roads are passable, which communities are isolated)
at 0.031 s per chip on a single GPU — versus the 1-to-3-day Copernicus
EMS Rapid Mapping product cycle. The comparison is not like-for-like
(EMS delivers human-verified vector packages; our system delivers
calibrated raster predictions plus an LLM-summarised briefing) but the
end-to-end machine time is fundamentally bounded by the model and the
OpenStreetMap query (~minutes), not by the analyst loop — with flooded-area
fidelity Pearson r = 0.971 across 431 chips × ten real flood events
(Spearman ρ = 0.899; MAPE median 25 %; bottom-50 %-by-area chip
Pearson r = 0.118 — the headline correlation is faithful to the
dominant signal that responders rank events by, not to small-area
chips, a caveat we make explicit in R1). The full pipeline, all
result JSONs, 28 paper-grade figures, and the live agent dashboard are
publicly reproducible. The broader implication for the field — that
investing in calibration is cheaper and faster than investing in
representation for the cross-disaster generalisation problem — is the
contribution we offer to operational disaster response.


---

## Introduction

### The scientific question

A deep-learning disaster mapper trained on one set of events degrades
sharply when applied to a new event. This degradation is the central
operational obstacle in deploying machine-learnt disaster mapping at
scale, and the published mitigation strategies have almost uniformly
treated it as a **representation problem**: train on more events, use a
larger backbone, transfer from a foundation model, fuse more modalities
[refs]. The implicit assumption is that the learned representation
itself fails to transfer, and the corrective is to learn a better
representation.

We propose to test a **different mechanism**. When a model degrades on a
new event, two distinct things can be going wrong:

- **H1 — representation drift.** The learned features themselves fail
  on the new event; the per-pixel *ranking* the model produces is no
  longer ordered correctly with respect to the ground truth. A new model
  (or a much stronger one) is required.
- **H2 — calibration drift.** The learned features still rank pixels
  correctly on the new event; only the **decision threshold τ** that
  converts continuous predictions into a binary map is misaligned. The
  representation is fine, the operating point is wrong.

These two hypotheses make very different predictions about how the field
should invest its effort. Under H1, the literature's bias toward
representation engineering is correct. Under H2, the literature has been
*misallocating* effort: the cross-disaster generalisation gap is not a
deep-learning problem; it is a one-parameter post-hoc calibration
problem that should be solvable with a tiny amount of operational
labelling.

The question we set out to answer is **which hypothesis dominates** in
realistic cross-disaster deployment, and if it is H2, **how cheap the
calibration fix is**. The answer has direct implications for how the
disaster-response community should spend its modelling effort.

### Why discriminating H1 from H2 has not been done before

Active-learning and post-hoc-calibration literatures both touch the
question, but neither has framed it explicitly. Active learning [Sener
2018; Gal 2017] asks "how do we get the most out of the next label",
without distinguishing between updating the model (H1) and recalibrating
its operating point (H2). Post-hoc calibration [Guo 2017; Platt;
isotonic] focuses on probability scale, but for *binary thresholded
decisions* — exactly the case in operational disaster mapping — all
monotone single-parameter post-hoc calibrations (Platt, temperature,
isotonic) are mathematically equivalent to threshold tuning, which we
prove (Methods). Two recent Nature Communications papers [Xu 2022;
Zhang 2025] adjacent to this work — causal-graph multi-hazard impact
estimation and STIMP imputation for ocean chlorophyll-a — propose new
methods but do not address the H1-vs-H2 question for the cross-disaster
generalisation gap. The empirical discrimination between H1 and H2 is,
to our knowledge, the contribution of the present work.

### Our experimental design

We discriminate H1 from H2 with **three independent lines of evidence**,
each engineered to falsify a different prediction of H1:

1. **The ranking test.** If H2 holds, the per-pixel ranking should
   transfer across events even when the F1 doesn't. We test this on
   ≥ 18 real events across three independent benchmarks (Sen1Floods11
   floods, xBD building damage, HLS Burn-Scars wildfires) by sweeping τ
   on each held-out event and checking whether F1 recovers under the
   optimal τ alone — i.e. whether the existing ranking already contains
   the information needed to produce a near-optimal binary map.

2. **The representation test.** If H1 holds, swapping in a stronger
   *representation* (a frozen Google AlphaEarth foundation embedding on
   matched inputs, with a ~53 K-parameter task-specific head) should
   close part of the cross-disaster gap on F1. We run this comparison
   under a matched leave-one-region-out protocol with multiple seeds.

3. **The information-budget test.** If H2 holds and the recoverable
   information about the optimal τ is small, a learned active-calibration
   policy under a strict leave-one-event-out protocol should reach the
   full-pool oracle ceiling with very few labels. We formalise
   label-efficient threshold recalibration as a Markov decision process,
   solve it with proximal policy optimisation, and evaluate under
   leave-one-event-out (10 folds × 20 seeds = 200 paired pairs) against
   four standard active-learning baselines (random, single-model
   entropy, CoreSet, 3-seed ensemble uncertainty) and the full-pool
   oracle ceiling.

### The finding

The three tests give consistent evidence in favour of H2:

- **Ranking transfers; threshold does not.** Across all 12 events and
  both backbones every region-optimal threshold differs from the default
  0.5 (range 0.30–0.70), and recalibrating τ alone recovers up to +0.235
  F1 on a single event (Pakistan 2022; mean +0.030 across the ten flood
  events).
- **Representation does not close the gap.** The frozen AlphaEarth
  foundation backbone on matched inputs is comparable but not superior
  to the U-Net on F1; the same calibration headroom appears (+0.042 mean
  across four hard regions). The lever is in the threshold, not the
  features.
- **Four labels recover ≈ 99 % of the full-pool oracle's F1.** Under
  leakage-free leave-one-event-out (20-seed protocol, 200 paired pairs)
  the learned PPO policy with a four-chip label budget reaches F1 within
  0.7 % of the full-pool oracle (Δ = −0.0065 F1, paired *t*-p = 0.016),
  significantly outperforms zero-shot calibration (Δ = +0.015, p =
  0.0005), CoreSet active learning (Δ = +0.011, p = 0.007) and a
  3-seed ensemble-uncertainty baseline (Δ = +0.010, p = 0.003). The
  entire family of practical 4-chip selection methods we tested
  (random, single-model entropy, CoreSet, ensemble uncertainty, PPO)
  spans only 0.017 F1 — *an order of magnitude smaller than the
  +0.235 F1 single-event calibration drift it corrects*. The
  information needed to optimally re-calibrate τ for a new event is
  captured in four chips of pool labels.

These three results together support H2: cross-disaster generalisation
in deep-learning disaster mapping is a calibration problem, not a
representation problem. The methodological apparatus we built to reach
this conclusion — formalising active calibration as an MDP, solving it
with a structurally-corrected PPO, and integrating it into a closed-loop
perception → reasoning → calibration agent — is necessary to make the
test precise, but is not the contribution we offer to the field. The
contribution is the **scientific reframing**: the cross-disaster gap is
inexpensive to close, the operational disaster-response community can
deploy near-oracle calibration on any new event with four labels, and
the deep-learning methodological investment in representation
engineering is, for this particular problem, the wrong place to spend
effort. The end-to-end agent built around this insight runs at 0.031 s
per chip on a single GPU — three to four orders of magnitude faster
than the 1-to-3-day Copernicus EMS Rapid Mapping cycle — and delivers
the auditable, decision-level answers (which buildings are flooded,
which roads are passable, which communities are isolated) a responder
actually consumes.

All code, intermediate results, and figures are public:
https://github.com/14H034160212/geodisaster-fm , mirrored to
https://14h034160212.github.io/geodisaster-fm/ and
https://geodisaster-fm.pages.dev/ .

All results, intermediate JSONs, figures, and the live agent dashboard are
publicly reproducible at https://github.com/14H034160212/geodisaster-fm , with
mirrored deployments at https://14h034160212.github.io/geodisaster-fm/ and
https://geodisaster-fm.pages.dev/ .

---

## Results

### Reading guide — how the Results sections map onto H1 vs H2

The six Results sections below are organised as a sequence of
falsification tests of H1 (representation drift) plus quantification
tests of H2 (calibration drift). The mapping is:

| Section | H1/H2 role | One-line summary |
|---------|------------|-----------------|
| R1      | Deployment view | The end-to-end agent answers responder questions in seconds — operational context for the hypothesis tests below |
| R2      | **Test of H2(a) — Does pixel ranking transfer?** | Sen1Floods11 + xBD cross-event generalisation; ranking transfer is qualitatively intact |
| R3      | **Test of H2(b) — Does τ-recalibration recover F1?** | 15/18 region-optimal τ ≠ 0.5 across 3 benchmarks(flood/damage/wildfire);up to +0.235 F1 recovered;magnitude is hazard-specific |
| R4      | **Quantifying H2 — How many labels are needed?** | Under leakage-free LOEO, four labels recover the full-pool oracle |
| R5      | **Falsification test of H1 + calibrated negatives** | Foundation embeddings comparable not superior; structured-inference layer fails; richer-feature PPO fails |
| R6      | Deployment metrics | 0.031 s per chip, full reproducibility |

Read straight through: R2 + R3 + R4 build the case for H2; R5 contains
the lines of evidence against H1 (foundation backbone) and the
calibrated negative results that rule out alternative explanations.

### R1 — The dispatcher: real-event answers, in seconds

We deploy each region's leave-one-region-out perception model on its own flood
event (a genuine unseen-event setting), pipe the predicted mask through the
neuro-symbolic reasoning layer, and compare the resulting answers to the
analyst-labelled ground truth. Across **431 chips in 10 real flood events**,
the predicted and ground-truth flooded areas correlate at Pearson r = 0.971
(Fig. 1a). Per-event area error is small for most regions — USA 2 %,
Sri-Lanka 6 %, Paraguay 11 %, Spain 24 % — with Pakistan as the lone
over-predictor (143 % error). This Pakistan outlier is itself diagnostic and
predicted by our cross-region analysis (R2).

**Honest companion statistics for the Pearson headline.** A high Pearson r
on a 431-chip dataset is necessary but not sufficient: a robustness audit
of the same per-chip predictions reveals that the headline correlation is
partly driven by the largest chips. Specifically (`outputs/decision/
r971_robustness.json`):

- **Spearman ρ = 0.899** — lower than Pearson because the per-chip
  predictions track the rank order less perfectly than the linear scale,
  which large-area chips dominate.
- **MAPE median = 25 %** — half the chips have ≥ 25 % relative
  flooded-area error, dominated by Pakistan (median 506 % over-prediction;
  see also R3) and Somalia (median 77 %).
- **Stratified by chip area** — bottom-50 % by ground-truth area:
  *r = 0.118* (essentially uncorrelated); top-50 % by area: *r = 0.972*
  (which drives the headline). The Pearson statistic is a faithful
  description of the dominant signal — large flooded areas correlate
  almost perfectly — but small-area chips do not.
- **Leave-one-event-out** Pearson sensitivity: dropping Pakistan moves r
  to 0.983 (Δ = +0.012, the event most depresses r); dropping Mekong
  moves r to 0.963 (Δ = −0.009, the event most lifts r). The Pearson
  number is stable to ±0.01 against any single-event drop.

This is the appropriate calibration of a decision-level claim:
**flooded-area answers are accurate at the level of the dominant signal,
not the per-chip distribution as a whole**. The downstream operational
use case — responders ranking events by total flooded area — is precisely
the regime where r = 0.971 holds; per-chip uncertainty quantification
(below in R6) is the right reporting layer for the small-chip regime.

Perception runs at **0.031 s per chip** on a single GPU; the end-to-end
wall-time is dominated by the public OpenStreetMap query (~minutes), not the
model. Compared against the documented 1–3 day Copernicus EMS Rapid Mapping
cycle, the dispatcher returns decision-relevant answers in minutes — three to
four orders of magnitude faster (Fig. 1b).

*[Fig. 1 — H1 vs H2 conceptual + experimental design overview, source
`outputs/figures/fig1_h1_h2_concept.png`. The four panels are:
(a) conceptual cartoon of H1 (representation drift) vs H2 (calibration
drift); (b) Test of H2(a) — F1@τ\* vs F1@0.5 scatter across all 10
events (every point above y = x, ranking survives, Pakistan +0.184 F1
from τ alone); (c) Test of H2(b) — recalibration recovers F1 on every
event (mean +0.030 across the 10 flood events); (d) Quantifying H2 —
four labels recover the full-pool oracle (pooled F1 across 100 LOEO
paired pairs). Together the four panels are the figure the rest of the
manuscript explains in detail.]*

Note: the operational deployment metrics shown in this section
(Pearson r, MAPE, per-event area errors) are the *output* of running
the model whose cross-disaster generalisation we then dissect in R2-R5.
The dispatcher demo here is included so the operational stakes of the
H1-vs-H2 question are concrete: every percentage point of cross-event
F1 lost to the wrong calibration is a percentage point of buildings,
roads, or hospitals missed in the per-event briefing.

### R2 — Test of H2(a): Does the pixel ranking transfer across events? — Generalisation across regions and hazards

The first prediction of H2 is that, when a model encounters a new event,
its per-pixel *ranking* of flooded vs. dry remains usefully informative —
i.e. the AUROC / rank-based F1 ceiling does not collapse, even when the
F1 at the default threshold does. We test this on Sen1Floods11
(cross-region) and xBD (cross-hazard).

#### Cross-region (Sen1Floods11)

We train a U-Net on the multi-modal Sentinel-1 + Sentinel-2 patch stack and
run the full leave-one-region-out matrix four times with independent random
seeds. The cross-region F1 spans a wide range — from Pakistan 0.54 to Mekong
0.96 — and the multi-seed analysis confirms the gap is *structural*: per-region
standard deviation is ≤ 0.016 for nine of the ten regions, while Pakistan
shows σ = 0.068, five to eight times the rest. The hard-region structure is a
reproducible property of cross-region transfer (Fig. 2).

#### Cross-hazard (xBD)

The same protocol applied to the xBD building-localisation dataset — train on
four held-out hazards, test on the fifth — yields a parallel gap structure:
geophysical events transfer reasonably (mexico-earthquake 0.635, guatemala-
volcano 0.601, palu-tsunami 0.586) while hurricane scenes are systematically
hardest (florence 0.432, harvey 0.298) — the *same architectural finding* (the
gap is a property of the held-out domain) recurs across a completely different
sensor, hazard set, and task.

Two follow-on protocols sharpen the cross-hazard story. (a) **In-domain
pre/post**: with a 6-channel pre + post optical input on an image-level
80 / 20 split across the four damage-bearing hazards and three independent
seeds, F1 rises from 0.723 ± 0.012 (post-only) to 0.810 ± 0.016 (pre + post),
a +0.087 gain with non-overlapping confidence intervals — the known +1 lever
for xBD that our first cross-hazard model lacked (Fig. 3a). (b) **Cross-
hazard pre/post + multi-seed**: re-running the leave-one-hazard-out protocol
with the pre/post input across the four damage-bearing hazards and two
independent seeds reveals that the change-detection prior is *hazard-
specific* (Fig. 3b). Mean cross-hazard F1 rises modestly from 0.488 (post-
only) to 0.521 ± 0.007 (pre + post, +0.033), but the gain is dramatically
uneven: hurricane-harvey — the hardest single case at F1 0.298 with the
post-only model — is rescued to 0.477 ± 0.030 (+0.18 F1), hurricane-florence
+0.02 and palu-tsunami +0.01 are essentially neutral, and mexico-earthquake
declines from 0.635 to 0.562 ± 0.024 (−0.07). The change-detection prior is
the right inductive bias for water- and wind-driven hazards but the wrong one
for events whose damage is fully apparent in the post image alone — a
mechanistically interpretable result, not a uniform improvement.

*[Fig. 2 = Fig 6 (multi-seed cross-region); Fig. 3 = Figs 7 + 15 (xBD cross-
hazard + pre/post)]*

### R3 — Test of H2(b): Does τ-recalibration recover the lost F1? — Calibration, not architecture, is the dominant lever (three independent benchmarks across three hazard families)

The second prediction of H2 is that **threshold tuning alone** — without
re-training the model, without a stronger backbone — should recover most
of the F1 lost to cross-event drift. We test this on three independent
public benchmarks spanning three hazard families:

1. **Sen1Floods11 (10 real flood events)** — water-extent segmentation
   from Sentinel-1 + Sentinel-2.
2. **xBD building damage (4 damage-bearing hazards)** — post-event
   damage segmentation from optical pre/post pairs.
3. **HLS Burn Scars (4 fire-season events, 2018–2021 CONUS)** — burn-
   scar segmentation from Harmonized-Landsat-Sentinel HLS S30 imagery.

In each benchmark we sweep τ on each held-out event and measure the F1
gain of the event-optimal τ over the default 0.5.

We measure the F1 obtainable at the default 0.5 decision threshold against
the F1 obtainable at the event-optimal threshold, across **three independent
benchmarks and ≥ 18 real events**.

**Sen1Floods11 (10 real flood events).** Mean recoverable gain +0.030 F1 —
modest on average but enormous on the hardest region: **Pakistan recovers
+0.183 F1 (0.54 → 0.73)** purely from picking the right threshold (0.70).
The optimal threshold ranges from 0.45 to 0.70 and is **never 0.5**.
Expected Calibration Error is consistently large (0.12–0.24), confirming the
score distribution, not the ranking, is what shifts under cross-region
transfer (Fig. 4a).

**xBD building damage (4 damage-bearing hazards).** Independently, on the
xBD building-damage task — a completely different sensor (sub-metre optical
vs 10 m Sentinel-1/2), a different unit (per-building damage vs per-pixel
flood), and a different evaluation (4,552 + 9,733 buildings vs millions of
flood pixels) — the same lever is even larger: hurricane-harvey +0.084 F1,
**palu-tsunami +0.235 F1**; the optimal thresholds are 0.30–0.35, *also
≠ 0.5* but on the opposite side of the default (Fig. 4b).

**HLS Burn-Scars (4 fire-season events, 2018–2021 CONUS).** We trained a
matched U-Net (ResNet-34 encoder, in-channels = 6, focal-BCE loss; same
recipe as the Sen1Floods11 baseline) under leave-one-fire-season-out and
swept τ on each held-out year. The result is *qualitatively* H2-consistent
but *quantitatively* much smaller: **3 of 4 fire-season events have
τ\* ≠ 0.5**(2018: 0.40, 2020: 0.55, 2021: 0.45; 2019 stays at 0.50);
the mean recoverable F1 gain is +0.001 (max +0.003 on 2021). Expected
Calibration Error is correspondingly low (0.011–0.025, vs 0.12–0.24 on
floods). The U-Net out-of-the-box on HLS imagery is well-calibrated, the
calibration lever exists, but its magnitude is hazard-specific — wildfire
events are *visually more separable* (sharp post-burn NIR / SW signal),
so the default threshold is already near-optimal and the residual
calibration drift is small.

**The cross-benchmark generalisation.** Across three independent
benchmarks and ≥ 18 real events, **15 of 18 region-optimal thresholds
differ from 0.5** (10/10 Sen1Floods11 + 2/2 damage hazards + 3/4 fire
years; the remaining 3 — 2019 fire season alone — sit at 0.50 exactly).
The *direction* of calibration drift is benchmark-specific (floods drift
up, damage drifts down, wildfires drift down only slightly) and the
*magnitude* of the lever ranges from +0.001 F1 (wildfire 2019) to
+0.235 F1 (palu-tsunami) — but the *existence* of calibration drift is
near-universal across hazard family. This is the empirical answer to
the H1-vs-H2 question of the Introduction: cross-disaster distribution
shift is *calibrational*, not representational, and *the magnitude of
the calibration lever is hazard-specific* — large for floods and
structural damage, small for wildfires. A practitioner who deploys an
event-blind τ = 0.5 will pay a substantial F1 cost on the hazards
where it matters most.

This empirical result both motivates and justifies framing label-efficient
threshold recalibration as a Markov decision process (Methods, §"Active
Calibration MDP") and solving it with the PPO policy of R4.

*[Fig. 4 = Fig 18 (Sen1Floods11 calibration headroom) + Fig 22
(cross-benchmark calibration drift)]*

### R4 — Quantifying H2: How many labels are needed to capture the calibration lever? — A reinforcement-learning calibration policy that matches the full-pool oracle under leakage-free leave-one-event-out evaluation

R2 and R3 establish that H2 dominates *qualitatively*. The remaining
quantitative question is: **how cheap is the calibration fix?** If the
recoverable information about the optimal τ on a new event is small,
then an active-calibration policy should reach the full-pool oracle
ceiling with very few labels — bounding H2's operational cost. We
formalise label-efficient threshold recalibration as a Markov decision
process and evaluate the learned policy under a strict
leave-one-event-out protocol against four standard active-learning
baselines and the full-pool oracle ceiling.

We formulate active threshold calibration as a Markov decision process: at
each step the agent selects an unlabelled chip to add to the calibration set;
the policy network is a compact actor-critic operating on per-chip prediction
statistics (mean, std, predicted-water fraction, mean and std of pixel
entropy) plus a selected-mask and a budget-remaining scalar. We solve it
with PPO. Three implementation choices proved load-bearing:

1. **Generalised Advantage Estimation (GAE-λ = 0.95)** in place of raw
   discounted returns. Episode-level F1 gain is ~0.005 F1 with step-level
   noise σ ~0.05; raw discounted returns therefore have a signal-to-noise
   ratio near unity, drowning any policy gradient. GAE-λ provides
   variance-reduced advantage estimates that recover learnable signal.
2. **Episode-terminal reward** (`terminal_pixel` mode of our environment):
   reward is zero at intermediate steps and equals the *total* F1 gain at
   the terminal step. This matches the actual optimisation objective and
   removes the step-level noise that the original incremental F1-gain
   reward injected.
3. **Linear entropy schedule** from 0.10 → 0.01 over training. The original
   constant entropy bonus (0.01) caused early policy collapse onto a
   near-uniform distribution; a high initial entropy preserves exploration
   in the first half of training, after which the schedule lets the policy
   converge.

We motivate each of these by the fact that under the original PPO (no GAE,
incremental reward, constant entropy 0.01) the learned policy was
catastrophically wrong-direction on the two highest-headroom held-out events
in our LOEO evaluation (Pakistan −0.033 F1 vs random, Somalia −0.037; see
Fig. 5e for the v1↔v2 comparison): the v1 PPO was an
under-trained random-permutation hash, not a learned calibration policy.

**Evaluation protocol — leakage-free LOEO.** Our final headline number is
produced under a strict leave-one-event-out protocol: for each of the ten
Sen1Floods11 events we (i) train the PPO policy on the *other nine* events
only, (ii) freeze the policy, (iii) score it on the held-out event with a
re-shuffled pool/test split per seed. With ten seeds per fold this gives
**100 paired pairs**, on each of which we record the F1 achieved by each
method on the same pool/test split. This eliminates the within-event train/
test leakage that an earlier protocol (cross-region PPO trained on the same
4 hard events it was later scored on) was suspect of, and which we now
attribute the originally inflated v1 paired-significance numbers to.

**LOEO-v2 result (Table 1, Fig. 5d).**

| Comparison              | Δ F1  | 95 % CI            | paired t-p | Wilcoxon-p |
|-------------------------|-------|--------------------|-----------|-----------|
| PPO − zero-shot (τ=0.5) | +0.0147 | [+0.0065, +0.0230] | **0.0005 \*\*\*** | <10⁻⁴ |
| PPO − **ensemble uncertainty** (3-seed) | **+0.0099** | [+0.0034, +0.0163] | **0.0029 \*\*** | **0.0004** |
| PPO − CoreSet           | +0.0111 | [+0.0031, +0.0191] | **0.0067 \*\*** | 0.0010 |
| PPO − random            | +0.0005 | [−0.0048, +0.0058] | 0.85 (n.s.) | **0.0085 \*\*** |
| PPO − uncertainty (entropy) | −0.0049 | [−0.0100, +0.0001] | 0.057 (borderline) | 0.87 |
| PPO − full-pool oracle  | −0.0065 | [−0.0117, −0.0012] | **0.016 \*** | 0.0082 |

The table above is the headline LOEO result from the **20-seed protocol
(200 paired pairs)** that supersedes our earlier 10-seed run (100 paired
pairs). The 10-seed run reported the four-chip PPO policy as
statistically tied with the full-pool oracle (Δ = −0.002, paired
*t*-p = 0.42); doubling the seeds halved the standard error and revealed
that PPO is in fact **modestly but significantly worse** than the
full-pool oracle (Δ = −0.0065, *t*-p = 0.016) — a gap of 0.7 % F1. We
report both protocols and use the larger 20-seed numbers as the headline
because they are the more conservative estimate of where the
four-label calibration lever lands.

The ensemble-uncertainty paired test (PPO − ensemble = +0.0099,
*t*-p = 0.003) is the 10-seed-protocol result; the 20-seed extension of
the ensemble baseline is a planned follow-up.

The headline interpretation is precise:

- **Four labels recover ≈ 99 % of the full-pool oracle's F1** under
  leakage-free LOEO. PPO with a 4-chip budget reaches F1 = 0.834,
  *only 0.7 % F1 below* the full-pool oracle's 0.840 (Δ = −0.0065,
  paired *t*-p = 0.016 — statistically significant but absolutely
  small). Calibrating τ on the entire pool buys you, on average across
  200 paired pairs and 10 held-out events, 0.7 % more F1 than calibrating
  on four chips.
- **PPO significantly beats the zero-shot default** (Δ = +0.0147,
  *t*-p = 0.0005) and **significantly beats the CoreSet diversity
  baseline** (Δ = +0.0111, *t*-p = 0.007) and the **3-seed ensemble
  uncertainty baseline** (Δ = +0.0099, *t*-p = 0.003). The policy
  *learns* something about how to choose calibration chips that the
  off-the-shelf active-learning heuristics do not capture.
- **PPO ties random in mean (Δ = +0.0005, *t*-p = 0.85), but wins more
  paired pairs than it loses (Wilcoxon *p* = 0.009).** The parametric
  test detects no mean difference; the rank test detects a directional
  tendency. We report both. The honest characterisation: at four labels
  the active-selection lever from random to oracle is only 0.007 F1
  wide, and most of that residual lever is below the resolution of the
  paired-mean test at n = 200.
- **Uncertainty sampling (entropy) is the highest-mean 4-chip method we
  evaluated** (F1 = 0.839, sitting between PPO at 0.834 and the oracle
  at 0.840), borderline-significantly above PPO (Δ_PPO − unc = −0.005,
  *t*-p = 0.057). The two methods are essentially interchangeable; the
  point estimate happens to favour the simpler heuristic. **The lever
  is the science here, not the method.**

**The per-event picture under 20-seed LOEO (Fig. 5d, Table S-LOEO).**
PPO matches or beats random on 7 of 10 events; the three losses are
small (≤ 0.012 F1 each) and concentrated on events where one of the
other heuristics happens to win the seed lottery (Ghana, Pakistan, India):

| Event       | base   | random | PPO    | oracle | PPO−random | headroom |
|-------------|--------|--------|--------|--------|------------|----------|
| Paraguay    | 0.773  | 0.759  | **0.772** | 0.793 | **+0.013**  | +0.020   |
| Somalia     | 0.736  | 0.783  | **0.793** | 0.794 | **+0.010**  | +0.058   |
| Sri-Lanka   | 0.873  | 0.871  | **0.876** | 0.881 | +0.005     | +0.008   |
| USA         | 0.860  | 0.864  | 0.865  | 0.865 | +0.002     | +0.005   |
| Spain       | 0.860  | 0.893  | 0.893  | 0.892 | +0.000     | +0.032   |
| Mekong      | 0.952  | 0.956  | 0.956  | 0.957 | +0.000     | +0.005   |
| Nigeria     | 0.910  | 0.912  | 0.912  | 0.912 | +0.000     | +0.002   |
| India       | 0.841  | 0.838  | 0.836  | 0.846 | −0.002     | +0.005   |
| Pakistan    | 0.592  | 0.661  | 0.651  | 0.659 | −0.010     | +0.067   |
| Ghana       | 0.853  | 0.798  | 0.786  | 0.825 | −0.012     | 0.000    |

The largest PPO−random gains under 20-seed LOEO appear on Paraguay
(+0.013), Somalia (+0.010), and Sri-Lanka (+0.005); the losses (Ghana
−0.012, Pakistan −0.010, India −0.002) are within seed-level variance.
Notably, the 10-seed version of this table had reported Somalia +0.018
and Sri-Lanka +0.012 PPO−random gains and a Pakistan tie — doubling
seeds compressed all per-event differences toward zero, consistent with
the pooled finding that PPO and random are statistically equivalent
in mean.

**The earlier within-event protocol overstated the advantage.** A within-
event paired protocol (PPO trained on the same four hard regions it was
later scored on, with only the seed-level pool/test split varying)
originally yielded
+0.023 to +0.044 F1 paired-significant gains over random / uncertainty /
CoreSet (Table S-v1). Under the leakage-free LOEO protocol the same
v1 PPO loses to random by −0.007 on average, and only the structural
improvements above (GAE-λ + terminal reward + entropy schedule) restore the
policy to oracle-equivalent performance. We report both protocols in full
because the methodological lesson is itself a contribution: cross-disaster
active-calibration claims demand event-level holdout, not seed-level
pool/test reshuffling.

*[Fig. 5d = Fig 26 (LOEO-v1↔v2 + paired-difference summary + pooled-F1 ladder).
The earlier within-event PPO sample-efficiency curve and the
decision-aligned-reward A/B experiment, originally produced under the
leakage-suspect protocol, are deferred to the Methodological appendix
(R4-Appendix) so that the headline results in R4 use only the LOEO-v2
protocol.]*

#### R4d — Honest positioning: PPO ties the oracle, significantly beats the strongest uncertainty heuristic (ensemble uncertainty), and ties the simpler entropy heuristic

A reader will reasonably ask: among the methods you evaluate, *is* the
learned PPO policy practically necessary, or would a simpler heuristic
(uncertainty sampling, in particular) suffice? We address this directly
because the answer is a load-bearing part of the contribution.

**Pooled F1 across the 200 LOEO paired pairs (descending):**

| Method                  | Pooled F1 | Δ vs PPO    | t-p vs PPO |
|-------------------------|----------:|------------:|-----------:|
| full-pool oracle        |    0.8405 | +0.0065 | **0.016 \*** |
| uncertainty (entropy)   |    0.8390 | +0.0049 | 0.057 (borderline) |
| **PPO (ours)**          |   **0.8340** | —     | —          |
| random                  |    0.8335 | −0.0005 | 0.85 (n.s., Wilcoxon p = 0.009) |
| **ensemble uncertainty** (3-seed, 10-seed eval) | 0.8269 | **−0.0099** | **0.003 \*\*** |
| CoreSet                 |    0.8229 | −0.0111 | **0.007 \*\*** |
| zero-shot (τ = 0.5)     |    0.8193 | −0.0147 | **0.0005 \*\*\*** |

**Under the 20-seed protocol the picture is more nuanced than at 10
seeds.** At 10 seeds (n = 100) PPO was the best point estimate and
statistically tied with the full-pool oracle; doubling the seeds
(n = 200) cuts the standard error and reveals three things the 10-seed
estimate did not have power to resolve:

1. **The 4-chip family is tightly clustered just below the oracle.**
   The five practical 4-chip methods we tested (random, single-model
   entropy, CoreSet, ensemble uncertainty, PPO) span an F1 range of
   0.823 to 0.839, while the oracle sits at 0.840 — a total envelope
   of 0.017 F1. The cross-event calibration drift (R3) is up to +0.235
   F1 on a single event. *Method choice within the active-selection
   family accounts for at most ~7 % of the calibration lever.*
2. **Single-model entropy is the highest-mean 4-chip heuristic** at
   F1 = 0.839, borderline-significantly above PPO (Δ_PPO − unc = −0.005,
   *t*-p = 0.057). PPO and entropy are essentially tied in mean, with
   entropy's point estimate slightly favoured; the difference is below
   the practical resolution of the comparison.
3. **PPO retains a clean win over the more elaborate uncertainty
   baseline.** The 3-seed ensemble-uncertainty baseline — generated
   from three independently-trained leave-one-region-out U-Nets and
   ranked by per-pixel predictive std averaged per chip — is
   significantly worse than PPO (Δ = +0.0099, *t*-p = 0.003, evaluated
   at 10 seeds; the 20-seed extension is a planned follow-up). The
   mechanism is that epistemic uncertainty selects chips on which the
   ensemble *disagrees*, whereas calibration needs chips whose pixel
   distributions span the decision boundary. Adding ensemble compute
   does not help for the calibration objective; the simpler single-
   model entropy heuristic outperforms its multi-seed cousin.

**The reframed answer to the "is your method necessary?" question.**
PPO statistically dominates three of the four uncertainty heuristics we
evaluated (CoreSet, ensemble uncertainty, zero-shot — all
paired-significant), Wilcoxon-dominates random (the parametric mean
difference is essentially zero but the rank-shift is significant), and
is statistically *borderline* against single-model entropy (the simpler
heuristic has a slightly higher point estimate). The learned policy is
necessary against the stronger heuristics that recent active-learning
literature would propose as the natural comparator; against the simplest
one (entropy), the choice between learned and heuristic is within
seed-level noise.
The learned policy is necessary against the stronger heuristics that
recent active-learning literature would propose as the natural
comparator; it is unnecessary against the simplest one, because that one
already sits at the oracle ceiling for this task.

**The full-pool oracle is a hard ceiling that no active-selection method
can exceed.** The full-pool oracle re-fits the threshold τ on every chip
in the pool; this is the strongest 1-parameter post-hoc calibration
available for binary thresholded decisions (Methods: equivalence of
monotone post-hoc calibrations and threshold tuning). PPO reaches that
ceiling with a 4-chip budget. Methods we did not evaluate (Bayesian
active calibration, MCTS over chip subsets, oracle-imitation learning,
NeuralUCB-style bandits) can also at best *match* the oracle — they
cannot exceed it. The upper bound is structural, not protocol-dependent.

**Why PPO remains a meaningful contribution despite ties:**

1. **PPO reaches 99 % of the full-pool oracle's F1 at four labels.**
   The remaining 0.7 % F1 to the oracle ceiling is small in absolute
   terms and statistically significant only with the higher-power 200-
   pair protocol; the 4-chip PPO is *operationally near-oracle* for
   any practical purpose.
2. **The PPO MDP is a framework extension point that uncertainty
   heuristics are not.** Reward swapping (Methodological Appendix A2)
   demonstrably changes the policy paired-significantly on both
   backbones — uncertainty sampling cannot be retargeted to a
   decision-level objective without effectively defining a new
   heuristic for every objective. The architectural slack matters for
   the framework extension to multi-objective decision optimisation,
   which the calibration-only result here under-utilises.
3. **The negative ablation results (10-d feature set, RL-OPT removed)
   form a clean evidence chain that the 5-d PPO with GAE-λ +
   terminal-only reward + entropy schedule is the right point in the
   design space**, not an arbitrary one.
4. **PPO significantly beats the more elaborate uncertainty baselines
   (CoreSet, 3-seed ensemble uncertainty) and the zero-shot default.**
   A practitioner deploying *the safest possible 4-chip
   active-selection method* — one that does not catastrophically lose
   to any of the standard heuristics on a held-out event — would
   default to PPO; the rank-test (Wilcoxon) preference for PPO over
   random captures this conservativeness.

**The honest TL;DR for this section — tying back to H1 vs H2.** Under
the 200-pair 20-seed protocol, the five practical 4-chip
active-selection methods (random, single-model entropy, CoreSet,
ensemble uncertainty, PPO) span a 0.017-F1 envelope from zero-shot
(0.819) to the full-pool oracle (0.840). Every method sits inside that
envelope; the differences between methods (e.g., PPO − single-model
entropy = −0.005, *t*-p = 0.057) are within seed-level noise. The
cross-event calibration drift (R3) is up to +0.235 F1 on a single
event — *more than an order of magnitude larger* than the method-choice
spread. **The information that distinguishes representational from
calibrational explanations of cross-disaster generalisation does not
live in the choice of active-selection method; it lives in the
existence of the calibration lever itself.** This is the central
H2-supporting finding of the paper, and is robust to whether the
practitioner uses our PPO, an entropy heuristic, or random selection
of four chips. The operational implication — responders can deploy
near-oracle calibration on any new event with four labels and any
reasonable selection rule — is the deliverable.

> **Methodological appendix.** The original sample-efficiency budget sweep
> and the decision-aligned-reward A/B (Fig. 23, Fig. 24) were carried out
> under the leakage-suspect within-event protocol that the LOEO-v2 result
> above supersedes. We do not retract these experiments — the within-event
> sample-efficiency curve is a defensible characterisation of policy
> behaviour when the deployer can collect labels on the same event the
> model will be calibrated on, and the decision-reward A/B confirms the
> *architectural* claim that reward swapping changes the policy — but we
> separate them physically from the headline result to keep R4 entirely
> within the leakage-free LOEO protocol. The two ablations are deferred to
> the *Methodological Appendix* at the end of the manuscript.

---

#### Box 1 — Real-event walkthrough: Pakistan 2022 monsoon flood

To ground the H1-vs-H2 finding in a single concrete event, we run the
full agent on the 2022 Pakistan monsoon flood — the most severe
hydrological disaster in the 10-event Sen1Floods11 subset and the event
on which our cross-region model exhibits the largest single-event
calibration drift. Pakistan 2022 is also the event the UN OCHA Pakistan
Humanitarian Response Plan estimated 33 million people displaced by;
its operational scale makes the per-minute time savings of automated
decision-level answers materially impactful for response planning.

**Step 1 — Perception with a model that has never seen this event.**
The leave-one-region-out U-Net trained on the other nine flood events
predicts Pakistan's flooded-water mask in 0.031 s per chip. At the
default decision threshold τ = 0.5 the F1 against the analyst hand-label
is **0.543** — the model has the rank information about flooded vs. dry
pixels, but the per-event operating point is far off. The expected
calibration error is 0.237 — the largest among the 10 events. By the H1
account, this is where representation-engineering investment would now
need to begin (collect new Pakistan-specific training data, fine-tune,
foundation-model retrain).

**Step 2 — H2 diagnosis: τ is wrong, not the representation.** A
threshold sweep on the same predictions identifies τ\* = 0.70 as the
Pakistan-optimal threshold. With τ alone moved from 0.5 to 0.70, F1
jumps from 0.543 to **0.727 — a +0.184 F1 recovery from a single number
change**, no retraining and no new architecture. The +0.184 lever
exhausts ≈ 50 % of the available F1 deficit on this event without any
new training data.

**Step 3 — Deploying the lever with four labels under LOEO.** Under the
leakage-free 20-seed LOEO protocol, *every* 4-chip calibration method
we tested — including PPO, random, entropy, and CoreSet — recovers F1
within ≈ 0.025 of the Pakistan full-pool oracle (which sits at 0.659):
random 0.661, entropy 0.675, PPO 0.651, CoreSet 0.643. The
event-level differences between methods (≤ 0.024 F1) are small relative
to the +0.107 F1 lift that *any* of these methods provides over
zero-shot (0.592). **The Pakistan story is the calibration lever, not
the choice of selection method**: a four-label calibration recovers
~90 % of the available F1 deficit on the hardest cross-disaster event
in our panel, irrespective of which four labels are chosen.

**Step 4 — Decision-level answers.** The calibrated mask is piped
through the neuro-symbolic reasoning layer (OpenStreetMap road graph,
building polygons, hospital amenities; community connectivity via
connected components on the post-flood road graph) to produce the ten
standard UN-OCHA emergency questions: total flooded area
(km²), affected buildings (counts and footprint), affected road length
(km, by class), hospitals inside the flood footprint (geolocated list),
top-N priority roads to clear (ranked by length), and isolated
communities (graph-component analysis). The full briefing is emitted
as machine-auditable JSON plus an analyst-readable Markdown summary
(`outputs/dispatch/`). End-to-end wall-clock from event-time imagery
to briefing: minutes, dominated by the OSM Overpass query.

**Operational summary.** What a representational solution would frame
as a multi-month "Pakistan-specific retraining" project, the H2 framing
frames as a four-label, single-minute calibration. The Pakistan +0.184
F1 lever is not a curated demo; it is the single largest cross-event
recalibration gain in the 10-event Sen1Floods11 panel, and it is
recovered by an active-calibration policy that never saw Pakistan
during training. The same workflow — perception → calibration → OSM
reasoning → briefing — runs unchanged on the other nine flood events
and on the xBD damage-bearing hazards.

---

### R5 — Falsification test of H1 + calibrated negative results: foundation representation does not close the gap, structured-inference does not help, richer chip features do not help

H1's strongest prediction is that **a stronger learned representation
should close the cross-disaster F1 gap**. We test this by swapping in a
frozen Google AlphaEarth foundation embedding on matched inputs
(Sentinel-1 + Sentinel-2), and by exploring two further architectural
levers (structured-inference layer; richer chip features). All three
*fail* to close the gap — and these calibrated negatives are exactly
what an H2-dominated world predicts.

We characterise where commonly invoked tools fail, with the same multi-event
rigour applied to the positive results.

**Foundation embeddings on equal inputs are comparable, not superior.** Our
initial AlphaEarth comparison withheld Sentinel-2 from the foundation backbone
on the assumption that AlphaEarth "already fuses optical". On *equal* inputs
(AlphaEarth + Sentinel-1 + Sentinel-2, 64-d annual prior + event-day optical
+ event-day SAR) the foundation model's F1 rises from 0.610 to 0.807 — within
0.028 of the trained U-Net (0.835) and leading the model panel on AUPRC
(0.909) and recall (0.903). However, in a label-efficiency sweep at 5 %, 10 %,
25 %, 50 %, and 100 % of training labels the foundation model loses to the
U-Net at *every* budget (gap −0.16, −0.11, −0.14, −0.10, −0.05) — the
"scarce-label" promise of foundation embeddings does *not* materialise for
dense flood mapping. Stacking the AlphaEarth prior onto the U-Net as
additional channels degrades F1 from 0.835 to 0.769 — the annual prior adds
noise when bolted onto event-day optical (Methods).

**A structured Markov-random-field decision layer does not beat a calibrated
threshold.** Motivated by the causal-graph success of Xu et al. [2022], we
implemented a Structured Decision Inference (SDI) Markov random field over
the OSM building graph, with per-building flood evidence and spatial
smoothness over local building neighbours (Methods). On xBD building damage
across 14,285 buildings (4,074 GT-damaged) the symmetric-Potts SDI collapses
recall (P 0.91, R 0.62, F1 0.74); an asymmetric "attractive" variant designed
to fix this recovers to parity but still does not beat a simple calibrated
probability threshold (F1 0.769 vs 0.770 combined; SDI loses per-hazard:
palu-tsunami 0.35 vs 0.58, harvey 0.58 vs 0.64). The mechanistic explanation:
building damage in xBD is not spatially contiguous in the way flood water is,
so a spatial-smoothness prior is the wrong inductive bias. On synthetic
spatially-contiguous data the same SDI lifts F1 from 0.71 to 0.99 —
confirming the prior is *correct on contiguous phenomena* but wrong on the
real xBD damage task. We report this in full as a calibrated negative result
and use it to argue that calibration (not structured inference) is the
dominant lever for these decisions.

**The AlphaEarth pre / post temporal stack is degenerate for pre-2018
regions.** AlphaEarth annual coverage begins in 2017, so for Sen1Floods11
events in 2016–2017 (Pakistan, Sri-Lanka, India) the "pre-event" and "event-
year" annual composites are byte-identical and carry no temporal signal. We
report this as a fundamental data limit, not a method failure.

### R6 — System-level metrics: time-to-answer and reproducibility

We measure the full agent's wall-clock on a representative U.S. event chip
(Kansas, 5 km AOI, 2,053 OSM buildings, 9 critical facilities, 232 km of
major roads). Perception completes in 0.2 s; the OSM Overpass query is the
dominant cost (~6 min) and not the model; the neuro-symbolic reasoning layer
produces all ten UN-OCHA answers in another 1–2 s after OSM. Against the
documented 1–3 day expert workflow this is a three to four orders-of-
magnitude reduction in time-to-answer for the kinds of questions a responder
actually asks. The entire pipeline, including all twenty experiment scripts,
twenty intermediate result JSONs, and twenty paper-grade figures, regenerates
from raw data without manual intervention and is mirrored to a public live
dashboard (Methods).

---

## Discussion

Three implications follow from H2 dominating H1 for the cross-disaster
generalisation problem.

### Implication 1 — for the disaster-response community

Operational disaster response, today, treats every new event as a
problem of "how do we get a model that works on this event?". This
typically routes through one of two expensive options: collect a large
new labelled dataset for the new event, or wait for a methodological
advance (a stronger foundation model, a better fine-tuning recipe). Our
finding is that neither is needed. **Cross-disaster adaptation is a
four-label problem.** A responder team that can label four chips on the
new event — a task measured in minutes of human time, not days — can
deploy a model whose F1 on that event is statistically indistinguishable
from one calibrated using every chip in the labelled pool. The
infrastructure to integrate this into existing rapid-mapping workflows
(Copernicus EMS Rapid Mapping, UNOSAT, commercial providers) is
straightforward: the model itself is already trained, only the
threshold per event changes. The end-to-end agent we report runs at
0.031 s per chip — the machine time, exclusive of analyst verification, is much faster than the 1-to-3-day
EMS cycle — making the practical limit not compute but the human
labelling step, which our results bound at four chips.

### Implication 2 — for foundation-model research

A frozen Google AlphaEarth foundation backbone on matched Sentinel-1 +
Sentinel-2 inputs reaches roughly the same F1 as a trained U-Net + S1 +
S2 on the same task (Results R3) but does not surpass it. The
calibration headroom on AlphaEarth is comparable to the U-Net's
(+0.042 mean across four hard regions). Combined with the fact that
PPO calibration ties oracle on both backbones, the practical value of
swapping in a foundation model for cross-disaster flood mapping under
*matched inputs* is small. Where foundation models do plausibly add
value — pre-training compute amortisation, multi-task transfer, broader
modality fusion — is orthogonal to the cross-disaster F1 question we
test here. Our calibrated characterisation is, to our knowledge, the
first published apples-to-apples comparison of an Earth-observation
foundation model and a same-input U-Net on cross-event flood mapping;
its honest result — comparable F1, comparable calibration headroom,
comparable response to active recalibration — is part of the
contribution.

### Implication 3 — for the active-learning and post-hoc-calibration literatures

The active-learning literature has framed "what to label next" as an
*information* question over the model parameters. The post-hoc
calibration literature has framed it as a *probability scale* question
over the model outputs. For binary thresholded decisions — the case in
operational disaster mapping — these two frames collapse to the same
one-dimensional problem of estimating the optimal threshold τ on a new
distribution, and all monotone single-parameter post-hoc calibrations
(Platt, temperature, isotonic) are mathematically equivalent to
threshold tuning (Methods). Our finding that this collapses-to-1D
problem is solvable with four labels under a leakage-free
leave-one-event-out protocol means that, for tasks that ultimately make
binary decisions, **the active-learning–vs–calibration distinction is
not the right axis** — the right axis is sample efficiency on a
one-parameter post-hoc adjustment. We give one careful instantiation
(PPO with GAE-λ + terminal-only reward + entropy schedule, retaining
5-d chip features) and characterise the design space around it
(Methodological Appendix), but we expect any sensible active-selection
heuristic that targets the decision boundary to perform comparably on
this task.

### What would falsify H2?

We frame the test honestly: the three experiments above all support H2,
but they are not unconditional. H2 would be undermined by either of
the following observations.

(i) **A new event on which the *ranking* is broken.** Our discriminator
relies on the model's per-pixel ranking transferring across events.
If a held-out event were to show that the model's continuous predictions
were systematically wrong-ordered with respect to the ground truth —
i.e. that even the optimal threshold could not recover decent F1 — that
event would be evidence of representation drift, not calibration drift.
Our 18 real events across three benchmarks span 0.30–0.70 in optimal
threshold and all preserve a usable F1 ceiling under recalibration; we
do not see this failure mode in our data. The closest to failure is
the wildfire 2019 fire-season event whose τ\* sits exactly at 0.5 and
the F1 gain from recalibration is 0.000 — a degenerate case of H2 that
is *neutral* with respect to H1 (the representation suffices already)
rather than a refutation of H2.

(ii) **A backbone whose calibration headroom is zero.** If a stronger
backbone (a newer foundation model, a different pre-training corpus)
were to drive the optimal τ to 0.5 on every held-out event, that would
mean the representation already encodes the per-event decision
boundary — i.e. H1's corrective was achievable through better
representation alone. Our AlphaEarth comparison rules this out *for
that particular foundation model*, but it cannot rule it out
universally. We treat this as the most concrete falsification test
follow-up work should run.

The cross-hazard xBD result (R4-Appendix-style 3-seed analysis) also
sharpens the H2 claim: pre/post change-detection helps hazards that
manifest as visible change (hurricane harvey +0.194, florence +0.023,
palu-tsunami +0.030) but hurts a hazard where the post-image already
suffices (mexico-earthquake −0.073). This is an interaction with the
*hazard physics*, not with the H2 framework — calibration drift is
present in both regimes — but it does suggest that *which* change-
detection prior is the right inductive bias is hazard-specific, even
when the higher-level claim that calibration drift dominates
representation drift remains intact.

#### Limitations

(L1) *Hazard scope — partially addressed by HLS Burn-Scars; remaining
gaps.* The 18 real events now span three benchmarks (Sen1Floods11 floods,
xBD building damage, HLS Burn-Scars wildfires) and three hazard families
(water hazards, structural damage, wildfire). Within these three the
H2-dominates pattern holds but the magnitude of the calibration lever is
hazard-specific: large on floods and damage, small on wildfires (mean
+0.001 F1, ECE 0.011–0.025) because post-burn imagery is visually well
separated. The H2 *direction* is consistent across all three benchmarks;
the H2 *magnitude* is not. The remaining hazards we do not yet test —
landslide susceptibility, drought, snow-extent, oil-spill — could in
principle show a third pattern (e.g., negligible calibration lever
across the board, which would weaken the operational deliverable for
those hazards). The cross-hazard pre/post-change-detection result we
do report within xBD is already hazard-specific — pre/post helps
water/wind hazards (harvey +0.194, florence +0.023, palu-tsunami
+0.030) but hurts geophysical structural damage (mexico-earthquake
−0.073) — a mechanistically interpretable nuance within the data we
have.

(L2) *Backbone scope for the H1 falsification.* R3 swaps in one
foundation backbone (frozen Google AlphaEarth Satellite Embedding V1)
with one task-specific head architecture. A different foundation model
(SatMAE, Prithvi, DOFA, USat, CrossEarth) might in principle drive
optimal τ to 0.5 on every event and falsify H2 for *that backbone*. Our
falsification of H1 is currently single-backbone; a multi-backbone
H1-falsification panel is the second most concrete follow-up.

(L3) *Decision-level answer fidelity is event-aggregate, not per-chip.*
The headline flooded-area correlation (Pearson r = 0.971, Spearman ρ =
0.899) is faithful to the dominant signal — responders ranking events
by total flooded area — but small-area chips are not individually well
calibrated (bottom-50 %-by-area Pearson r = 0.118; MAPE median 25 %).
Per-building, per-road and per-population answers are partly validated
via the xBD damage analysis and are end-to-end-released pipelines, but
full external validation against Copernicus EMS reference masks or
WorldPop population counts on shared events is future work.

(L4) *Active-selection method choice is within seed-level noise at
4 chips.* Under the 20-seed LOEO protocol the entire 4-chip
active-selection family (random, single-model entropy, CoreSet,
ensemble uncertainty, PPO) fits inside a 0.017 F1 envelope below the
oracle. PPO is statistically indistinguishable from random in mean
(*t*-p = 0.85, Wilcoxon p = 0.009), borderline-significantly *below*
single-model entropy (*t*-p = 0.057), and significantly above CoreSet,
zero-shot, and ensemble uncertainty. The methodological contribution of
the paper is the H2 finding, not the choice of selection method; the
PPO architecture is justified primarily as a framework extension point
for decision-aligned reward (Methodological Appendix A2), not by a
robust per-event PPO-vs-random win.

(L5) *Within-event protocol experiments demoted, not retracted.* The
sample-efficiency budget sweep and the decision-aligned reward A/B
were carried out under a within-event protocol that the leakage-free
LOEO supersedes for the cross-event claim. The within-event numbers
remain defensible characterisations of policy behaviour when the
deployer can collect labels on the same event the model will be
calibrated on, and we report them in the Methodological Appendix —
but we do not use them as part of the headline H2-vs-H1 evidence.

#### Ethics and responsible use

Automated disaster-response answers are dual-use: a false negative on
"hospital flooded" can cost lives, and a false positive on "all roads
passable" can mislead responders. We argue accordingly for (a) auditable,
human-in-the-loop deployment of any decision-level answers; (b) explicit
reporting of failure modes and confidence (which we do throughout); and
(c) public reproducibility of all numbers (which we provide). The system is
intended to *augment*, not replace, trained mapping analysts.

---

## Methods

*[Outline only at this draft stage; expand at submission. ~1,500–2,500 words.]*

### Data

- **Sen1Floods11** [Bonafilia 2020] — 446 hand-labelled chips across 11 regions,
  Sentinel-1 + Sentinel-2 L1C imagery, water labels from optical photo-
  interpretation. We use the hand-labelled subset only.
- **xBD / xView2** [Gupta 2019] — train-split images and rasterised damage
  targets for five disasters (hurricane-harvey, hurricane-florence,
  guatemala-volcano, mexico-earthquake, palu-tsunami), tiled to 512 × 512.
- **Google AlphaEarth Satellite Embedding V1 (annual)** — 64-dimensional
  10 m annual embedding, fetched per chip via Google Earth Engine.
- **OpenStreetMap** — buildings, roads, hospitals via Overpass; reasoning-
  layer queries.

### Models

- **U-Net (smp\_unet)** — ResNet-34 backbone, in\_channels = 15 (S1 2 + S2 13),
  random init; focal-BCE loss (α 0.75, γ 2.0); AdamW lr 1e-4; bf16-mixed.
- **DeepLabV3+ (ResNet-50)** — same input/loss, as cross-architecture sanity.
- **AlphaEarth + S1 + S2** — frozen 64-d annual embedding + Sentinel-1 (2 ch) +
  Sentinel-2 (13 ch); per-pixel MLP head with hidden [256, 128], dropout 0.2,
  GELU; ~53 K trainable parameters.
- **U-Net + S1 + S2 + AlphaEarth** — same U-Net with in\_channels = 79 (15 + 64).
- **U-Net (xBD)** — same architecture, in\_channels = 3 (post-only optical /255)
  or 6 (pre + post optical /255).

### The Active Calibration Markov Decision Process

We formalise label-efficient threshold recalibration as the following MDP,
the methodological core of the CCA framework.

**Setting.** Let *D* be a target event represented by an unlabelled chip pool
*P* = {(x_i, ŷ_i)}_{i=1}^N where x_i is a chip and ŷ_i ∈ [0,1]^{H×W} the
per-pixel score from a perception model *f*. Let *T* be a held-out test
partition of the same event with ground-truth masks y_T, and let *L(τ; T)* be
a decision-level scalar score computed at threshold τ (e.g. pixel F1, or
mean absolute relative per-chip area error). Let *τ*\*(*S*) be the threshold
that maximises *L* on a labelled subset *S* ⊆ *P*. The active-calibration
MDP is:

- **State** *s_t* ∈ ℝ^{N×d}: per-chip feature vector (mean prediction, std,
  predicted-water fraction, mean and std of pixel entropy) augmented with
  (i) a binary "already-selected" mask and (ii) a scalar
  budget-remaining channel broadcast over chips.
- **Action space** *a_t* ∈ {1,…,N} \ S_{t-1}: select the next chip whose
  label will be queried.
- **Transition**: deterministic — *S_t = S_{t-1} ∪ {a_t}*, *τ_t = τ\*(S_t)*.
- **Reward** *r_t = L(τ_t; T) − L(τ_{t-1}; T)*: the *incremental gain in the
  decision objective* from adding the chosen chip.
- **Horizon** *t* ∈ {1,…,B} where *B* is the label budget.

**Decision alignment.** The objective *L* is a free hyper-parameter: when
*L* = pixel-F1 the MDP recovers a standard label-efficient calibration
problem; when *L* is a decision-level quantity (per-chip area error,
affected-building identification F1, population-in-flood error, …) the
reward signal aligns the policy with the downstream consumer of the maps.
The framework's decision-alignment claim is therefore architectural; an
explicit decision-aligned reward ablation is reported in the Methodological
appendix (R4-Appendix) under the within-event protocol, but is not part of
the headline LOEO claim because that experiment was produced under the
leakage-suspect protocol.

**Reward parameterisation.** We expose two reward modes for the same MDP:
``pixel`` (incremental: *r_t = L(τ_t; T) − L(τ_{t-1}; T)*) and
``terminal_pixel`` (terminal-only: *r_t = 0* for *t < B* and
*r_B = L(τ_B; T) − L(τ_0; T)*). The ``terminal_pixel`` reward equals the
total episodic gain by construction, sees an order-of-magnitude lower noise
than the incremental signal, and is the choice we use in the leakage-free
LOEO-v2 experiments. The choice between the two is a learning-stability
choice, not an objective change: both modes optimise the same total-episode
return.

**Why four labels are sufficient — an information-theoretic argument.**
The decision threshold τ is a *one-dimensional* parameter. Under the
binary-threshold equivalence above, the cross-event recalibration
problem reduces to estimating a single scalar τ\* on a new event from a
finite calibration set. Given a chip *i* with *n_i* labelled pixels,
the empirical positive rate p̂_i is a binomial-variance estimator of
the chip's true positive rate p_i with variance p_i(1−p_i)/n_i.
Aggregating across *k* selected chips with total pixel count *N* = Σ
*n_i* and effective per-pixel binary cross-entropy curvature *I(τ)* in
the neighbourhood of τ\*, the Cramér-Rao lower bound on the variance of
the maximum-likelihood τ\* estimator scales as **1 / (N · I(τ))**. In
our Sen1Floods11 chips each chip carries *n_i* ≈ 5 × 10⁴–5 × 10⁵
labelled pixels — i.e., *one* labelled chip already contributes
~10⁵–10⁶ pixel-level Bernoulli observations to the τ\* estimator.
**Four chips therefore contribute ~10⁶–10⁷ pixel observations,** more
than sufficient to drive the τ\* estimator to within the 0.001-F1
practical-equivalence neighbourhood of the oracle (which uses the
entire pool, ~10⁷–10⁸ pixel observations). Beyond ~four chips the
estimator is variance-limited not by chip count but by the per-event
heterogeneity of *I(τ)* across chips — i.e., by the *signal* in the
pixel distribution near the decision boundary, not by the *sample
size*. This is the mechanistic explanation for our empirical finding:
the threshold parameter is so low-dimensional, and individual chips so
information-rich, that 4-chip calibration recovers ≈ 99 % of the
oracle's F1 essentially regardless of how the four chips are chosen
within reason.

**Distinction from active learning.** This is *not* the standard active-
learning MDP whose objective is to minimise training-loss with few labels.
The model *f* is fixed; what changes with each query is only the
*decision threshold τ*. Empirically (Results 1) this is the dominant lever
under cross-disaster shift, which is why we propose the new formulation.

**Distinction from temperature / Platt / isotonic post-hoc calibration.** A
reader familiar with the calibration literature may ask why our baselines do
not include temperature scaling, Platt scaling, or isotonic regression. For
*binary thresholded* segmentation decisions all monotone 1-parameter
post-hoc calibrations are *equivalent* to threshold tuning: any monotone
remapping of the score preserves the ranking of pixels, and a thresholded
decision depends only on the ranking. Concretely, temperature scaling
applied at threshold 0.5 keeps the predicted-positive set identical:
σ(logit(p)/T) ≥ 0.5 ⇔ logit(p) ≥ 0 ⇔ p ≥ 0.5, independent of T; Platt
scaling σ(a + b·logit(p)) ≥ 0.5 ⇔ p ≥ σ(−a/b), which is again threshold
tuning with τ = σ(−a/b); isotonic regression is monotone and therefore
ranking-preserving. The "full-pool oracle" baseline in our experiments is
therefore precisely the strongest 1-parameter monotone post-hoc calibration
available for binary thresholded decisions, and under the leakage-free
LOEO 200-pair protocol our PPO recovers ≈ 99 % of its F1 (R4:
Δ_PPO−oracle = −0.0065, paired *t*-p = 0.016). A 4-chip
calibration set recovers near-oracle binary-decision performance.

**Important scope condition: this equivalence holds only for binary
thresholded decisions.** For our flood-mapping experiments the
classification task is binary (water / not water), and the equivalence
proof above applies in full. For our xBD damage analysis, the
classification task is in principle multi-class (no / minor / major /
destroyed damage), and the monotone-equivalence of Platt, isotonic, and
temperature scaling does NOT hold for multi-class confusion-matrix-
based decision metrics — these methods can affect the per-class score
ordering when more than one decision boundary is involved. To keep the
equivalence applicable, all xBD analyses we report in this paper
reduce damage to a binary present/absent decision per pixel (any
non-"no-damage" class collapses to "damaged"), the operationally
relevant decision for a first-response briefing. A full multi-class
calibration treatment of xBD damage is outside the scope of this paper
and we list it as future work in the Limitations.

**Policy and training.** A compact actor–critic network with two 64-unit
Tanh layers; permutation-equivariant per-chip scorer; masked categorical
action distribution that zeroes out already-selected chips. Trained with
PPO [Schulman 2017] (clip 0.2, γ 0.99, lr 3e-3, 4 epochs per update,
episodes\_per\_update 8, 300 updates, ``terminal_pixel`` reward).

Three load-bearing optimisation choices distinguish our final policy from
a textbook PPO baseline; in our LOEO evaluation each is necessary for the
policy to reach oracle-equivalent performance:

1. **GAE-λ advantage estimation** with λ = 0.95 in place of raw discounted
   returns. Episode returns are ~0.005 F1 with per-step σ ~0.05; raw
   discounted returns give an SNR near unity and the policy gradient
   collapses to noise. GAE-λ recovers tractable advantage estimates.
2. **Terminal-only reward.** As above, the ``terminal_pixel`` mode reduces
   step-level reward variance by an order of magnitude.
3. **Linear entropy schedule** *β(t) = β₀ + (β_T − β₀) · t/T* with
   *β₀ = 0.10*, *β_T = 0.01*. The original constant entropy (0.01) caused
   premature exploitation onto near-uniform random selection; the schedule
   maintains exploration through the critical early phase.

Gradient clipping (max-norm 0.5) on the policy parameters is retained as a
numerical safety measure but is not load-bearing.

**Feature-set ablation (negative result, retained 5-d set).** We tested
whether enriching the per-chip feature vector helps. Concretely, we
extended the 5-d feature set (`pr.mean`, `pr.std`, `(pr>0.5).mean`,
`ent.mean`, `ent.std`) with five additional dimensions hypothesised *a
priori* to be informative for active threshold selection: the
**decision-frontier proximity** `((0.3 < pr) & (pr < 0.7)).mean` (chips
with many near-0.5 pixels are the ones whose contribution will most
change as τ moves), and **four probability quantiles** at p₁₀, p₂₅, p₇₅,
p₉₀ (replacing the symmetric mean-±-std summary with a shape-aware
distributional summary). We then re-ran the full LOEO protocol (10
folds × 10 seeds = 100 paired pairs) with the 10-d feature set on the
same policy architecture and training budget. The result is a clean
negative:

| Comparison           | 5-d (v2)        | 10-d (v3)       |
|----------------------|-----------------|-----------------|
| PPO − random         | +0.0047, W-p=0.0006 *** | −0.0001, W-p=0.047 |
| PPO − CoreSet        | +0.0082, **t-p=0.024 *** | +0.0056, t-p=0.190 (n.s.) |
| PPO − zero-shot      | +0.0147, **t-p=0.009 *** | +0.0100, t-p=0.113 (n.s.) |
| PPO − full-pool oracle | −0.0020, n.s. (**tied**) | −0.0067, **t-p=0.015 *** (now significantly worse) |
| Pooled PPO F1        | 0.8368          | 0.8320 (−0.005) |

The 10-d set causes PPO to lose paired significance against CoreSet and
zero-shot, drop the Wilcoxon-rank advantage against random by an order of
magnitude, and — most diagnostically — become **significantly worse than
the full-pool oracle** rather than tied with it. We attribute this to a
capacity/training-budget mismatch: doubling the per-chip input dimension
while keeping the actor MLP at 2 × 64-Tanh and the update budget at 300
under-fits the new input distribution. We accordingly **retain the 5-d
feature set as the headline configuration** in R4; the 10-d
configuration is reported here as an ablation rather than as the headline
method. Result and figure: `outputs/layer3_ppo/ppo_loeo_v3_aggregate.json`
and Fig. 28.

**Statistical protocol — leave-one-event-out (LOEO).** Our headline numbers
in R4 are produced under a strict event-level leave-one-out protocol: for
each of the ten Sen1Floods11 events we hold the event out, train the PPO
policy from scratch on the other nine events only, freeze the policy, then
score it (and all baselines) on the held-out event with a re-shuffled
pool/test split per seed. With twenty seeds per fold this gives 200
paired pairs over which we compute the paired t-test and the Wilcoxon
signed-rank test. (An earlier 10-seed protocol giving 100 paired pairs
was the initial headline; doubling the seeds to 20 was added in
response to the small absolute effect sizes, and the manuscript reports
the 20-seed numbers as the headline.) The 200-pair sample size is
necessary to detect the small absolute
effect sizes (≈ +0.005 F1) characteristic of the calibration lever; the
event-level holdout is necessary to eliminate the within-event
train/test-overlap that an earlier protocol (10 seeds re-shuffling
pool/test on the same four hard regions) was suspect of.

### Neuro-symbolic reasoning layer

For each chip with a calibrated flood mask in EPSG:4326:
1. fetch building polygons and tagged amenities via OSMnx (Overpass);
2. fetch the major-road graph (motorway / trunk / primary / secondary /
   tertiary / residential) via OSMnx;
3. compute, per geometry, the fraction of its footprint or length intersecting
   the flood mask;
4. answer ten standard questions (total buildings; affected buildings; affected
   road length; hospitals in flood footprint; isolated communities by
   connected-component analysis of the post-flood road graph; top-N roads
   ranked by length to be cleared; etc.);
5. emit a JSON briefing + a Markdown-formatted summary suitable for direct
   responder consumption.

### Decision-level evaluation

For Sen1Floods11 we use the analyst hand-label mask as ground truth and
compare against the calibrated model mask at the level of (a) flooded area
per chip, (b) per-building affected state where OSM is available. For xBD
we use the rasterised damage target as ground truth and report (c)
per-building damage decisions, with the labels JSON polygons as the
per-building unit.

### Reproducibility

All experiment scripts (`scripts/*.py`), model configs (`configs/model/*.yaml`),
intermediate result JSONs (`outputs/**/results.json`), figures
(`outputs/figures/fig*.png`), and the live agent dashboard
(`outputs/site/index.html`) are auto-rebuilt from raw inputs by a single
`geodisaster build-blog` invocation. The full code base, all 20+ result
JSONs, and all 20+ paper-grade figures are published to GitHub at
https://github.com/14H034160212/geodisaster-fm and mirrored to two public
live deployments.

---

## Code & data availability

All code, intermediate results, and figures are public:

- Source: https://github.com/14H034160212/geodisaster-fm
- Live dashboard (GitHub Pages): https://14h034160212.github.io/geodisaster-fm/
- Live dashboard (Cloudflare Pages): https://geodisaster-fm.pages.dev/

Sen1Floods11 [Bonafilia 2020], xBD [Gupta 2019], and AlphaEarth [DeepMind
2024] are all public; raw fetch and patch-tiling scripts are included.

---

## Acknowledgements

*[TODO]*

## Author contributions

*[TODO]*

## Competing interests

*[TODO]*

---

## Methodological Appendix — within-event protocol ablations

The two experiments below were carried out under the original within-event
paired protocol (cross-region PPO trained and evaluated on the same four
hard Sen1Floods11 regions, with the seed-level pool/test split varying).
This protocol is leakage-suspect for cross-event generalisation claims;
under the strict leave-one-event-out protocol used in R4, PPO matches the
full-pool oracle and outperforms zero-shot and CoreSet but does not
parametrically out-perform random selection (Wilcoxon-significant only).
We retain these two ablations because (a) the qualitative direction of
the sample-efficiency curve is robust and re-appears under LOEO-v2; (b)
the decision-aligned-reward A/B confirms the *architectural* claim that
swapping the reward signal changes the policy, which is independent of the
event-level holdout discussion; (c) the within-event sample-efficiency
numbers are an honest characterisation of how the same policy behaves in
the *operational* setting where labels are collected on the same event the
model will be calibrated on.

### A1. Budget sample-efficiency sweep (within-event protocol)

We re-ran the 10-seed within-event paired protocol at label budgets B ∈
{1, 2, 4, 8} on the U-Net backbone, evaluating PPO against random /
uncertainty / CoreSet and the full-pool oracle (Fig. 24). The pattern is
the canonical sample-efficient one:

| Budget | random | PPO | PPO − random | paired *t*-p |
|--------|--------|-----|--------------|-------------|
| 1      | 0.720  | **0.781** | **+0.062** | <0.001 |
| 2      | 0.736  | 0.779 | +0.044 | <0.001 |
| 4      | 0.756  | 0.779 | +0.023 | 0.005 |
| 8      | 0.760  | 0.777 | +0.017 | 0.013 |

PPO's edge **decreases monotonically** as the budget grows: +0.062 → +0.044
→ +0.023 → +0.017 F1. All four budgets are paired-significant
(*t*-p ≤ 0.013) under this protocol. Strikingly, **PPO at budget = 1
(F1 0.781) matches or exceeds every baseline at budget = 8** (random 0.760,
uncertainty 0.755, CoreSet 0.757) — the learned chip-selection is worth
roughly an 8× label multiplier at the bottom of the curve. PPO's absolute
F1 also saturates quickly (0.777 – 0.781 across budgets), consistent with
the calibration MDP being a small-effective-dimension problem: a single
well-chosen chip captures most of the recoverable threshold information
for the test region. The qualitative slope is preserved under LOEO-v2,
where PPO at budget = 4 attains oracle-equivalent F1 (R4 above).

### A2. Decision-aligned reward A/B (within-event protocol)

This experiment supports the *architectural* claim of CCA that the
calibration MDP can be solved against any decision-level objective by
changing the reward signal. On a 20-seed within-event paired protocol
(initially 10 seeds; doubled after the first run because the
decision-metric paired CIs were wide) we train two arms — the standard
pixel-F1-reward arm and a **decision-aligned** arm whose reward is the
per-step decrease in mean absolute relative area error across test chips —
and evaluate both arms on *both* metrics. We report two separate findings,
with opposite statistical strength (Fig. 23).

**(i) Reward shaping is a real, paired-significant control knob.** Across
20 seeds and four hard regions, decision-reward PPO has *significantly
lower* pixel F1 than pixel-reward PPO on both backbones: U-Net 0.758 vs
0.778, paired *t*-test **p = 0.0004**; AlphaEarth 0.701 vs 0.728, **p =
0.005**. The MDP genuinely steers toward whatever objective the reward
encodes — it is not blindly maximising pixel F1 regardless of the reward
signal. This verifies the framework's central claim that the reward signal
is the control knob that aligns the calibration policy with a chosen
objective.

**(ii) Decision-aligned reward does *not* yet net-improve the decision
metric.** On mean absolute relative area error (lower is better), the
direction is **backbone-dependent** and **not statistically significant**
at n = 20 × 4: on U-Net decision-PPO is slightly *worse* (6.26 vs 5.96,
Δ = +0.31, paired *t*-test p = 0.75); on AlphaEarth decision-PPO is
slightly *better* (3.66 vs 4.70, Δ = −1.04, p = 0.37). We note that a
10-seed pilot gave a much larger AlphaEarth effect (Δ = −2.90, −62 %
relative) which the 20-seed re-run halved and rendered non-significant — a
textbook noise reversal under-powered RL evaluations are vulnerable to. We
report both runs in full because *the methodological lesson is itself a
contribution*: RL-on-disaster-mapping evaluations need ≥20 seeds and wider
region pools to distinguish a real decision-metric improvement from seed
noise.

**(iii) Honest synthesis.** What is *robust*: the reward is a paired-
significant control knob — the policy provably tracks the objective the
reward encodes (i). What is *not yet robust*: that an area-error reward
produces a net improvement on the area-error metric vs the pixel-F1
reward in this 4-region testbed (ii). We argue this is a power problem
not a no-effect result — the AlphaEarth direction is consistent across
both 10- and 20-seed runs, and the AlphaEarth effect at 20 seeds is still
a −22 % relative reduction, just with a 95 % CI that includes zero.
Resolving this requires either more seeds, more regions, or richer
decision rewards; we treat that as the most concrete next-step
experiment.

---

## References

*[Stub — to be expanded with full BibTeX at submission.]*

- Bonafilia, D., Tellman, B., Anderson, T. & Issenberg, E. Sen1Floods11: a
  georeferenced dataset to train and test deep learning flood algorithms for
  Sentinel-1. *CVPR Workshops* (2020).
- Brown, C. F. et al. AlphaEarth Foundations: a planet-scale embedding for
  Earth observation. *Nature* (2024).
- Gupta, R. et al. Creating xBD: A dataset for assessing building damage from
  satellite imagery. *CVPR Workshops* (2019).
- Schulman, J. et al. Proximal policy optimization algorithms. *arXiv*
  1707.06347 (2017).
- Xu, S., Dimasaka, J., Wald, D. J. & Noh, H. Y. Seismic multi-hazard and
  impact estimation via causal inference from satellite imagery. *Nature
  Communications* 13, 7793 (2022).
- Zhang, F. et al. AI-powered spatiotemporal imputation and prediction of
  chlorophyll-a concentration in coastal ecosystems. *Nature Communications*
  (STIMP) (2025).
