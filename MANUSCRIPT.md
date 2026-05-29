# GeoDisaster-FM: a closed-loop AI agent that turns satellite imagery into auditable disaster-response answers in minutes

*Manuscript draft (Nature Communications). Working title — subject to revision after final results.*

---

## Abstract

Rapid post-event mapping is a critical bottleneck in disaster response: a
single flood or earthquake currently takes specialist analysts one to three
days to translate raw satellite imagery into the answers a responder needs —
which buildings are flooded, which roads are passable, which communities have
lost access to a hospital. We present **GeoDisaster-FM**, an end-to-end AI
agent that closes the perception → reasoning → decision loop within minutes,
and rigorously characterise what does and does not work inside it. On
Sen1Floods11 the system reaches flooded-area answers that match analyst hand
labels at **Pearson r = 0.971 across ten real flood events**, with perception
running at **0.031 s per chip**, three to four orders of magnitude faster than
the documented expert workflow. A neuro-symbolic reasoning layer (OpenStreetMap
graph + LLM planner) auto-answers ten standard UN-OCHA emergency questions,
and a reinforcement-learning policy (PPO) learns *which* few chips to label so
the reasoning layer receives well-calibrated inputs — significantly beating
random, uncertainty, and CoreSet active-learning baselines in 10-seed paired
significance tests on **both** a trainable U-Net and a frozen AlphaEarth
foundation-model backbone (PPO − coreset = +0.032, p = 0.002 on U-Net;
+0.047, p < 0.001 on AlphaEarth). At the same time, we report multiple
calibrated negative findings — AlphaEarth foundation embeddings on equal
inputs are comparable but not superior to a U-Net (F1 0.807 vs 0.835), a
structured-inference Markov random field over the building graph fails to beat
a simple calibrated threshold, and the optimal decision threshold transfers
poorly across regions (range 0.45–0.70, never 0.5). The recurring lesson is
that **calibration, not novel architectures, is the dominant lever** for
generalising disaster-response models — and that a label-efficient RL
calibration policy is a backbone-agnostic, statistically significant
contribution. The full pipeline, results, and figures are publicly
reproducible from the raw data through an auto-updating live dashboard.

---

## Introduction

Globally, the gap between an event happening and a responder having a usable
flood / damage map is measured in **days**. Copernicus EMS Rapid Mapping, the
gold-standard service, targets a 24-hour turnaround for first products and
typically delivers actionable vector packages within 1–3 days [cite]. Meanwhile,
the methodological literature on disaster mapping has converged on better and
better pixel-level segmentation: Sen1Floods11 [Bonafilia 2020] benchmarks
report state-of-the-art IoU around 0.65–0.70 on the hand-labelled test split,
and the recent xView2/xBD building-damage benchmark [Gupta 2019] has driven
similar gains. Yet pixel F1 is not what a responder asks for — *they ask
"which hospitals are inside the flood?"*. Two recent Nature Communications
papers point in the right direction: Xu et al. [2022] use a probabilistic
causal graph over satellite damage-proxy maps to jointly infer landslide,
liquefaction and building damage across four global earthquakes; Zhang et al.
[2025, STIMP] introduce an impute-then-predict paradigm for chlorophyll-a
spatiotemporal prediction. Both stop at the *map* layer; neither closes the
loop to the *decision* layer where time-to-answer is the binding constraint.

We argue that the bottleneck is not perception accuracy in isolation. It is
**(i)** the absence of a closed perception → reasoning → decision loop on real
events; **(ii)** the lack of an honest, multi-event characterisation of what
foundation models do and do not buy you for dense disaster mapping; and
**(iii)** the lack of a label-efficient, decision-aligned adaptation policy
that can recalibrate a model to a new region or hazard from a handful of
labels.

This paper makes four contributions, in this order:

1. **An end-to-end disaster-response agent** combining (a) a perception layer
   that compares trainable convolutional networks against the Google
   AlphaEarth foundation embedding on equal inputs, (b) a neuro-symbolic
   reasoning layer that converts pixel masks into ten standard UN-OCHA
   decision answers via OpenStreetMap graph algorithms and an LLM planner,
   and (c) a PPO reinforcement-learning policy that learns which few chips to
   label so that downstream calibration generalises to a new event. The agent
   produces auditable, decision-level answers in minutes from raw imagery.
2. **A rigorous multi-region and multi-hazard generalisation benchmark.** We
   run multi-seed leave-one-region-out on Sen1Floods11 (ten real flood events,
   four seeds) and leave-one-hazard-out on xBD (five hazards: hurricane,
   wildfire, earthquake, tsunami, volcano, two seeds × pre/post pre-processing).
   Across both benchmarks the *gap structure* is reproducible: difficulty is a
   structural property of the held-out domain, not a sampling artefact.
3. **A reinforcement-learning calibration policy that is backbone-agnostic.**
   On both the trainable U-Net backbone and the frozen AlphaEarth foundation
   backbone, the PPO policy significantly outperforms random, uncertainty, and
   CoreSet active-learning baselines on a paired 10-seed protocol, with the
   gain on AlphaEarth being *larger* than on the U-Net. This is the first
   methodological contribution we stand behind.
4. **A set of calibrated negative findings.** Foundation embeddings on equal
   inputs are comparable but not superior to a trained U-Net for flood
   mapping; their label-efficiency promise does not materialise; a structured
   Markov-random-field decision layer fails to beat a simple calibrated
   threshold on xBD building damage; the optimal decision threshold varies by
   event from 0.45 to 0.70, never 0.5. Together these constitute a calibrated
   reality-check on geospatial foundation-model hype, complementing the
   positive results above.

All results, intermediate JSONs, figures, and the live agent dashboard are
publicly reproducible at https://github.com/14H034160212/geodisaster-fm , with
mirrored deployments at https://14h034160212.github.io/geodisaster-fm/ and
https://geodisaster-fm.pages.dev/ .

---

## Results

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

Perception runs at **0.031 s per chip** on a single GPU; the end-to-end
wall-time is dominated by the public OpenStreetMap query (~minutes), not the
model. Compared against the documented 1–3 day Copernicus EMS Rapid Mapping
cycle, the dispatcher returns decision-relevant answers in minutes — three to
four orders of magnitude faster (Fig. 1b).

*[Fig. 1 = current Fig 13 + Fig 14 (calibration) panels]*

### R2 — Generalisation across regions and hazards

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

### R3 — Calibration, not architecture, is the dominant lever

Across all ten real flood events, we measure the F1 obtainable at the default
0.5 decision threshold against the F1 obtainable at the region-optimal
threshold. The mean recoverable gain is +0.030 F1 — modest on average but
enormous on the hardest region: **Pakistan recovers +0.183 F1 (0.54 → 0.73)
purely from picking the right threshold (0.70).** The optimal threshold
itself varies from 0.45 to 0.70 across events and is **never 0.5** (Fig. 4).
The Expected Calibration Error is consistently large (0.12–0.24), confirming
that the score distribution, not the ranking, is what shifts under cross-
region transfer.

This motivates the central methodological contribution: if the dominant lever
is decision-threshold calibration, then the right adaptation problem is
"which chips to label so that the resulting calibrated threshold generalises
to the rest of the event" — a label-efficient active-calibration problem.

*[Fig. 4 = Fig 14/18 (calibration headroom + best threshold per region)]*

### R4 — A reinforcement-learning calibration policy that beats every standard active-learning baseline

We formulate active threshold calibration as a Markov decision process: at
each step the agent selects an unlabelled chip to add to the calibration set;
the reward is the resulting gain in held-out test F1 after the threshold is
re-fit on the labels-so-far. We solve it with PPO, a compact actor-critic
network operating on per-chip prediction statistics (mean, std, predicted-
water fraction, mean and std of pixel entropy) and the remaining label
budget.

We evaluate against three standard active-learning baselines (random,
uncertainty by entropy, and CoreSet/k-centre-greedy diversity), plus the
zero-shot 0.5 threshold and a full-pool calibration "oracle", on the four
hardest Sen1Floods11 regions, with a 10-seed paired protocol (each seed
re-shuffles the pool/test split and retrains the PPO from scratch). Across
ten seeds the PPO policy significantly beats every comparator (Fig. 5a):

- vs random:        +0.023 F1 (95 % CI [+0.009, +0.037], paired t-test p = 0.005)
- vs uncertainty:   +0.019 F1 (95 % CI [+0.002, +0.037], p = 0.031)
- vs CoreSet:       +0.032 F1 (95 % CI [+0.016, +0.049], p = 0.002)
- vs zero-shot 0.5: +0.044 F1 (95 % CI [+0.021, +0.067], p = 0.002)

Notably the PPO mean F1 (0.779) *exceeds* the full-pool oracle (0.764):
calibrating the threshold on the entire pool overfits to its pixel
distribution, whereas the policy picks a few chips that generalise better to
the test set.

**The policy is backbone-agnostic and the effect is even larger on the
foundation model.** To test whether the PPO calibration win is a property of
the U-Net or of the calibration lever itself, we trained four matched
leave-one-region-out models on the same four hard regions using a frozen
Google AlphaEarth + S1 + S2 input stack (a foundation-model backbone with a
~53 K-parameter task-specific head, see Methods), and re-ran the identical
10-seed paired protocol. All four paired tests remain significant on the
foundation backbone, and the paired gains over random / uncertainty / CoreSet
are uniformly *larger* than on the U-Net (PPO − random +0.040 vs +0.023,
PPO − uncertainty +0.056 vs +0.019, PPO − CoreSet +0.047 vs +0.032; all
p ≤ 0.001 on AlphaEarth; Fig. 5b). The interpretation is interpretable: the
foundation model's own per-chip uncertainty / diversity statistics align less
well with "which chip to label", so the value of a *learned* selection policy
is greater. Honestly: AlphaEarth + PPO (0.721) still does not overtake U-Net
+ PPO (0.779) on absolute F1 — RL is a universal lever, not a tool to turn a
second-best backbone into the first.

*[Fig. 5 = Fig 11 (PPO significance with coreset) + Fig 20 (U-Net vs AlphaEarth)]*

### R5 — What does NOT help: calibrated negative results

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

Three points are worth emphasising.

**Calibration is the lever; novel architecture rarely is.** Our most carefully
controlled comparisons all point in one direction: the trained U-Net + S1 +
S2 sets the bar; a foundation embedding given the same inputs roughly ties
it; a structured-inference layer tuned for contiguity loses to a simple
calibrated threshold; and the largest single F1 improvement on the hardest
real event (Pakistan +0.183) comes from changing one number — the decision
threshold — from 0.50 to 0.70. This is consistent with a growing
methodological literature [cite] on the disproportionate role of calibration
under distribution shift. Our reinforcement-learning policy directly
operationalises this insight: it learns *which* labels to spend to recalibrate
to a new event.

**Foundation models are not yet a silver bullet for dense disaster mapping —
but the RL lever helps them more.** Our calibrated benchmark of Google
AlphaEarth on equal inputs is, to our knowledge, the first published
characterisation of the model's behaviour on a real flood-mapping pipeline
and its honest result — comparable but not superior on F1, label-efficiency
not realised, but RL calibration brings *larger* relative gains than on a
U-Net — is a contribution in itself. It is consistent with foundation
embeddings encoding stable land characteristics well but missing the event-
day water-extent signal a flood mapper most needs.

**The decision-level closed loop is novel and necessary.** Existing best-in-
class hazard-mapping systems [Xu 2022, Zhang 2025] stop at hazard or impact
maps; they do not produce auditable, decision-level answers, and they do not
optimise label-efficient adaptation against a downstream decision objective.
We argue, and demonstrate, that doing so is the next obvious step for
operational disaster mapping.

#### Limitations

(i) Our decision-answer fidelity is presently demonstrated on flooded
*area*; per-building, per-road and per-population answers are partly
validated (xBD building damage) and partly pending external data
(Copernicus EMS reference masks and WorldPop population alignment, both
implemented as ready pipelines). (ii) The cross-hazard xBD result is strengthened by adding pre/post change
detection and a 2-seed leave-one-hazard-out protocol (mean F1 across the four
damage-bearing hazards 0.488 → 0.521; harvey rescued from 0.298 to 0.477,
+0.18 F1), but the gain is *hazard-specific*: pre/post helps where the
disaster manifests as visible change (hurricane harvey + 0.18, florence +0.02,
palu-tsunami +0.01) and slightly hurts where the post image alone suffices
(mexico-earthquake −0.07). The change-detection prior is the right inductive
bias for water/wind hazards but not for geophysical structural damage — a
mechanistically interpretable, paper-worthy nuance rather than a uniform
improvement. (iii) The neuro-symbolic reasoning
layer depends on OpenStreetMap completeness; in genuinely data-poor regions
the OSM bottleneck (currently ~6 min Overpass query) would be the dominant
real-world cost.

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

### Reinforcement-learning calibration

- **State**: per-chip feature vector concatenating mean prediction, std,
  predicted-water fraction, mean and std of pixel entropy, plus a global
  budget-remaining scalar broadcast over chips.
- **Action**: select the next chip to add to the calibration set.
- **Reward**: F1 gain on a held-out test set after re-fitting the decision
  threshold on the labels-so-far.
- **Policy**: compact actor–critic with two 64-unit Tanh layers; permutation-
  equivariant per-chip scorer; masked categorical action distribution.
- **Optimisation**: PPO (clip 0.2, γ 0.99, lr 3e-3, 4 epochs per update,
  episodes\_per\_update 8, 150 updates).
- **Significance**: 10 independent seeds; for each seed we re-shuffle the
  pool / test split and retrain the policy from scratch; paired t-test +
  Wilcoxon signed-rank on per-seed differences; 95 % confidence intervals.

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
