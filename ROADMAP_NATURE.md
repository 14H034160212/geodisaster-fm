# Roadmap to Nature Communications — GeoDisaster-FM (benchmark-calibrated)

**Target:** Nature Communications (primary) · Nature Machine Intelligence /
Nature Sustainability (alternates). Nature flagship = stretch, revisit only if
real-world impact validation is exceptional.

## 0. The bar — two Nature Communications role models

We calibrate against two papers the PI set as the standard (repo root):

- **Xu et al. 2022 (`s41467-022-35418-8`)** — seismic multi-hazard impact via a
  probabilistic **causal-graph** over satellite Damage Proxy Maps; variational
  inference + EM; **4 global earthquakes**; ROC/AUC vs ground truth; reveals
  causal mechanisms. *(Closest analogue to us.)*
- **Zhang et al. 2025 (`s41467-025-62901-9`, STIMP)** — impute-then-predict
  paradigm; **4 global coastal oceans**; beats geoscience + leading-AI baselines
  by large margins.

**Their shared formula (what we must hit):** a *novel method* → *clear, quantified
win* over *strong baselines* → on *multiple real global events* → plus a
*scientific insight* + practical impact.

**Where we must reach them:** (a) method novelty — we currently have standard
U-Net + a small RL calibration; (b) a winning result — our headline is a negative
foundation-model finding. **Where we can exceed them:** the full
perception→reasoning→**decision** closed loop with *time-to-answer* (both papers
stop at hazard/impact maps), and breadth across hazards + modalities.

## 0b. Reframed contribution (2026-05-28, after testing the method honestly)

We TESTED the "structured decision layer wins" hypothesis and it FAILED: on xBD
building damage (thousands of GT buildings) a Structured-Decision-Inference MRF
ties-or-loses to a simple calibrated threshold on every slice (combined 0.769 vs
0.770; palu 0.35 vs 0.58; harvey 0.58 vs 0.64). Reported as a full negative
result. The recurring project finding is **"calibration is king"** — our fancier
methods (foundation embeddings, structured inference) tie or lose to simple
calibration. We do NOT claim a novel winning method like the role-model papers.

**The honest four-pillar contribution we stand behind:**
1. **End-to-end dispatcher** (perception→reasoning→decision, time-to-answer) —
   the closed loop the role models lack.
2. **Rigorous multi-region + multi-hazard generalisation benchmark** —
   Sen1Floods11 cross-region (multi-seed) + xBD leave-one-hazard-out; the
   "difficulty is structural" gap recurs across sensor/hazard/task.
3. **Honest mechanism findings** — event-day optical dominates; foundation model
   ≈ U-Net; calibration > structure.
4. **A small but significant RL calibration result** (+0.023 F1, p<0.005).

This is a rigorous *system + benchmark + honest-evaluation* paper. Venue: be
realistic — strong domain venue (TGRS/RSE) or a systems/benchmark-friendly track;
matching the role models' "novel winning method" bar is NOT supported by current
evidence, and we will not pretend otherwise.

**Status legend:** ✅ done · 🟡 partial/in-progress · ❌ not started

---

## 1. The thesis (the "hook")

> An end-to-end AI system that converts raw satellite imagery into **actionable,
> auditable disaster-response answers in minutes instead of 1–3 days**, validated
> **across regions and across hazards**, with a rigorous and honest account of
> *what actually drives performance* — event-day imagery plus a label-efficient,
> self-calibrating adaptation policy, not foundation-model embeddings alone.

Why this is Nature-family (not a methods journal): the contribution is a
**working system + real-world impact + rigorous, honest evaluation**, not a new
SOTA architecture. The negative foundation-model finding *strengthens* the rigor
narrative; it is not the headline.

## 2. Three contributions (re-ordered to lead with a winning method)

1. **Structured decision-inference layer (the headline, must WIN).** Turns
   footprint predictions + infrastructure graph into calibrated decision-level
   answers; beats strong baselines on **decision-level metrics vs ground truth**
   across multiple real global events. 🟡 (seed = `validate_decisions_vs_gt`)
2. **Generalization + label-efficient adaptation** — multi-region (Sen1Floods11)
   + multi-hazard (xBD) generalization, plus a *significant* RL policy that picks
   which few chips to label, vs active-learning/DA baselines. 🟡 (cross-region ✅,
   cross-hazard 🟡, RL ✅ but small — needs baselines + generalization)
3. **The Dispatcher + honest mechanism analysis** — the full
   perception→reasoning→decision loop with *time-to-answer* (our edge over the
   role models), plus the calibrated foundation-model benchmark (event-day
   optical dominates; AlphaEarth comparable-not-better). 🟡/✅

---

## 3. Gap-filling checklist (prioritized)

### P0 — required for a credible Nature Comms submission
- [ ] **Decision-level WIN vs strong baselines (the headline).** On real events
      with ground truth, our structured decision layer must beat baselines on
      *decision* metrics (affected-building/road identification F1, lost-access,
      people-exposed). Baselines: (i) raw-threshold mask → answers, (ii) naive
      "all-buildings-in-flood-bbox", (iii) prior-only / fragility-style proxy.
      Like Xu et al.'s ROC/AUC-vs-GT but at the decision level. ❌ (seed running:
      `validate_decisions_vs_gt`, USA held-out)
- [ ] **Real-event end-to-end validation with ground truth, ≥8 events, multi-
      hazard, multi-continent** (Sen1Floods11 hand-labels = real floods w/ GT
      now; + Copernicus EMS real floods; + xBD other hazards). Match the role
      models' "4 global events" and exceed on count + hazard diversity. 🟡
- [ ] **Quantified time-to-answer.** Wall-clock of the automated pipeline vs the
      documented expert-mapping timeline — the headline impact number. ❌
- [ ] **A scientific insight** (not just accuracy), à la their causal mechanisms:
      e.g. *which* structured priors most improve decision accuracy, or a
      quantified law of cross-region/hazard transfer degradation. ❌
- [ ] **Multi-hazard generalization (xBD) finished** with multi-seed CIs. 🟡
- [ ] **Multi-region generalization + significance.** ✅ (leave-one-region-out,
      4-seed CIs)

### P1 — strongly strengthens the methods story
- [ ] **RL policy generalized** beyond threshold calibration → joint
      "which chips to label + how to adapt", run on BOTH benchmarks. 🟡
- [ ] **Active-learning / domain-adaptation baselines** (entropy, CoreSet, BALD,
      BatchBALD, simple DA) so the RL gain is benchmarked, with significance. ❌
- [ ] **Calibration analysis** (ECE before/after) — the threshold-transfer story
      that motivates the RL policy. 🟡 (ECE computed; needs a figure + narrative)
- [ ] **Population exposure** (WorldPop) → "people affected" per event — a
      high-impact, policy-relevant output. 🟡 (reasoner returns null when absent)

### P2 — breadth / robustness
- [ ] **Missing-optical / cloud robustness** — where AlphaEarth's annual prior
      *should* help: degrade/remove event-day S2 and test. ❌ (could flip the
      foundation-model story from negative to "useful when optical is missing")
- [ ] **Cross-sensor / resolution transfer** (Sen1Floods11 10 m ↔ xBD sub-meter)
      framing of "one pipeline, many sensors". ❌

### P3 — polish for submission
- [ ] Failure-case analysis + honest limitations section. 🟡
- [ ] Compute/parameter-efficiency table. 🟡
- [ ] Ethics / responsible-deployment statement (dual-use, false-negative risk). ❌
- [ ] Public code + data + reproducible figures. ✅ (GitHub + auto-blog)

---

## 4. Paper structure (Nature Comms: results-first)

- **Title / Abstract** — system + impact + rigor.
- **Introduction** — disaster-response latency problem; gap (pixels ≠ decisions;
  foundation-model hype vs evidence); our system.
- **Results**
  - R1 — *Decision-level accuracy that wins*: structured decision layer vs
    baselines, on real events vs ground truth (the headline win — like Xu et
    al.'s ROC/AUC, at the decision level).
  - R2 — *The Dispatcher in action*: end-to-end on real events; time-to-answer
    (the impact figure; our edge over the role models).
  - R3 — *Generalization across regions and hazards* (Sen1Floods11 + xBD matrix).
  - R4 — *Label-efficient self-calibration*: the RL policy, significant, vs AL/DA
    baselines.
  - R5 — *What drives performance + a scientific insight*: honest modality /
    foundation-model analysis + a quantified transfer/structured-prior law.
- **Discussion** — impact, generality, honest limitations, responsible use.
- **Methods** — data, models, RL formulation, reasoning graph, evaluation.
- **Main figures (6–8):** (1) system schematic + time-to-answer; (2) cross-
  region/hazard matrix; (3) RL label-efficiency vs baselines w/ CIs; (4) modality
  / foundation-model analysis; (5) real-event qualitative panels; (6) reasoning-
  answer accuracy. Extended data: ablations, calibration, failure cases.

---

## 5. Phasing (suggested order of work)

1. **Finish in-flight:** xBD cross-hazard matrix; AlphaEarth label-eff (✅ done,
   negative); decision-vs-GT seed (USA, running); fold into blog.  ← *now*
2. **Build the winning decision layer + baselines (the headline).** Generalize
   `validate_decisions_vs_gt` into a structured decision-inference method; define
   the 3 baselines; show a decision-level win vs GT on USA, then more events.
   ← *the make-or-break for matching the role models*
3. **P0 ground-truth scale-up:** Sen1Floods11 real floods (have GT) + Copernicus
   EMS + xBD → ≥8 global multi-hazard events; perception + decision accuracy +
   time-to-answer.
4. **P1 method strengthening:** AL/DA baselines + generalized RL + significance on
   both benchmarks; calibration figure; population exposure.
4. **P2 robustness:** missing-optical experiment (the foundation-model redemption
   test); population exposure.
5. **Write + internal review + ultrareview; submit.**

## 6. Honest risk register
- **Biggest risk:** without real-event ground-truth validation + a credible
  time-to-answer number, this is a benchmark paper, not a Nature paper. P0 is the
  make-or-break.
- The RL gain is currently small (+0.023 F1); needs baselines + generalization to
  carry a "method" claim, OR stays a supporting result behind the system.
- Reasoning layer depends on OSM completeness — must report where it fails.
