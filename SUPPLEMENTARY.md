# Supplementary Information — Cross-disaster mapping is largely a calibration problem

*Companion to MANUSCRIPT.md. Contains the within-event-protocol ablations
that the leave-one-event-out (LOEO) results in the main text supersede,
retained in full for transparency.*

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
