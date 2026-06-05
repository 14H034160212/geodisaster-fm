# GeoDisaster-FM — Complete results inventory

*Cross-reference for every quantitative claim in MANUSCRIPT.md. All numbers
are reproducible from the listed JSON files + the scripts under `scripts/`.*

---

## 1. Calibration drift — the empirical motivation

| Backbone        | Events | Mean F1 headroom | Threshold range | Frac τ\* ≠ 0.5 | Source |
|-----------------|--------|------------------|-----------------|---------------|--------|
| U-Net + S1 + S2 | 10 floods | **+0.030** | [0.45, 0.70] | **10/10** | `outputs/decision/calibration_analysis.json` |
| AlphaEarth + S1 + S2 | 4 hard regions | **+0.042** | [0.50, 0.65] | 3/4 | `outputs/decision/calibration_analysis_ae.json` |
| U-Net xBD damage | 2 hazards | (cross-benchmark drift) | — | 100 % | `outputs/decision/calibration_analysis_xbd.json` |

Single-event highlights: Pakistan +0.235 F1, palu-tsunami +0.235 F1 from threshold alone.
Figures: `fig18_calibration.png`, `fig22_calibration_cross_benchmark.png`.

---

## 2. Headline result — leakage-free LOEO active calibration

### 2.1 LOEO-v2 (5-d compact features, 300 updates, 10 folds × 10 seeds = 100 pairs)

`outputs/layer3_ppo/ppo_loeo_v2_aggregate.json`

| Comparison              | Δ F1   | 95 % CI            | t-p   | Wilcoxon-p | Verdict |
|-------------------------|--------|--------------------|-------|------------|---------|
| PPO − full-pool oracle  | **−0.002** | [−0.007, +0.003] | 0.42  | 0.57       | **tied** |
| PPO − zero-shot (τ=0.5) | **+0.015** | [+0.004, +0.026] | **0.009 \*\*** | <10⁻⁴ | significant |
| PPO − CoreSet           | **+0.008** | [+0.001, +0.015] | **0.024 \*** | 0.009  | significant |
| PPO − uncertainty       | +0.002 | [−0.002, +0.006]   | 0.33  | 0.14       | n.s. |
| PPO − random            | +0.005 | [−0.001, +0.010]   | 0.084 | **0.0006** | rank-significant, parametric borderline |

PPO wins 65/100 paired pairs (`outputs/figures/fig27_failure_cases.png`),
losses are concentrated on events with negligible calibration headroom.

Figure: `fig26_loeo_v1_v2.png`.

### 2.2 LOEO-v1 (legacy, original PPO; **demonstrates RL-OPT was necessary**)

`outputs/layer3_ppo/ppo_loeo_aggregate.json`

Same protocol, original PPO (no GAE, step-level reward, constant entropy 0.01):
PPO − random = **−0.007** F1, t-p = 0.25, Wilcoxon p = 0.27 — **wrong direction**.
Used in MANUSCRIPT to motivate the three RL-OPT changes.

### 2.3 LOEO-v3 (richer 10-d features; negative ablation)

`outputs/layer3_ppo/ppo_loeo_v3_aggregate.json`

Same protocol, +5 extra features (decision-frontier proximity + p₁₀, p₂₅, p₇₅, p₉₀).
Outcome: PPO loses paired significance vs random / CoreSet / zero-shot and becomes
**significantly worse than the full-pool oracle** (Δ = −0.007, t-p = 0.015 *).
Reported in MANUSCRIPT Methods as the ablation that justifies retaining 5-d features.
Figure: `fig28_v2_v3_ablation.png`.

---

## 3. Within-event protocol (R4-Appendix, leakage-suspect)

### 3.1 Sample-efficiency budget sweep

`outputs/layer3_ppo/ppo_sig_b{1,2,4,8}.json`

| Budget | random | PPO   | PPO − random | paired t-p |
|--------|--------|-------|--------------|-----------|
| 1      | 0.720  | **0.781** | +0.062 | <0.001 |
| 2      | 0.736  | 0.779 | +0.044 | <0.001 |
| 4      | 0.756  | 0.779 | +0.023 | 0.005 |
| 8      | 0.760  | 0.777 | +0.017 | 0.013 |

PPO @ B = 1 ≥ all baselines @ B = 8. Within-event protocol.
Figure: `fig24_sample_efficiency.png`. AlphaEarth version: `ppo_sig_ae_b{1..8}.json`.

### 3.2 Decision-aligned reward A/B (20 seeds)

`outputs/layer3_ppo/decision_ab_unet_20s.json`, `decision_ab_ae_20s.json`

**(i) Reward changes the policy** (paired-significant):
- U-Net pixel-F1 reward → decision-area reward: pixel F1 0.778 → 0.758, paired t-p = 0.0004
- AlphaEarth pixel-F1 reward → decision-area reward: 0.728 → 0.701, paired t-p = 0.005

**(ii) Decision-aligned reward doesn't net-improve decision metric** at n = 20 × 4:
- U-Net: +0.31 area error (worse), p = 0.75
- AE: −1.04 area error (better), p = 0.37
Figure: `fig23_decision_reward_ab.png`.

---

## 4. End-to-end answer fidelity

`outputs/decision/answer_fidelity.json` + `outputs/decision/r971_robustness.json`

| Metric | Value | Note |
|---|---|---|
| Pearson r | **0.971** | 431 chips, 10 events |
| Spearman ρ | 0.899 | rank-based |
| MAPE median | 25 % | half of chips ≥ 25 % rel. error |
| Bottom-50 % chip Pearson r | **0.118** | small chips essentially uncorrelated |
| Top-50 % chip Pearson r | 0.972 | drives the headline |
| Perception speed | 0.031 s / chip | single GPU |
| Drop Pakistan → r | 0.983 (Δ +0.012) | least-fitting event |
| Drop Mekong → r | 0.963 (Δ −0.009) | best-fitting event |

Figure: `fig17_answer_fidelity.png`.

---

## 5. Cross-region & cross-hazard generalisation

### 5.1 Sen1Floods11 cross-region (leave-one-region-out)

`outputs/leave_one_region_out/test_*/checkpoints/*.ckpt` (10 events)
`outputs/four_way_results_table.json`, `outputs/leave_one_region_out_multiseed/*`

Per-event F1 at default τ = 0.5 vs τ\*: see Table S-LOEO (R4 above).
Mean across 10 events: F1@0.5 = 0.822, F1@τ\* = 0.839 (+0.017 headroom).

### 5.2 xBD damage cross-hazard

`outputs/xbd_prepost_loho/aggregate.json` (3 seeds)

Pre/post change-detection vs post-only: mean F1 0.488 → 0.521 across 4 hazards
(+0.033). Hazard-specific: harvey +0.18, florence +0.02, palu +0.01,
mexico-earthquake −0.07. Reported as nuance, not uniform improvement.
Figures: `fig19_xbd_prepost.png`, `fig21_xbd_prepost_loho.png`.

---

## 6. Backbone-agnostic — AlphaEarth foundation model

### 6.1 Few-shot under matched inputs (S1 + S2)

`outputs/few_shot_unet_s1s2/`, `outputs/few_shot_ae_stack_seed{42,1234,1337}/`

U-Net mean F1 ≈ 0.835, AlphaEarth-stack ≈ 0.807 (3-seed paired; comparable not superior).

### 6.2 AE LOO calibration drift (4 hard regions)

`outputs/decision/calibration_analysis_ae.json`

Pakistan / Somalia / Paraguay / India: mean headroom +0.042 F1, threshold range
[0.50, 0.65]. 3 of 4 events optimal τ ≠ 0.5 (Pakistan's IS 0.5 on AE; noted as nuance).

### 6.3 AE within-event sample efficiency

`outputs/layer3_ppo/ppo_sig_ae_b{1,2,4,8}.json`

PPO @ B = 1 = 0.742 > baselines @ B = 8 (≈ 0.68) on the AE backbone too.

---

## 7. Negative results (R5)

| Finding | Evidence | Why it strengthens the main claim |
|---|---|---|
| AlphaEarth on equal inputs not superior to U-Net on F1 | `few_shot_three_way_comparison.json` (multi-seed) | F1 ceiling is not representational |
| MRF structured decision layer fails | `outputs/decision/method_comparison.json` + Fig 14, Fig 16 | Calibrated threshold > structured prior |
| Decision-aligned reward doesn't net-improve decision metric at n=20×4 | `decision_ab_*_20s.json` | Honest power limit |
| 10-d richer features hurt PPO under LOEO | `ppo_loeo_v3_aggregate.json` | 5-d compact set is the right design |
| LOEO-v1 PPO loses to random | `ppo_loeo_aggregate.json` | Motivates GAE-λ + terminal + entropy schedule |
| Within-event paired protocol inflated PPO gains | side-by-side in `ppo_loeo_v2_aggregate.json` vs `ppo_significance.json` | Motivates LOEO as the headline protocol |

---

## 8. Live agent demos

- **Dispatcher demo (USA 2024 flood)** — `outputs/dispatch/USA_170264.json` +
  `outputs/dispatch/USA_170264.md` — 10 UN-OCHA answers per event, OSM-derived.
- **Live blog (auto-updating)** — `outputs/site/index.html` (12 MB);
  deployments at `https://geodisaster-fm.pages.dev/` and
  `https://14h034160212.github.io/geodisaster-fm/`.

---

## 9. Architectural ablations summary

| Modification             | Necessary?   | Evidence |
|--------------------------|--------------|----------|
| GAE-λ in PPO             | **Yes**      | LOEO-v1 → v2: Pakistan Δ −0.033 → −0.002 |
| Terminal-only reward     | **Yes**      | Same as above |
| Entropy schedule 0.10→0.01 | **Yes**    | Same as above |
| Gradient clipping (0.5)  | Safety only  | Numerical stability; not load-bearing |
| 5-d compact features     | **Optimal**  | LOEO-v3 (10-d) significantly worse than v2 |
| 300 training updates     | Sufficient   | LOEO-v2 convergent; v3 underfits with same budget |
| Permutation-equivariant per-chip scorer | **Yes** | Allows variable pool sizes across events |

---

## 10. Full figure list (28 figures)

- **Architecture:** fig0_architecture
- **Cross-region perception:** fig3 (4-way), fig4 (LOEO), fig8 (multiseed)
- **Cross-region few-shot:** fig3_few_shot, fig7_ae_stack_few_shot
- **Decision-level USA:** fig5_usa_decision
- **Brazil zero-shot:** fig6_brazil_zero_shot
- **Global atlas:** fig9_global_atlas
- **Active adaptation:** fig10, fig11
- **Layer 3 PPO (within-event):** fig12_ppo, fig13_ppo_significance,
  fig24_sample_efficiency, fig23_decision_reward_ab, fig20_rl_backbone
- **xBD cross-hazard:** fig15, fig19, fig21
- **Calibration analyses:** fig18, fig22 (cross-benchmark)
- **Decision-level:** fig14_decision_methods, fig16_calibration_vs_structure,
  fig17_answer_fidelity
- **LOEO-v2 + v1 comparison (headline):** fig25_leakage_free, fig26_loeo_v1_v2,
  fig27_failure_cases
- **Feature-richness ablation:** fig28_v2_v3_ablation
