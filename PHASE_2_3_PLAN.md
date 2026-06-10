# Phase 2 + 3 — Pre-submission strengthening plan

*Generated 2026-06-10. Goal: address Nat-Commun referee concerns MW1
(hazard scope) and MW2 (backbone scope) before submission, raising
expected acceptance probability from ~30 % to ~55–65 %.*

---

## Phase 2 — New hazard family (MW1)

### Goal
Add at least one non-flood, non-xBD hazard family to the cross-event
LOEO panel so that the H2-dominates finding is supported across **three**
benchmarks rather than two.

### Candidate datasets (ranked by ROI)

| # | Dataset | Hazard | Size | Inputs | Notes |
|---|---|---|---|---|---|
| 1 | **Landslide4Sense** | landslide | ~3,800 chips, 4 train/val/test events | S2 12-band + DEM | Public, binary segmentation, well-defined; CVPR 2022 benchmark |
| 2 | **CalFire / EFFIS wildfire perimeter** | wildfire | ~150 events globally (variable size) | S1 + S2 + MODIS hotspots | Public, binary segmentation, requires per-event clip prep |
| 3 | **MMFlood / Floods+** | extended flood | ~95 additional events | S1 + S2 | Extends existing benchmark — does not really add a new hazard family |
| 4 | **SEN12-FLOOD-EXT** | flood | ~10 events | S1 + S2 | Like #3 |

**Selection:** **Landslide4Sense** (option 1). Best fit because:
- Hazard *family* is genuinely different from flood + structural damage
- Binary segmentation task — compatible with our binary-threshold
  calibration equivalence proof
- Dataset is small enough to train + LOEO in 1 week of CPU/GPU time
- Public + DOI'd + cited

### Phase 2 deliverables

1. `data/processed/patches/landslide4sense_*` — patched landslide events
2. `outputs/leave_one_event_out_landslide/test_<event>/checkpoints/` —
   trained U-Net LOEO checkpoints for each held-out event
3. `outputs/decision/calibration_analysis_landslide.json` — per-event
   τ\* + F1@0.5 vs F1@τ\* (the H2(b) test on the new hazard)
4. `outputs/layer3_ppo/chip_cache_landslide.pkl` — chip cache
5. `outputs/layer3_ppo/ppo_loeo_landslide_aggregate.json` — full
   leave-one-event-out 100/200-pair on the new hazard
6. New panel in Fig 1: H2(b) recalibration gain on landslide events
7. MANUSCRIPT R5 — replace "two benchmarks, twelve events" with
   "three benchmarks, ≥18 events"

### Phase 2 timeline

| Step | Work | Time |
|---|---|---|
| 2.1 | Acquire Landslide4Sense (HuggingFace mirror) + organise into patch manifests | 1 day |
| 2.2 | Adapt U-Net config (3-channel optical RGB or 12-band S2) + train LOEO | 2-3 days GPU |
| 2.3 | Run calibration_analysis on the new events | 1 day |
| 2.4 | Build chip_cache, run LOEO-v2 + ensemble baseline | 1-2 days CPU |
| 2.5 | Update MANUSCRIPT R3 + R5 + Methods + Fig 1 | 1 day writing |
| | **Total** | **~1-2 weeks** |

---

## Phase 3 — Multi-backbone H1 falsification (MW2)

### Goal
Show that the H1 falsification is not specific to AlphaEarth. Run the
same H1 test (foundation model on matched inputs vs trained U-Net) on
at least **two additional foundation models**.

### Candidate foundation models (ranked by ROI)

| # | Foundation | Source | Pretraining | Notes |
|---|---|---|---|---|
| 1 | **Prithvi** (NASA-IBM) | HuggingFace public | HLS S2-only, 100 M pixels | Strongest competitor; geo-aware; segmentation-ready |
| 2 | **DOFA** | HuggingFace public | Multi-modal Earth Obs (S1/S2/L8/MODIS) | Multi-modal — closest match to our S1+S2 setting |
| 3 | **SatMAE** | Stanford repo | Functional spectral pretraining | Masked autoencoder; older but cited; well-supported |
| 4 | **USat** | Recent paper | Joint multi-sensor SSL | Newer; less established eval |
| 5 | **CrossEarth** | Pre-print | Geo-spatial vision foundation | Pre-print quality; deferred |

**Selection:** **Prithvi + DOFA** (two strongest representatives:
NASA-grade single-model + multi-modal).

### Phase 3 deliverables

1. `outputs/leave_one_region_out_prithvi/` — Prithvi LOO checkpoints
2. `outputs/leave_one_region_out_dofa/` — DOFA LOO checkpoints
3. `outputs/decision/calibration_analysis_prithvi.json` and `..._dofa.json`
4. Updated R3 (R3 currently has only AE) to show **three foundation
   models all comparable to U-Net on F1, all exhibiting calibration
   headroom** — i.e. H1 falsification on three backbones, not one
5. New supplementary figure: 4-way backbone comparison (U-Net /
   AlphaEarth / Prithvi / DOFA) per event

### Phase 3 timeline

| Step | Work | Time |
|---|---|---|
| 3.1 | Set up Prithvi + DOFA inference pipelines (HuggingFace models, our patch loader) | 1-2 days |
| 3.2 | Train per-region LOO heads on Prithvi + DOFA frozen embeddings | 2-3 days GPU |
| 3.3 | Run calibration_analysis on Prithvi + DOFA | 1 day |
| 3.4 | Update MANUSCRIPT R3 + R5 + Methods + supplementary figures | 1 day writing |
| | **Total** | **~1-2 weeks** |

---

## Phase 4 — Cover-letter + submission package (MW7-adjacent)

- Author affiliations + reviewer suggestions (need from advisor)
- Nature-Communications official LaTeX template conversion
- Final read-through against the COVER_LETTER.md
- One head-to-head EMS comparison (optional — strengthens MW7)

Timeline: ~3 days

---

## Total timeline + expected acceptance probability

| Phase | Calendar weeks | Acceptance probability |
|---|---|---|
| 0 — Phase 1 writing fixes (this session) | — | ~30–35 % |
| 1 — Phase 2 done (landslide added) | +1–2 weeks | ~40–45 % |
| 2 — Phase 3 done (Prithvi + DOFA added) | +2–3 weeks | **55–65 %** |
| 3 — Phase 4 done (cover letter + LaTeX) | +0.5 week | ~60–70 % |

**Realistic critical-path total: 4–5 weeks** to submission-ready.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Landslide4Sense download + preprocessing slower than expected | Have CalFire as backup hazard option |
| Prithvi/DOFA segmentation head training does not converge | Use simpler frozen-embedding + per-pixel MLP head (same as our AlphaEarth recipe) |
| Phase 3 reveals one foundation model DOES exceed U-Net | This is a *good* outcome — paper becomes "H2 dominates for these backbones; here is the specific backbone where representation matters" |
| Advisor wants additional experiments mid-stream | Defer Phase 4 (cover letter) until experiments locked |

---

## How we know we're done

A second-round reviewer will be unable to write either of these two
sentences:

1. "The hazard scope is one benchmark family." → addressed by Phase 2.
2. "The H1 falsification is single-backbone." → addressed by Phase 3.

When the manuscript can answer both with cited per-event numbers in
new tables, we are submission-ready.
