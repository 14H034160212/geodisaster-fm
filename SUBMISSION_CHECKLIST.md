# Nature Communications submission checklist

*Updated 2026-06-19. Items marked ☐ need human input before clicking submit.*

## Manuscript

- [x] 149-word abstract (limit ~150; no citations)
- [x] Title acknowledges hazard-specificity ("largely")
- [x] Four-test hypothesis structure (H1 ranking / representation /
      information-budget / zero-label)
- [x] All internal cross-references consistent
- [x] Nature-style concise thematic Results/Discussion subheadings
      (R# scaffolding + Overview table removed; in-text + caption refs
      rewritten as descriptive phrases)
- [x] Section order Introduction → Results → Discussion → Methods
- [x] Within-event ablations moved to SUPPLEMENTARY.md
- [x] Official Nature Portfolio template (sn-jnl class, sn-nature style)
- [x] **Single-file Overleaf-proof `main.tex`** (body + 7 figures inlined;
      no \input; Unicode→LaTeX substituted; calc/\tightlist handled) —
      compiles 0-error: empty-stack/undefined-cs/unicode/missing-fig/
      undefined-cite/fatal all 0 → clean 34-page PDF
- [x] System architecture figure (Fig 2) + 6 result figures, all with
      formal captions
- [x] 24-entry bibliography, sn-nature.bst, all @article (no @inproceedings
      empty-stack bug)
- [x] Self-contained zip verified to compile standalone:
      `nature_submission_20260618.zip`
- ☐ Final English copy-edit pass (optional polish)

## Authorship / affiliations(需要作者提供)

- ☐ Affiliation 1 (Qiming Bao)
- ☐ Affiliation 2 + corresponding email (Yanbing Bai)
- ☐ Author contributions statement (replace TODO in manuscript)
- ☐ Competing interests: confirm "none"
- ☐ 3–5 suggested reviewers + exclusions (with advisor)

## Data & code availability

- [x] GitHub repo public: https://github.com/14H034160212/geodisaster-fm
- [x] Live dashboard: https://geodisaster-fm.pages.dev/
- [x] All result JSONs committed; every claim traceable
      (RESULTS_INVENTORY.md)
- [x] Third-party data all public: Sen1Floods11 (CC), xBD (CC),
      HLS Burn-Scars (CC-BY-4.0), AlphaEarth/Prithvi/DOFA weights (public)
- ☐ Archive a release tag + Zenodo DOI at submission time

## Reporting standards

- ☐ Nature Portfolio reporting summary form (filled at portal)
- [x] Statistical tests stated per claim (paired t + Wilcoxon, n, CI)
- [x] Seeds / protocol / splits documented in Methods
- [x] Ethics & dual-use statement present

## Cover letter

- [x] Four-test summary with zero-label decomposition selling point
- ☐ Affiliations + signatures
- ☐ Suggested reviewers section

## Known accepted-risk items (documented, not blocking)

- Wildfire calibration lever is small (+0.001) — framed as
  hazard-specific magnitude, supported by "largely" in title
- PPO ties random in mean — framed honestly; headline does not depend
  on PPO superiority
- U-Net gradient entry is single-seed — flagged in figure caption
- DOFA absolute deficit partly pipeline-attributable — two caveats in R5
