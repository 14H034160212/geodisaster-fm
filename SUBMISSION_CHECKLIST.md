# Nature Communications submission checklist

*Updated 2026-06-13. Items marked ☐ need human input before clicking submit.*

## Manuscript

- [x] 149-word abstract (limit ~150)
- [x] Title acknowledges hazard-specificity ("largely")
- [x] Four-test hypothesis structure (H1 ranking / representation /
      information-budget / zero-label)
- [x] All internal cross-references consistent (round-3 audit)
- [x] Within-event ablations moved to SUPPLEMENTARY.md
- [x] Official Nature Portfolio template build (`latex_sn/`, sn-jnl class, sn-nature style): compiles to 42-page PDF with all tables + figures + bibliography (16 refs, sn-nature.bst)
- [x] Generic LaTeX build also retained (`latex/main.tex`)
- [x] Six main figures with formal captions (`latex/figures.tex`)
- ☐ Convert `[Author Year]` text citations to `\citep{}` (optional —
      Nature Portfolio accepts format-free initial submissions)
- ☐ Final English copy-edit pass

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
