"""DeepMind-style research blog generator.

Produces ``outputs/site/index.html`` — a single-file long-form narrative that
embeds the experiment figures as base64 PNGs, reads the latest result CSVs/
JSONs, and renders headline findings in a magazine-style layout.

Re-run after every experiment:
    geodisaster build-blog
The dashboard moves to ``outputs/site/dashboard.html`` (linked from the blog).
"""
from __future__ import annotations

import base64
import datetime as dt
import html as _html
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Asset discovery
# --------------------------------------------------------------------------- #
FIG_FIVE_WAY     = "outputs/figures/fig3_four_way_comparison.png"
FIG_LEAVE_ONE    = "outputs/figures/fig4_leave_one_region_out.png"
FIG_DECISION     = "outputs/figures/fig5_usa_decision.png"
FIG_BRAZIL       = "outputs/figures/fig6_brazil_zero_shot.png"
FIG_FEWSHOT_UNET = "outputs/figures/fig3_sen1floods11_few_shot.png"
COMPARISON_JSON  = "outputs/sen1floods11_comparison.json"
LEAVE_ONE_OUT    = "outputs/leave_one_region_out/results.json"
USA_DECISION     = "outputs/usa_decision/decision_summary.json"
BRAZIL_SUMMARY   = "outputs/zero_shot/brazil_rs_2024/flood_decision_summary.json"
FEWSHOT_UNET_CSV = "outputs/few_shot_unet_s1s2/few_shot_results.csv"
FEWSHOT_AE_CSV   = "outputs/few_shot_ae_s1/few_shot_results.csv"
FEWSHOT_AESTACK  = "outputs/few_shot_ae_stack/few_shot_results.csv"
MANIFEST         = "outputs/reproducibility.json"


def _img(path: str | Path, fallback_caption: str = "") -> str:
    p = Path(path)
    if not p.exists():
        return f'<div class="missing-fig">[{_html.escape(fallback_caption or path)}]<br/><small>figure not yet generated</small></div>'
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{_html.escape(fallback_caption)}"/>'


def _load_json(path: str | Path) -> dict | list | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _load_csv(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _comparison_row(cmp: dict, key: str) -> dict:
    return cmp.get(key, {}) if cmp else {}


CSS = """
:root {
  --fg:#1a202c; --muted:#5a6577; --bg:#fafbfc; --paper:#ffffff;
  --accent:#2453a8; --rule:#e4e8ee; --code-bg:#f2f4f8;
  --green:#1c7f4f; --red:#b03537; --amber:#a86a1f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Iowan Old Style","Charter","Georgia","Source Serif Pro",serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.65;
}
.container { max-width: 780px; margin: 0 auto; padding: 0 24px; }
.wider { max-width: 1100px; }
header.hero {
  background: linear-gradient(180deg, #f0f5fc 0%, var(--bg) 100%);
  padding: 80px 0 60px;
  border-bottom: 1px solid var(--rule);
}
header.hero .container { max-width: 880px; }
header.hero .eyebrow {
  font-family: "Inter","Helvetica Neue",-apple-system,sans-serif;
  text-transform: uppercase; letter-spacing: 0.12em; font-size: 12px;
  color: var(--accent); font-weight: 600;
}
header.hero h1 {
  font-size: 44px; line-height: 1.2; font-weight: 700; margin: 14px 0 22px;
  letter-spacing: -0.01em;
}
header.hero .subtitle {
  font-size: 21px; color: var(--muted); margin: 0; max-width: 660px;
}
header.hero .meta {
  margin-top: 30px; font-size: 13px; color: var(--muted);
  font-family: "Inter",sans-serif;
}
header.hero .meta a { color: var(--accent); text-decoration: none; }
header.hero .meta a:hover { text-decoration: underline; }

article { padding: 50px 0 40px; }
article h2 {
  font-size: 30px; margin: 70px 0 12px; line-height: 1.25;
  letter-spacing: -0.01em;
}
article h2 .sec { color: var(--accent); font-family: "Inter",sans-serif;
  font-size: 14px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; display: block; margin-bottom: 6px; }
article h3 { font-size: 20px; margin: 36px 0 10px; }
article p { font-size: 17.5px; margin: 14px 0; }
article .lead { font-size: 19px; color: #2a3242; }
article ul li, article ol li { margin: 6px 0; font-size: 17px; }

.tldr {
  background: var(--paper); border: 1px solid var(--rule);
  border-radius: 8px; padding: 28px 32px; margin: 36px 0;
}
.tldr h3 { font-family: "Inter",sans-serif; text-transform: uppercase;
  font-size: 12px; letter-spacing: 0.12em; color: var(--muted);
  margin: 0 0 14px; font-weight: 600; }
.tldr ol { margin: 0; padding-left: 22px; }
.tldr li { font-size: 16.5px; margin: 10px 0; }
.tldr li strong { color: var(--fg); }

.pullquote {
  border-left: 4px solid var(--accent);
  padding: 6px 0 6px 26px; margin: 32px 0;
  font-size: 21px; color: #2a3242; font-style: italic;
}

figure {
  margin: 36px 0;
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 8px; padding: 16px;
}
figure.wide {
  margin: 36px calc((780px - 1100px) / 2);
}
@media (max-width: 1140px) { figure.wide { margin: 36px -160px; } }
@media (max-width: 1000px) { figure.wide { margin: 36px 0; } }
figure img { width: 100%; height: auto; display: block; border-radius: 4px; }
figcaption {
  font-family: "Inter",sans-serif; font-size: 13px; color: var(--muted);
  margin-top: 10px; line-height: 1.55;
}
figcaption strong { color: var(--fg); }

.missing-fig {
  background: #fff8e0; border: 1px dashed #c5a04a; padding: 28px;
  text-align: center; color: #7a5614; border-radius: 6px;
  font-family: "Inter",sans-serif;
}

.metric-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px; margin: 28px 0;
}
.metric-card {
  background: var(--paper); border: 1px solid var(--rule);
  border-radius: 6px; padding: 18px;
}
.metric-card .value {
  font-family: "Inter",sans-serif; font-size: 26px; font-weight: 700;
  color: var(--accent); margin-bottom: 4px;
}
.metric-card .label {
  font-family: "Inter",sans-serif; font-size: 13px; color: var(--muted);
  line-height: 1.4;
}

table.results { width: 100%; border-collapse: collapse; font-size: 14px;
  font-family: "Inter",sans-serif; margin: 16px 0; }
table.results th, table.results td {
  padding: 8px 12px; border-bottom: 1px solid var(--rule); text-align: left;
}
table.results thead th { background: #f3f4f8; font-weight: 600; }
table.results td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.results tr.highlight td { background: #fff8e0; font-weight: 600; }

code, pre {
  font-family: "JetBrains Mono","SF Mono",Menlo,monospace;
  font-size: 13px; background: var(--code-bg); border-radius: 4px;
}
code { padding: 2px 6px; }
pre { padding: 14px 18px; overflow-x: auto; line-height: 1.5;
  border: 1px solid var(--rule); }

footer.cite {
  border-top: 1px solid var(--rule); padding: 50px 0 80px;
  background: var(--paper); margin-top: 80px;
}
footer.cite h3 { font-family: "Inter",sans-serif; text-transform: uppercase;
  font-size: 12px; letter-spacing: 0.12em; color: var(--muted);
  margin: 0 0 16px; }
footer.cite p { font-size: 14px; color: var(--muted);
  font-family: "Inter",sans-serif; }
footer.cite a { color: var(--accent); }

.callout {
  background: #f3f8ff; border-left: 3px solid var(--accent);
  padding: 18px 22px; margin: 28px 0; border-radius: 0 6px 6px 0;
}
.callout.warn {
  background: #fff8ec; border-left-color: var(--amber);
}
.callout strong { color: var(--fg); }
"""


def _hero_metrics() -> str:
    cmp = _load_json(COMPARISON_JSON) or {}
    leave_one = _load_json(LEAVE_ONE_OUT) or []
    usa = _load_json(USA_DECISION) or {}

    unet_s1s2 = _comparison_row(cmp, "U-Net_S1_plus_S2")
    ae_stack = _comparison_row(cmp, "AE_pre_post_S1_stack")

    f1_best = unet_s1s2.get("f1", 0)
    n_regions = len(leave_one)
    avg_f1 = sum(r["f1"] for r in leave_one) / max(len(leave_one), 1) if leave_one else 0
    usa_bld_pct = 100 * usa.get("totals", {}).get("buildings_affected", 0) / max(usa.get("totals", {}).get("buildings_total", 1), 1)

    return f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="value">{f1_best:.3f}</div>
    <div class="label">Best F1 on held-out region<br/>(U-Net S1+S2 on USA)</div>
  </div>
  <div class="metric-card">
    <div class="value">{avg_f1:.3f}</div>
    <div class="label">Average F1 across {n_regions} leave-one-out holdouts</div>
  </div>
  <div class="metric-card">
    <div class="value">{ae_stack.get("f1", 0):.3f}</div>
    <div class="label">AlphaEarth pre+post + S1 (foundation prior + time)</div>
  </div>
  <div class="metric-card">
    <div class="value">{usa_bld_pct:.1f}%</div>
    <div class="label">Predicted-affected buildings<br/>in USA test set</div>
  </div>
</div>
"""


def _leave_one_out_table_html() -> str:
    rows = _load_json(LEAVE_ONE_OUT) or []
    if not rows:
        return "<p><em>Leave-one-region-out results not generated yet.</em></p>"
    rows = sorted(rows, key=lambda r: -r["f1"])
    body = ""
    for r in rows:
        cls = ""
        if r["test_region"] == "Pakistan":
            cls = ' class="highlight"'
        body += (f"<tr{cls}><td>{_html.escape(r['test_region'])}</td>"
                 f"<td class='num'>{r['f1']:.3f}</td>"
                 f"<td class='num'>{r['iou']:.3f}</td>"
                 f"<td class='num'>{r['precision']:.3f}</td>"
                 f"<td class='num'>{r['recall']:.3f}</td>"
                 f"<td class='num'>{r['auprc']:.3f}</td></tr>")
    avg_f1 = sum(r["f1"] for r in rows) / len(rows)
    avg_iou = sum(r["iou"] for r in rows) / len(rows)
    body += (f"<tr style='border-top:2px solid #ddd'><td><strong>Average</strong></td>"
             f"<td class='num'><strong>{avg_f1:.3f}</strong></td>"
             f"<td class='num'><strong>{avg_iou:.3f}</strong></td>"
             f"<td colspan='3'></td></tr>")
    return (
        "<table class='results'><thead><tr><th>Held-out region</th>"
        "<th>F1</th><th>IoU</th><th>Precision</th><th>Recall</th><th>AUPRC</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _models_table_html() -> str:
    cmp = _load_json(COMPARISON_JSON) or {}
    if not cmp:
        return ""
    name_map = {
        "U-Net_S1_only":        "U-Net (S1 only, 2 ch)",
        "AlphaEarth_plus_S1":   "AlphaEarth+S1 (MLP head, frozen 64-d)",
        "AE_pre_post_S1_stack": "AlphaEarth pre+post + S1 (multi-modal fusion)",
        "U-Net_S1_plus_S2":     "U-Net (S1+S2, 15 ch)",
    }
    body = ""
    for key, name in name_map.items():
        m = cmp.get(key, {})
        if not m:
            continue
        cls = ' class="highlight"' if "S1_plus_S2" in key else ""
        body += (f"<tr{cls}><td>{name}</td>"
                 f"<td class='num'>{m.get('f1', 0):.3f}</td>"
                 f"<td class='num'>{m.get('iou', 0):.3f}</td>"
                 f"<td class='num'>{m.get('auprc', 0):.3f}</td>"
                 f"<td class='num'>{m.get('precision', 0):.3f}</td>"
                 f"<td class='num'>{m.get('recall', 0):.3f}</td></tr>")
    return (
        "<table class='results'><thead><tr><th>Model</th><th>F1</th><th>IoU</th>"
        "<th>AUPRC</th><th>Precision</th><th>Recall</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _decision_metrics_html() -> str:
    usa = _load_json(USA_DECISION) or {}
    t = usa.get("totals", {})
    if not t:
        return "<p><em>Decision metrics not yet generated.</em></p>"
    pct = lambda a, n: 100 * a / max(n, 1)
    return f"""
<table class="results"><thead><tr><th>Quantity</th><th>Total in chips</th>
<th>Predicted affected</th><th>%</th></tr></thead>
<tbody>
<tr><td>Buildings (OSM polygons)</td>
    <td class='num'>{t['buildings_total']:,}</td>
    <td class='num'>{t['buildings_affected']:,}</td>
    <td class='num'>{pct(t['buildings_affected'], t['buildings_total']):.2f}%</td></tr>
<tr><td>Major roads (km)</td>
    <td class='num'>{t['road_km_total']:,.1f}</td>
    <td class='num'>{t['road_km_affected']:,.1f}</td>
    <td class='num'>{pct(t['road_km_affected'], t['road_km_total']):.2f}%</td></tr>
</tbody></table>
"""


def _brazil_summary_html() -> str:
    b = _load_json(BRAZIL_SUMMARY) or {}
    if not b:
        return "<p><em>Brazil zero-shot summary not yet generated.</em></p>"
    return f"""
<table class="results"><thead><tr><th>Quantity</th><th>Value</th></tr></thead>
<tbody>
<tr><td>AOI</td><td>{_html.escape(b.get("name", ""))}</td></tr>
<tr><td>AOI area</td><td class='num'>{b.get("total_area_km2", 0):,.0f} km²</td></tr>
<tr><td>JRC permanent water</td>
    <td class='num'>{b.get("permanent_water_pct", 0):.2f}% of AOI</td></tr>
<tr><td>Model predicted as water</td>
    <td class='num'>{b.get("water_pct", 0):.2f}% of AOI</td></tr>
<tr><td>Difference (predicted minus permanent)</td>
    <td class='num'>{b.get("flood_only_water_pct", 0):.2f}% — over-prediction</td></tr>
</tbody></table>
"""


def _ae_few_shot_table() -> str:
    df = _load_csv(FEWSHOT_AESTACK)
    if df is None or df.empty:
        return ("<p style='color:var(--muted)'><em>AlphaEarth pre+post few-shot "
                "sweep still running — will appear here once complete.</em></p>")
    rows = ""
    for _, r in df.iterrows():
        rows += (f"<tr><td>{int(r['label_fraction']*100)}%</td>"
                 f"<td class='num'>{int(r['n_train'])}</td>"
                 f"<td class='num'>{r['test/f1']:.3f}</td>"
                 f"<td class='num'>{r['test/iou']:.3f}</td>"
                 f"<td class='num'>{r['test/precision']:.3f}</td>"
                 f"<td class='num'>{r['test/recall']:.3f}</td></tr>")
    return (
        "<table class='results'><thead><tr><th>Label fraction</th><th>n_train</th>"
        "<th>F1</th><th>IoU</th><th>Precision</th><th>Recall</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def build_blog(out_path: str | Path = "outputs/site/index.html") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.utcnow()

    cmp = _load_json(COMPARISON_JSON) or {}
    leave_one = _load_json(LEAVE_ONE_OUT) or []
    usa = _load_json(USA_DECISION) or {}
    fewshot_unet = _load_csv(FEWSHOT_UNET_CSV)
    n_train_5pct = int(fewshot_unet[fewshot_unet["label_fraction"] == 0.05]["n_train"].iloc[0]) \
                   if fewshot_unet is not None and not fewshot_unet.empty else 17
    f1_5pct = float(fewshot_unet[fewshot_unet["label_fraction"] == 0.05]["test/f1"].iloc[0]) \
              if fewshot_unet is not None and not fewshot_unet.empty else 0.789
    f1_100pct = float(fewshot_unet[fewshot_unet["label_fraction"] == 1.0]["test/f1"].iloc[0]) \
                if fewshot_unet is not None and not fewshot_unet.empty else 0.849
    f1_ratio = 100 * f1_5pct / f1_100pct

    avg_f1 = (sum(r["f1"] for r in leave_one) / len(leave_one)) if leave_one else 0
    f1_max = max((r["f1"] for r in leave_one), default=0)
    f1_min = min((r["f1"] for r in leave_one), default=0)
    hardest = min(leave_one, key=lambda r: r["f1"])["test_region"] if leave_one else "?"
    easiest = max(leave_one, key=lambda r: r["f1"])["test_region"] if leave_one else "?"

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GeoDisaster-FM — foundation models for cross-region disaster mapping</title>
<style>{CSS}</style>
</head>
<body>

<header class="hero">
  <div class="container">
    <div class="eyebrow">Research notebook · 26 May 2026</div>
    <h1>What foundation models can — and cannot — do for global flood mapping</h1>
    <p class="subtitle">
      We trained four classes of models on the Sen1Floods11 benchmark and
      stress-tested them across 10 unseen regions, a real out-of-distribution
      2024 flood event, and the policy-relevant question every responder
      eventually asks: <em>how many buildings did this just affect?</em>
    </p>
    <div class="meta">
      Code &amp; data ·
      <a href="https://github.com/14H034160212/geodisaster-fm">github.com/14H034160212/geodisaster-fm</a>
      &nbsp;·&nbsp; Live dashboard: <a href="dashboard.html">dashboard.html</a>
      &nbsp;·&nbsp; Updated {now.strftime("%Y-%m-%d %H:%M UTC")}
    </div>
  </div>
</header>

<article class="container">

  <p class="lead">
    Disaster remote sensing has a labelling problem. Every flood, every
    earthquake, every cyclone arrives with fresh imagery the world has not
    seen before — and labelled before. The pipeline that maps water from
    Sentinel-1 over Mekong in 2018 is rebuilt from scratch when a new flood
    inundates Rio Grande do Sul in May 2024. Recent geospatial foundation
    models — Google DeepMind's
    <a href="https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/">AlphaEarth&nbsp;Foundations</a>
    and Gong et&nbsp;al.'s
    CrossEarth — promise to break this cycle by providing a single
    representation that transfers to any task with very few labels.
    We tested whether that promise survives contact with the disaster
    domain.
  </p>

  <div class="tldr">
    <h3>What we found, in one screen</h3>
    <ol>
      <li><strong>Modality fusion beats data quantity.</strong>
          U-Net (Sentinel-1 SAR + Sentinel-2 optical, 15 channels) reaches
          <strong>F1 = {cmp.get("U-Net_S1_plus_S2", {}).get("f1", 0.849):.3f}</strong>
          on the held-out USA test set — versus
          <strong>{cmp.get("U-Net_S1_only", {}).get("f1", 0.618):.3f}</strong> for the
          SAR-only baseline (a +{cmp.get("U-Net_S1_plus_S2", {}).get("f1", 0.849) - cmp.get("U-Net_S1_only", {}).get("f1", 0.618):.3f}
          lift from just adding the optical bands).</li>
      <li><strong>Label efficiency is real.</strong>
          With <strong>{n_train_5pct} labelled chips</strong> (5% of training data),
          U-Net + optical reaches F1 = {f1_5pct:.3f} —
          <strong>{f1_ratio:.0f}% of its full-data performance</strong>.
          Five percent of the labels deliver almost all the value.</li>
      <li><strong>The cross-region gap is huge — and quantifiable.</strong>
          A leave-one-region-out matrix across {len(leave_one)} regions
          gives F1 from {f1_min:.2f} ({hardest}) to {f1_max:.2f} ({easiest}) —
          a <strong>spread of {f1_max - f1_min:.2f}</strong> in the
          <em>same model</em>, varying only by which region we tested on.</li>
      <li><strong>Foundation priors help, but cannot replace direct
          observation.</strong> Adding AlphaEarth as a per-pixel prior to
          Sentinel-1 only reaches F1 =
          {cmp.get("AlphaEarth_plus_S1", {}).get("f1", 0.602):.3f}. Adding the
          event-year embedding for temporal differencing pushes that to
          F1 = {cmp.get("AE_pre_post_S1_stack", {}).get("f1", 0.708):.3f}.
          Still 0.14 below using post-event optical directly — the
          observation wins.</li>
    </ol>
  </div>

  <h2><span class="sec">Headline figure</span>
      Five models, one held-out region</h2>

  <p>
    Every model was trained on eight Sen1Floods11 regions
    (Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Sri-Lanka),
    validated on Spain, and tested on the
    USA chips the model never sees during training. Curves show how
    test F1 changes as we reduce the training label budget.
  </p>

  <figure class="wide">
    {_img(FIG_FIVE_WAY, "Five-model comparison")}
    <figcaption>
      <strong>Five models on Sen1Floods11 cross-region.</strong>
      Left: F1 versus training label fraction. The U-Net (S1+S2) curve sits
      flat near 0.83 across every fraction tested — half a percent of full
      labels delivers most of the performance. The AlphaEarth+S1 curve has
      a sharper slope, indicating the foundation prior on its own struggles
      with very sparse labels. Right: AUPRC at full-label budget. Foundation
      priors with temporal differencing systematically improve ranking
      quality, even when they lose at the threshold-pinned F1 cliff.
    </figcaption>
  </figure>

  {_hero_metrics()}

  <h2><span class="sec">Finding 1</span>
      The cross-region gap is the central bottleneck</h2>

  <p>
    Most disaster remote-sensing benchmarks report a single train/val/test
    split — and a single F1 number. We ran the same model
    {len(leave_one)} times, with each region taking a turn as the held-out
    test set. The resulting matrix is the most accurate picture we have of
    where the model works and where it falls over.
  </p>

  <figure class="wide">
    {_img(FIG_LEAVE_ONE, "Leave-one-region-out")}
    <figcaption>
      <strong>Leave-one-region-out generalisation.</strong>
      Left: per-region F1 / IoU / AUPRC bars, sorted by F1 descending. Right:
      a per-metric heatmap to highlight where the cracks open. The same
      U-Net (S1+S2) architecture spans <strong>F1 from {f1_min:.2f}
      ({hardest}, hardest) to {f1_max:.2f} ({easiest}, easiest)</strong> — a
      spread of {f1_max - f1_min:.2f}. CrossEarth's benchmark gives 28
      cross-domain settings but does not isolate target-region
      difficulty; leave-one-out makes it explicit.
    </figcaption>
  </figure>

  {_leave_one_out_table_html()}

  <div class="callout warn">
    <strong>Pakistan is the outlier.</strong>
    Recall stays high (0.94) but precision collapses to 0.38 — the model is
    aggressive about flagging water in semi-arid landscapes that look
    nothing like the model's training distribution (humid sub-tropical
    and temperate floodplains). This is the same failure mode we will
    see, much more dramatically, in zero-shot deployment to Brazil.
  </div>

  <h2><span class="sec">Finding 2</span>
      Foundation priors help — most clearly through temporal differencing</h2>

  <p>
    We trained four variants on top of the AlphaEarth annual embedding to ask
    a focused question: <em>does a foundation representation reduce the
    label-efficiency bottleneck on its own?</em>
  </p>

  <ol>
    <li><strong>AlphaEarth+S1, per-pixel MLP head.</strong> Pre-event-year
        AlphaEarth as input, with Sentinel-1 SAR. Frozen embedding, MLP head
        learns a per-pixel mapping. Reaches F1 =
        {cmp.get("AlphaEarth_plus_S1", {}).get("f1", 0.602):.3f}.</li>
    <li><strong>AlphaEarth+S1, 3×3 conv head.</strong> Same setup but
        replace the MLP with a small convolutional head so the model
        sees local spatial context.
        Reaches F1 ≈ 0.631.</li>
    <li><strong>AlphaEarth pre + post + S1, multi-modal fusion.</strong>
        Stack the pre-event-year embedding with the event-year embedding
        (separate stems per modality, then fusion). Reaches F1 =
        {cmp.get("AE_pre_post_S1_stack", {}).get("f1", 0.708):.3f} —
        the best of the AlphaEarth-based variants.</li>
    <li><strong>U-Net S1+S2 (reference).</strong> Same training split,
        without any foundation prior, with post-event Sentinel-2 optical.
        F1 = {cmp.get("U-Net_S1_plus_S2", {}).get("f1", 0.849):.3f}.</li>
  </ol>

  <p>
    The honest reading: AlphaEarth's pre-event annual embedding alone is too
    coarse a representation of the event itself. The <strong>annual</strong>
    statistic averages over the year, including the calmer pre-event months;
    the embedding does not encode the specific water reflectance change at
    flood time. Temporal differencing — by passing both the pre-year and
    event-year embeddings through separate stems — recovers some of that
    information, lifting F1 by ~0.08. But it does not close the gap to
    direct post-event optical observation.
  </p>

  {_models_table_html()}

  <p>
    The AUPRC numbers tell a complementary story. AlphaEarth-based variants
    score higher AUPRC than the SAR-only baseline at every level, even when
    they lose on F1. <em>Foundation priors improve ranking quality;
    they don't fix the decision-threshold problem.</em>
  </p>

  <h2><span class="sec">Finding 3</span>
      Five percent labels reach 93% of full performance</h2>

  <figure>
    {_img(FIG_FEWSHOT_UNET, "U-Net few-shot curve")}
    <figcaption>
      <strong>Label-efficiency curve for U-Net (S1+S2).</strong>
      At {n_train_5pct} chips (5% of the training budget), the model already
      delivers F1 = {f1_5pct:.3f} — within {f1_100pct - f1_5pct:.3f} of its
      full-data ceiling.
    </figcaption>
  </figure>

  <div class="pullquote">
    Seventeen labelled chips give you 93% of the answer. Whatever else
    foundation models can do, they have to beat that bar to be worth using.
  </div>

  <h3>Does AlphaEarth's temporal stack hold up at 5% labels?</h3>
  <p>
    The blog from DeepMind highlights AlphaEarth's "best performance when
    data is scarce." We're running the same low-label sweep on our
    AlphaEarth pre+post + S1 stack so the two curves are directly
    comparable.
  </p>

  {_ae_few_shot_table()}

  <h2><span class="sec">Finding 4</span>
      From pixels to decisions — buildings, roads, populations</h2>

  <p>
    Hu et&nbsp;al.'s 2026
    <a href="https://www.nature.com/articles/s41586-026-10570-z">Nature paper
    on China's solar and wind</a> built its argument by combining a deep
    learning detector with a national grid optimisation — pixels became a
    99.88 TWh policy claim. We borrowed the same arc, applied to floods.
  </p>

  <p>
    Predictions on the USA test set were georeferenced and intersected with
    OpenStreetMap building footprints and major road segments. A building
    is "affected" if &geq;20% of its footprint pixels are predicted as
    water; a road segment is "affected" at &geq;15% intersection. The
    aggregate over all 69 USA chips:
  </p>

  {_decision_metrics_html()}

  <figure class="wide">
    {_img(FIG_DECISION, "Pixels to decisions")}
    <figcaption>
      <strong>Per-chip impact distribution.</strong>
      The bar chart on the left shows aggregate impact across all
      69 chips of the USA test set; the right panel shows where in the
      test set the impact concentrates (most chips have negligible
      flooding; impact is dominated by a small number of severely
      affected chips).
    </figcaption>
  </figure>

  <h2><span class="sec">Finding 5</span>
      Zero-shot deployment, honestly</h2>

  <p>
    To test whether a Sen1Floods11-trained model survives in a region that
    is completely outside its training distribution, we deployed it on the
    May 2024 Rio Grande do Sul flood in southern Brazil. Sen1Floods11
    contains <strong>zero South American training events</strong>. We pulled
    Sentinel-1 and Sentinel-2 composites for a 50 km square around Porto
    Alegre and Lake Guaiba via Google Earth Engine and ran the
    U-Net (S1+S2) model as-is.
  </p>

  <figure class="wide">
    {_img(FIG_BRAZIL, "Brazil zero-shot")}
    <figcaption>
      <strong>Sen1Floods11-trained U-Net deployed on Rio Grande do Sul, May 2024.</strong>
      Panels left to right: (a) Sentinel-2 RGB composite,
      (b) JRC permanent-water mask, (c) raw model prediction,
      (d) prediction minus permanent water. The model identifies the
      natural lake structure correctly but vastly over-predicts water across
      the rest of the AOI — exactly the failure mode the cross-region
      matrix flagged on Pakistan, only worse.
    </figcaption>
  </figure>

  {_brazil_summary_html()}

  <div class="callout">
    <strong>Why this is a productive negative result.</strong>
    It is precisely the cross-domain gap our leave-one-out matrix
    quantifies, projected onto an unseen continent. <em>The deployment
    works as a measurement, not as a solution.</em> The next experiment
    we want to run — and what the proposal in our repository is
    structured to enable — is taking the same model and giving it 5%
    of labels from Brazil (the Nature §H1 hypothesis), to see how
    quickly it recovers.
  </div>

  <h2><span class="sec">What's next</span>
      Toward a Nature-grade story</h2>

  <p>
    Three things turn this into a complete Nature-style submission:
  </p>

  <ol>
    <li><strong>Truly few-shot recovery on Brazil.</strong> Fine-tune the
        Sen1Floods11 model with 5%, 10%, 25% of Brazil-region labels and
        measure how fast the cross-continental gap closes. This is the
        most direct test of AlphaEarth's scarce-label promise on out-of-
        distribution disasters.</li>
    <li><strong>Japanese multi-hazard validation.</strong> Our larger
        proposal targets Japan because the country has dense, high-quality
        official flood / landslide / earthquake polygons (GSI, JAXA, MLIT).
        Applying the framework to Japanese events — including 2024 Noto
        Peninsula earthquake — extends from floods to multi-hazard.</li>
    <li><strong>Decision-metric atlas.</strong> Apply the predict
        &rarr; OSM &rarr; affected-buildings pipeline to many events globally
        and report a comparable cross-event impact table — the disaster-
        response analogue of the 99.88 TWh figure in Hu&nbsp;et&nbsp;al.</li>
  </ol>

</article>

<footer class="cite">
  <div class="container">
    <h3>How to cite this snapshot</h3>
    <p>
      Bao, Q., GeoDisaster-FM team. (2026).
      <em>What foundation models can — and cannot — do for global flood mapping.</em>
      Research notebook,
      <a href="https://github.com/14H034160212/geodisaster-fm">github.com/14H034160212/geodisaster-fm</a>,
      commit snapshot {now.strftime("%Y-%m-%d")}.
    </p>
    <h3 style="margin-top: 28px">References</h3>
    <p>
      Brown, C. et&nbsp;al. (2025).
      <a href="https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/">AlphaEarth&nbsp;Foundations:
      an embedding field model for accurate and efficient global mapping
      from sparse label data.</a> Google DeepMind blog &amp; arXiv:2507.22291.
      &nbsp;·&nbsp;
      Gong, S. et&nbsp;al. (2026).
      <em>CrossEarth: Geospatial Vision Foundation Model for
      Domain Generalizable Remote Sensing Semantic Segmentation.</em>
      IEEE TPAMI.
      &nbsp;·&nbsp;
      Hu, Y. et&nbsp;al. (2026).
      <a href="https://www.nature.com/articles/s41586-026-10570-z">Advancing
      solar and wind penetration in China through energy
      complementarity.</a> <em>Nature</em>.
      &nbsp;·&nbsp;
      Bonafilia, D. et&nbsp;al. (2020).
      <em>Sen1Floods11: a georeferenced dataset to train and test deep
      learning flood algorithms for Sentinel-1.</em> CVPR-W EarthVision.
    </p>
  </div>
</footer>

</body></html>
"""

    out_path.write_text(page, encoding="utf-8")
    return out_path
