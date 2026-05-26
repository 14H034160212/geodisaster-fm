"""Static HTML report generator.

Scans ``outputs/`` and ``data/`` for known artefacts (sweep CSVs, comparison
JSON, figures, reproducibility manifest, catalog) and renders a single
self-contained ``outputs/site/index.html`` with embedded base64 PNG figures.

Usage:
    geodisaster build-report
        # → writes outputs/site/index.html

Re-run after every experiment; the page reflects whatever's currently on disk.
Single file means it's trivial to scp, rsync, push to gh-pages, or just open
locally — no JavaScript, no external assets, no build system.
"""
from __future__ import annotations

import base64
import datetime as dt
import html as _html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Artefact discovery
# --------------------------------------------------------------------------- #
KNOWN_SWEEPS = {
    "U-Net SAR+Optical (S1+S2)":  "outputs/few_shot_unet_s1s2/few_shot_results.csv",
    "AlphaEarth + S1 (MLP head)": "outputs/few_shot_ae_s1/few_shot_results.csv",
}
KNOWN_FIGURES = [
    ("Fig 3 — Four-way label-fraction comparison", "outputs/figures/fig3_four_way_comparison.png"),
    ("Fig 3b — Few-shot curve (U-Net S1+S2 alone)", "outputs/figures/fig3_sen1floods11_few_shot.png"),
    ("Fig 4 — Leave-one-region-out (10 holdouts) — CrossEarth-style",
        "outputs/figures/fig4_leave_one_region_out.png"),
    ("Fig 5 — Pixels → decisions on USA test set (Hu Nature-style)",
        "outputs/figures/fig5_usa_decision.png"),
]
LEAVE_ONE_OUT_JSON = "outputs/leave_one_region_out/results.json"
DECISION_SUMMARY = "outputs/usa_decision/decision_summary.json"
KNOWN_TABLES = {
    "Sen1Floods11 100%-label comparison": "outputs/sen1floods11_comparison.json",
    "Few-shot full table":                 "outputs/four_way_results_table.json",
    "Single-run results table":            "outputs/sen1floods11_results_table.json",
}
REPRO_MANIFEST = "outputs/reproducibility.json"
CATALOG_YAML   = "data/catalog/japan_events.yaml"
SEN1_CATALOG   = "data/catalog/sen1floods11_events.yaml"


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _df_to_html(df: pd.DataFrame, classes: str = "results") -> str:
    return df.to_html(index=False, classes=classes, border=0, escape=True,
                      float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))


def _findings_cards(manifest: dict | None) -> str:
    headlines = (manifest or {}).get("extra", {}).get("headline_findings", {})
    if not headlines:
        return ""
    cards = []
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (k, v) in enumerate(headlines.items()):
        cards.append(
            f'<div class="card" style="border-left-color:{palette[i % len(palette)]}">'
            f'<div class="card-title">{_html.escape(k.replace("_", " "))}</div>'
            f'<div class="card-body">{_html.escape(str(v))}</div></div>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _experiment_status(now: dt.datetime) -> str:
    """A small table of which experiments produced results."""
    rows = []
    for label, csv in KNOWN_SWEEPS.items():
        p = Path(csv)
        if p.exists():
            df = pd.read_csv(p)
            rows.append((label, f"{len(df)} fractions", p.stat().st_mtime))
        else:
            rows.append((label, "not run", 0))
    for label, j in KNOWN_TABLES.items():
        p = Path(j)
        if p.exists():
            rows.append((label, "ok", p.stat().st_mtime))
    body = []
    for name, status, mtime in rows:
        ts = (dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
              if mtime else "—")
        body.append(f"<tr><td>{_html.escape(name)}</td>"
                    f"<td>{_html.escape(status)}</td><td>{ts}</td></tr>")
    return ("<table class='results'><thead><tr><th>Artefact</th><th>Status</th>"
            "<th>Updated</th></tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def _sweep_tables() -> str:
    parts = []
    for label, csv in KNOWN_SWEEPS.items():
        p = Path(csv)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        keep = ["label_fraction", "n_train", "test/f1", "test/iou",
                "test/precision", "test/recall"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.rename(columns={
            "label_fraction": "label_frac",
            "test/f1": "F1", "test/iou": "IoU",
            "test/precision": "Precision", "test/recall": "Recall",
        })
        parts.append(f"<h3>{_html.escape(label)}</h3>{_df_to_html(df)}")
    return "\n".join(parts)


def _leave_one_out_table() -> str:
    p = Path(LEAVE_ONE_OUT_JSON)
    if not p.exists():
        return ""
    rows = json.loads(p.read_text())
    if not rows:
        return ""
    df = pd.DataFrame([{
        "Test region": r["test_region"],
        "F1": round(r["f1"], 4),
        "IoU": round(r["iou"], 4),
        "Precision": round(r["precision"], 4),
        "Recall": round(r["recall"], 4),
        "AUPRC": round(r["auprc"], 4),
        "train_time_s": r["train_time_s"],
    } for r in rows]).sort_values("F1", ascending=False)
    f1_mean = df["F1"].mean()
    iou_mean = df["IoU"].mean()
    auprc_mean = df["AUPRC"].mean()
    f1_spread = df["F1"].max() - df["F1"].min()
    note = (f"<p style='font-size:13px;color:#6b7280;margin-top:0'>"
            f"<strong>{len(rows)} held-out regions</strong> · "
            f"avg F1 = {f1_mean:.3f} · avg IoU = {iou_mean:.3f} · "
            f"avg AUPRC = {auprc_mean:.3f} · "
            f"F1 spread = {f1_spread:.3f} (hardest: "
            f"{df.iloc[-1]['Test region']} F1 = {df.iloc[-1]['F1']:.3f})"
            f"</p>")
    return ("<h3>Leave-one-region-out — U-Net (S1+S2) on 10 unseen regions</h3>"
            + note + _df_to_html(df))


def _decision_summary() -> str:
    p = Path(DECISION_SUMMARY)
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    t = d.get("totals", {})
    n = d.get("n_chips_evaluated", 0)
    pct = lambda a, tot: 100 * a / max(tot, 1)
    rows = [
        {"Quantity": "Buildings (OSM polygons in chips)",
         "Total":     t.get("buildings_total"),
         "Affected":  t.get("buildings_affected"),
         "Affected %": f"{pct(t.get('buildings_affected', 0), t.get('buildings_total', 1)):.2f}%"},
        {"Quantity": "Major roads (km)",
         "Total":     t.get("road_km_total"),
         "Affected":  t.get("road_km_affected"),
         "Affected %": f"{pct(t.get('road_km_affected', 0), t.get('road_km_total', 1)):.2f}%"},
    ]
    df = pd.DataFrame(rows)
    return (f"<h3>Decision metrics — USA test set, {n} chips "
            f"(Hu et al. 2026 Nature pipeline style)</h3>"
            + _df_to_html(df))


def _comparison_table() -> str:
    p = Path("outputs/sen1floods11_comparison.json")
    if not p.exists():
        return ""
    data = json.loads(p.read_text())
    rows = []
    for name, m in data.items():
        rows.append({
            "model": name.replace("_", " "),
            "F1":     m.get("f1", 0),
            "IoU":    m.get("iou", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
            "AUPRC":  m.get("auprc", 0),
            "ECE":    m.get("ece", 0),
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    return f"<h3>Models @ 100% labels — test on USA (held-out, 69 chips)</h3>{_df_to_html(df)}"


def _figures_html() -> str:
    blocks = []
    for title, path in KNOWN_FIGURES:
        p = Path(path)
        if not p.exists():
            continue
        b64 = _img_b64(p)
        blocks.append(
            f'<figure><img src="data:image/png;base64,{b64}" alt="{_html.escape(title)}"/>'
            f'<figcaption>{_html.escape(title)}</figcaption></figure>'
        )
    return "\n".join(blocks) if blocks else "<p><em>No figures yet — run experiments first.</em></p>"


def _catalog_summary() -> str:
    parts = []
    for label, p in (("Japan multi-hazard", CATALOG_YAML),
                     ("Sen1Floods11 (auto-generated)", SEN1_CATALOG)):
        path = Path(p)
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text())
        events = raw.get("events", [])
        # Group by hazard
        groups: dict[str, list[str]] = {}
        for e in events:
            groups.setdefault(e.get("hazard", "?"), []).append(e["event_id"])
        rows = [f"<tr><td>{_html.escape(h)}</td><td>{len(ids)}</td>"
                f"<td><span class='ids'>{', '.join(_html.escape(i) for i in ids)}</span></td></tr>"
                for h, ids in sorted(groups.items())]
        parts.append(
            f"<h3>{_html.escape(label)}  ·  {len(events)} events total</h3>"
            "<table class='results'><thead><tr><th>Hazard</th><th>n</th><th>Event IDs</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
    return "\n".join(parts)


def _repro_summary() -> str:
    p = Path(REPRO_MANIFEST)
    if not p.exists():
        return "<p><em>No reproducibility manifest yet.</em></p>"
    m = json.loads(p.read_text())
    parts = []
    parts.append(f"<p><strong>Generated:</strong> {_html.escape(m.get('timestamp', '—'))}</p>")
    parts.append(f"<p><strong>Python:</strong> {_html.escape(m.get('python', '—').splitlines()[0])}</p>")
    parts.append(f"<p><strong>Git revision:</strong> {_html.escape(str(m.get('git_revision')))}</p>")

    pkgs = m.get("package_versions", {})
    rows = "".join(
        f"<tr><td>{_html.escape(k)}</td><td>{_html.escape(v)}</td></tr>"
        for k, v in pkgs.items()
    )
    parts.append(f"<details><summary>Package versions ({len(pkgs)})</summary>"
                 "<table class='results small'><tbody>" + rows + "</tbody></table></details>")

    arte = m.get("artefacts", {})
    art_rows = []
    for name, info in arte.items():
        sha = (info.get("sha256") or "")[:12]
        art_rows.append(
            f"<tr><td>{_html.escape(name)}</td>"
            f"<td><code>{_html.escape(info.get('path', ''))}</code></td>"
            f"<td><code>{sha}…</code></td></tr>"
        )
    parts.append(f"<details><summary>Artefacts ({len(arte)})</summary>"
                 "<table class='results small'><thead><tr><th>Name</th><th>Path</th><th>SHA-256</th></tr></thead>"
                 f"<tbody>{''.join(art_rows)}</tbody></table></details>")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Page template
# --------------------------------------------------------------------------- #
_CSS = """
:root { --fg:#1a202c; --muted:#6b7280; --bg:#fbfbfb; --card:#ffffff;
        --accent:#1f77b4; --border:#e5e7eb; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif; color:var(--fg); background:var(--bg); }
.container { max-width:1100px; margin:0 auto; padding:24px; }
header { border-bottom:1px solid var(--border); padding-bottom:18px; margin-bottom:30px; }
h1 { font-size:24px; margin:0 0 6px 0; }
header p { color:var(--muted); margin:4px 0; }
h2 { font-size:18px; margin-top:38px; padding-bottom:6px;
     border-bottom:2px solid var(--accent); display:inline-block; }
h3 { font-size:14px; color:var(--muted); margin-top:22px; }
table.results { border-collapse:collapse; width:100%; margin:8px 0 18px; font-size:13px; background:var(--card); }
table.results th, table.results td { padding:7px 11px; border-bottom:1px solid var(--border); text-align:left; }
table.results thead th { background:#f3f4f6; font-weight:600; }
table.results.small td, table.results.small th { font-size:12px; padding:5px 9px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
         gap:14px; margin:14px 0 24px; }
.card { background:var(--card); border:1px solid var(--border); border-left:4px solid #ccc;
        padding:14px 16px; border-radius:6px; }
.card-title { font-weight:600; font-size:12px; text-transform:uppercase;
              color:var(--muted); letter-spacing:0.4px; margin-bottom:8px; }
.card-body { font-size:13.5px; }
figure { margin:24px 0; }
figure img { max-width:100%; height:auto; border:1px solid var(--border);
             border-radius:6px; background:white; }
figcaption { font-size:12px; color:var(--muted); margin-top:6px; }
code { font-family:Menlo, Consolas, monospace; background:#f3f4f6; padding:0 4px;
       border-radius:3px; font-size:12px; }
details { margin:10px 0; }
details summary { cursor:pointer; font-weight:500; color:var(--accent); }
.ids { font-family:Menlo,Consolas,monospace; font-size:11px; color:var(--muted); }
footer { margin-top:60px; padding-top:20px; border-top:1px solid var(--border);
         color:var(--muted); font-size:12px; }
"""


def build_report(out_dir: str | Path = "outputs/site",
                 project_root: str | Path = ".",
                 filename: str = "dashboard.html") -> Path:
    project_root = Path(project_root).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.utcnow()

    manifest_path = Path(REPRO_MANIFEST)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GeoDisaster-FM — experiment dashboard</title>
<style>{_CSS}</style>
</head>
<body><div class="container">
<header>
  <h1>GeoDisaster-FM — experiment dashboard</h1>
  <p style="font-size:14px"><a href="index.html">← Back to research narrative</a></p>
  <p>Live experiment dashboard · last build {now.strftime("%Y-%m-%d %H:%M UTC")}</p>
  <p>Project: <code>{_html.escape(str(project_root))}</code></p>
</header>

<h2>Headline findings</h2>
{_findings_cards(manifest)}

<h2>Experiment status</h2>
{_experiment_status(now)}

<h2>Sen1Floods11 cross-region — model comparison</h2>
{_comparison_table()}
{_sweep_tables()}

<h2>Cross-domain robustness (CrossEarth-style)</h2>
{_leave_one_out_table()}

<h2>From pixels to decisions (Hu et al. Nature-style)</h2>
{_decision_summary()}

<h2>Figures</h2>
{_figures_html()}

<h2>Event catalog</h2>
{_catalog_summary()}

<h2>Reproducibility</h2>
{_repro_summary()}

<footer>
Single-file static report — re-run <code>geodisaster build-report</code> after each
experiment to refresh. Open this file in any browser, or serve via
<code>python -m http.server</code>, or push to GitHub Pages.
</footer>
</div></body></html>
"""

    out_path = out_dir / filename
    out_path.write_text(page, encoding="utf-8")
    return out_path
