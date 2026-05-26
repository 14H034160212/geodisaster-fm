"""Fig 0 — three-layer GeoDisaster-FM Dispatcher architecture.

Nature Methods–style schematic. Design choices:
  - Rectangular panels with thin (0.7 pt) black borders, no rounded corners.
  - Sans-serif typography (Helvetica / Arial / DejaVu Sans fallback).
  - Restricted palette: navy accent (#1a3d6e), pale blue fill (#dee5f0),
    light grey panel background (#f5f6f8), grey body text (#525861).
  - Panels labelled a, b, c lowercase bold — Nature convention.
  - Data flow strictly top-to-bottom with single-style arrow heads.
  - Inputs and outputs as compact bulleted lists, no decorative glyphs.
  - Status communicated by a small italicised stamp inside each panel,
    not by colour or icon.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---- Typography ----
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Nimbus Sans L", "DejaVu Sans",
]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False


# ---- Palette ----
NAVY        = "#1a3d6e"
PALE_BLUE   = "#dee5f0"
PANEL_GREY  = "#f5f6f8"
EDGE        = "#222222"
BODY_GREY   = "#525861"
SOFT_BLUE   = "#3e6db8"


def _rect(ax, x, y, w, h, *, fc=PANEL_GREY, ec=EDGE, lw=0.7, zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                            linewidth=lw, zorder=zorder))


def _arrow(ax, x0, y0, x1, y1, *, lw=0.9, label=None, label_offset=(0.05, 0)):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=EDGE, zorder=3,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=7.5, color=BODY_GREY,
                ha="left", va="center", style="italic")


def render(out_path: str | Path = "outputs/figures/fig0_architecture.png") -> Path:
    fig = plt.figure(figsize=(13, 8.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # =====================================================================
    # TITLE BLOCK
    # =====================================================================
    ax.text(6.5, 8.15,
            "Three-layer architecture of the GeoDisaster-FM Dispatcher",
            ha="center", fontsize=12, fontweight="bold", color=EDGE)
    ax.text(6.5, 7.85,
            "End-to-end metric: time to answer the 10-question UN-OCHA "
            "emergency questionnaire from raw satellite imagery",
            ha="center", fontsize=9, color=BODY_GREY)

    # =====================================================================
    # COLUMN HEADERS
    # =====================================================================
    col_y = 7.4
    ax.text(1.4, col_y, "Inputs",      ha="center", fontsize=9,
            fontweight="bold", color=BODY_GREY)
    ax.text(6.5, col_y, "Three-layer agent", ha="center", fontsize=9,
            fontweight="bold", color=BODY_GREY)
    ax.text(11.6, col_y, "Outputs",    ha="center", fontsize=9,
            fontweight="bold", color=BODY_GREY)

    # =====================================================================
    # PANEL  a  —  INPUTS  (left column)
    # =====================================================================
    inp_x, inp_y, inp_w, inp_h = 0.3, 1.3, 2.2, 5.8
    _rect(ax, inp_x, inp_y, inp_w, inp_h, fc="white", ec=EDGE, lw=0.7)
    ax.text(inp_x + 0.18, inp_y + inp_h - 0.32, "a", fontsize=12,
            fontweight="bold", color=EDGE)

    inputs = [
        ("Sentinel-1 GRD", "VV + VH, dB scale, 10 m"),
        ("Sentinel-2 L1C", "13 spectral bands, 10–60 m"),
        ("AlphaEarth", "64-d annual embedding, 10 m"),
        ("OpenStreetMap", "buildings, roads, facilities"),
        ("WorldPop", "100 m population density"),
        ("JRC Global Surface Water", "permanent-water occurrence"),
    ]
    # Layout 6 entries evenly inside the box (between top label and box bottom)
    top = inp_y + inp_h - 0.85
    bot = inp_y + 0.4
    step = (top - bot) / (len(inputs) - 1)
    for i, (head, sub) in enumerate(inputs):
        y = top - i * step
        ax.text(inp_x + 0.18, y, head, fontsize=8.6, fontweight="bold",
                color=EDGE)
        ax.text(inp_x + 0.18, y - 0.26, sub, fontsize=7.6, color=BODY_GREY)

    # =====================================================================
    # PANEL  b  —  THREE-LAYER AGENT  (centre column)
    # =====================================================================
    L_x  = 3.4
    L_w  = 6.2
    L1_y, L1_h = 1.3,  1.65
    L2_y, L2_h = 3.20, 1.90
    L3_y, L3_h = 5.30, 1.75

    # --- Layer 3 (top) — planned ---
    _rect(ax, L_x, L3_y, L_w, L3_h, fc=PANEL_GREY, ec=EDGE, lw=0.7)
    ax.text(L_x + 0.18, L3_y + L3_h - 0.32, "b", fontsize=12,
            fontweight="bold", color=EDGE)
    ax.text(L_x + 0.35, L3_y + L3_h - 0.32, "Layer 3 — Reinforcement-learning policy",
            fontsize=10, fontweight="bold", color=EDGE)
    ax.text(L_x + L_w - 0.15, L3_y + L3_h - 0.32, "planned",
            fontsize=7.6, color=BODY_GREY, ha="right",
            style="italic")
    ax.text(L_x + 0.35, L3_y + L3_h - 0.62,
            "Meta-RL trained across an atlas of ≥30 historical disasters; PPO baseline.",
            fontsize=8.2, color=BODY_GREY)

    # Action chips — evenly across panel
    actions = ["task imagery", "ask label", "issue alert", "dispatch responder"]
    chip_w = 1.30
    gap = (L_w - 0.7 - 4 * chip_w) / 3
    for i, label in enumerate(actions):
        x = L_x + 0.35 + i * (chip_w + gap)
        _rect(ax, x, L3_y + 0.50, chip_w, 0.32, fc="white", ec=EDGE, lw=0.5)
        ax.text(x + chip_w / 2, L3_y + 0.66, label,
                ha="center", fontsize=7.6, color=EDGE)
    ax.text(L_x + 0.35, L3_y + 0.20,
            "Reward = labels not wasted  +  response time saved  +  lives saved",
            fontsize=7.7, color=BODY_GREY, style="italic")

    # --- Layer 2 (middle) — implemented ---
    _rect(ax, L_x, L2_y, L_w, L2_h, fc=PALE_BLUE, ec=NAVY, lw=0.9)
    ax.text(L_x + 0.18, L2_y + L2_h - 0.32, "c", fontsize=12,
            fontweight="bold", color=EDGE)
    ax.text(L_x + 0.35, L2_y + L2_h - 0.32,
            "Layer 2 — Neuro-symbolic reasoner",
            fontsize=10, fontweight="bold", color=NAVY)
    ax.text(L_x + L_w - 0.15, L2_y + L2_h - 0.32, "implemented",
            fontsize=7.6, color=NAVY, ha="right", style="italic",
            fontweight="bold")
    ax.text(L_x + 0.35, L2_y + L2_h - 0.62,
            "NetworkX graph reasoning over OSM  +  LLM planner over Datalog templates",
            fontsize=8.2, color=EDGE)

    # Ten-question chips — 3 columns x 2 rows
    qs = [
        "Q1 hospitals in footprint",
        "Q3 buildings affected",
        "Q4 roads blocked (km)",
        "Q5 isolated populated areas",
        "Q7 top-5 roads to clear",
        "Q9 population disconnected",
    ]
    cw, ch = 1.78, 0.32
    col_gap = (L_w - 0.7 - 3 * cw) / 2
    for i, q in enumerate(qs):
        row, col = i // 3, i % 3
        x = L_x + 0.35 + col * (cw + col_gap)
        y = L2_y + 0.72 - row * (ch + 0.10)
        _rect(ax, x, y, cw, ch, fc="white", ec=NAVY, lw=0.5)
        ax.text(x + cw / 2, y + ch / 2, q, ha="center", va="center",
                fontsize=7.2, color=EDGE)

    # --- Layer 1 (bottom) — validated ---
    _rect(ax, L_x, L1_y, L_w, L1_h, fc=PANEL_GREY, ec=EDGE, lw=0.7)
    ax.text(L_x + 0.18, L1_y + L1_h - 0.32, "d", fontsize=12,
            fontweight="bold", color=EDGE)
    ax.text(L_x + 0.35, L1_y + L1_h - 0.32,
            "Layer 1 — Perception backbone",
            fontsize=10, fontweight="bold", color=EDGE)
    ax.text(L_x + L_w - 0.15, L1_y + L1_h - 0.32, "validated",
            fontsize=7.6, color=BODY_GREY, ha="right", style="italic")
    ax.text(L_x + 0.35, L1_y + L1_h - 0.62,
            "Frozen geospatial backbone: U-Net + Sentinel-2, or AlphaEarth + Sentinel-1.",
            fontsize=8.2, color=BODY_GREY)
    ax.text(L_x + 0.35, L1_y + L1_h - 0.90,
            "Produces a pixel-level disaster footprint at 10 m resolution.",
            fontsize=8.2, color=BODY_GREY)

    # Metric row — evenly distributed across the panel
    metrics_y = L1_y + 0.18
    metrics = [
        ("F1 = 0.849", "U-Net (S1+S2) on USA hold-out"),
        ("F1 = 0.828", "leave-one-region-out average (n = 10)"),
        ("F1 = 0.789", "U-Net (S1+S2) at 5 % labels (17 chips)"),
    ]
    metric_gap = (L_w - 0.7) / 3
    for i, (val, sub) in enumerate(metrics):
        x = L_x + 0.35 + i * metric_gap + metric_gap / 2
        ax.text(x, metrics_y + 0.22, val, fontsize=8.6,
                fontweight="bold", color=NAVY, ha="center")
        ax.text(x, metrics_y, sub, fontsize=7.2, color=BODY_GREY, ha="center")

    # =====================================================================
    # PANEL  e  —  OUTPUTS (right column)
    # =====================================================================
    out_x, out_y, out_w, out_h = 10.5, 1.3, 2.2, 5.8
    _rect(ax, out_x, out_y, out_w, out_h, fc="white", ec=EDGE, lw=0.7)
    ax.text(out_x + 0.18, out_y + out_h - 0.32, "e", fontsize=12,
            fontweight="bold", color=EDGE)

    outputs = [
        ("Briefing", "one-page emergency summary"),
        ("Dispatch report", "structured JSON, all 10 answers"),
        ("Impact map", "georeferenced GeoTIFF + overlays"),
        ("Action plan", "ranked decisions (alerts, dispatch)"),
        ("Atlas entry", "consolidated for self-improvement"),
        ("Time-to-answer", "minutes vs baseline 1–3 days"),
    ]
    top_o = out_y + out_h - 0.85
    bot_o = out_y + 0.4
    step_o = (top_o - bot_o) / (len(outputs) - 1)
    for i, (head, sub) in enumerate(outputs):
        y = top_o - i * step_o
        ax.text(out_x + 0.18, y, head, fontsize=8.6, fontweight="bold",
                color=EDGE)
        ax.text(out_x + 0.18, y - 0.26, sub, fontsize=7.6, color=BODY_GREY)

    # =====================================================================
    # Data-flow arrows — label placed OUTSIDE the panels they connect
    # to avoid overlap with panel content.
    # =====================================================================
    mid_x = L_x + L_w / 2  # centre of the agent column

    # a → d  (Inputs → Layer 1, perception inputs)
    arr_y1 = L1_y + L1_h - 0.40
    _arrow(ax, inp_x + inp_w, arr_y1, L_x, arr_y1, lw=1.0)
    ax.text((inp_x + inp_w + L_x) / 2, arr_y1 + 0.18,
            "Sentinel-1/2, AlphaEarth",
            ha="center", fontsize=7.4, color=BODY_GREY, style="italic")

    # a → c  (Inputs → Layer 2, graph inputs)
    arr_y2 = L2_y + L2_h - 0.40
    _arrow(ax, inp_x + inp_w, arr_y2, L_x, arr_y2, lw=1.0)
    ax.text((inp_x + inp_w + L_x) / 2, arr_y2 + 0.18,
            "OSM, WorldPop, JRC",
            ha="center", fontsize=7.4, color=BODY_GREY, style="italic")

    # d → c  (Layer 1 → Layer 2, internal flow)
    _arrow(ax, mid_x, L1_y + L1_h, mid_x, L2_y, lw=1.0)
    ax.text(mid_x + 0.12, (L1_y + L1_h + L2_y) / 2,
            "pixel disaster footprint",
            fontsize=7.4, color=BODY_GREY, style="italic", va="center")

    # c → b  (Layer 2 → Layer 3, internal flow)
    _arrow(ax, mid_x, L2_y + L2_h, mid_x, L3_y, lw=1.0)
    ax.text(mid_x + 0.12, (L2_y + L2_h + L3_y) / 2,
            "graph state + answers",
            fontsize=7.4, color=BODY_GREY, style="italic", va="center")

    # c → e  (Layer 2 → Outputs, briefings / reports)
    arr_y3 = L2_y + L2_h - 0.40
    _arrow(ax, L_x + L_w, arr_y3, out_x, arr_y3, lw=1.0)
    ax.text((L_x + L_w + out_x) / 2, arr_y3 + 0.18,
            "briefing / report",
            ha="center", fontsize=7.4, color=BODY_GREY, style="italic")

    # b → e  (Layer 3 → Outputs, action plan)
    arr_y4 = L3_y + 0.40
    _arrow(ax, L_x + L_w, arr_y4, out_x, arr_y4, lw=1.0)
    ax.text((L_x + L_w + out_x) / 2, arr_y4 + 0.18,
            "action plan",
            ha="center", fontsize=7.4, color=BODY_GREY, style="italic")

    # =====================================================================
    # FOOTER BAR  — full-width
    # =====================================================================
    _rect(ax, 0.3, 0.35, 12.4, 0.75, fc="white", ec=EDGE, lw=0.7)
    ax.text(0.55, 0.95,
            "Baseline (manual expert workflow)",
            fontsize=8.5, color=BODY_GREY)
    ax.text(0.55, 0.55, "1–3 days", fontsize=13, fontweight="bold", color=EDGE)

    ax.text(5.5, 0.95,
            "Proposed (Layer 1 + 2 + 3)",
            fontsize=8.5, color=BODY_GREY)
    ax.text(5.5, 0.55, "30 minutes (target)", fontsize=13, fontweight="bold",
            color=NAVY)

    ax.text(12.55, 0.95, "Improvement",
            fontsize=8.5, color=BODY_GREY, ha="right")
    ax.text(12.55, 0.55, "≥ 100×", fontsize=13, fontweight="bold",
            color=NAVY, ha="right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.2)
    plt.close()
    return out_path


if __name__ == "__main__":
    print(f"Saved: {render()}")
