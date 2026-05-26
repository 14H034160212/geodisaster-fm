"""Fig 0 — GeoDisaster-FM Dispatcher architecture.

Hybrid design: Nature Methods rigour (rectangles, sans-serif, panel labels)
but DeepMind-blog visual punch (colour-coded layers, large fonts, fewer
text elements). The goal is "what does this do?" answered in 3 seconds.

Layout: 3 large stacked layer cards, each colour-coded by status:
    Layer 3 (Reinforcement-learning policy)   — dashed grey, planned
    Layer 2 (Neuro-symbolic reasoner)          — solid navy, implemented (focus)
    Layer 1 (Perception backbone)              — solid teal, validated
Inputs feed in from the left as a compact stack, outputs flow out
to the right. A wide ribbon at the bottom carries the headline
"1–3 days → 30 minutes" claim.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Nimbus Sans L", "DejaVu Sans",
]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False


# Palette — three layer colours, restrained
NAVY        = "#1a3d6e"
NAVY_FILL   = "#dde5f0"
TEAL        = "#1f7a82"
TEAL_FILL   = "#d6ecee"
GREY_EDGE   = "#6c7480"
GREY_FILL   = "#f3f4f6"
BG_TINT     = "#fafbfc"        # very soft page background
EDGE        = "#1a202c"
BODY_GREY   = "#525861"
ACCENT_GOLD = "#a86a1f"


def _rect(ax, x, y, w, h, *, fc, ec, lw=1.0, dashed=False, zorder=2):
    rec = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                    linewidth=lw, zorder=zorder)
    if dashed:
        rec.set_linestyle((0, (5, 3)))
    ax.add_patch(rec)
    return rec


def _arrow(ax, x0, y0, x1, y1, *, lw=1.4, color=None):
    color = color or EDGE
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, zorder=4,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def render(out_path: str | Path = "outputs/figures/fig0_architecture.png") -> Path:
    fig = plt.figure(figsize=(14, 9), facecolor=BG_TINT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_facecolor(BG_TINT)
    ax.axis("off")

    # =====================================================================
    # TITLE
    # =====================================================================
    ax.text(7, 8.50,
            "GeoDisaster-FM Dispatcher",
            ha="center", fontsize=22, fontweight="bold", color=EDGE)
    ax.text(7, 8.10,
            "An AI emergency dispatcher that turns satellite imagery into "
            "actionable answers in minutes, not days",
            ha="center", fontsize=12, color=BODY_GREY)

    # =====================================================================
    # INPUTS (left) — compact icon-style stack
    # =====================================================================
    inp_x = 0.4
    inp_w = 2.4
    inp_y = 1.7
    inp_h = 5.5
    _rect(ax, inp_x, inp_y, inp_w, inp_h, fc="white", ec=GREY_EDGE, lw=0.9)
    ax.text(inp_x + inp_w / 2, inp_y + inp_h - 0.32,
            "INPUTS", ha="center", fontsize=10, fontweight="bold",
            color=BODY_GREY)

    inputs = [
        ("Sentinel-1",  "SAR"),
        ("Sentinel-2",  "Optical"),
        ("AlphaEarth", "Foundation prior"),
        ("OpenStreetMap", "Roads · Buildings · Facilities"),
        ("WorldPop",     "Population"),
        ("JRC GSW",      "Permanent water"),
    ]
    top = inp_y + inp_h - 0.95
    bot = inp_y + 0.30
    step = (top - bot) / (len(inputs) - 1)
    for i, (head, sub) in enumerate(inputs):
        y = top - i * step
        ax.text(inp_x + inp_w / 2, y, head,
                ha="center", fontsize=11.5, fontweight="bold", color=EDGE)
        ax.text(inp_x + inp_w / 2, y - 0.27, sub,
                ha="center", fontsize=9, color=BODY_GREY)

    # =====================================================================
    # THREE LAYER STACK (centre) — bigger boxes, fewer words
    # =====================================================================
    L_x = 3.6
    L_w = 7.0
    layer_h = 1.55
    gap = 0.40
    L1_y = 1.70
    L2_y = L1_y + layer_h + gap
    L3_y = L2_y + layer_h + gap

    # --- Layer 1 (bottom): perception, validated ---
    _rect(ax, L_x, L1_y, L_w, layer_h, fc=TEAL_FILL, ec=TEAL, lw=1.6)
    ax.text(L_x + 0.25, L1_y + layer_h - 0.40,
            "Layer 1   ·   Perception",
            fontsize=14, fontweight="bold", color=TEAL)
    ax.text(L_x + 0.25, L1_y + layer_h - 0.80,
            "Frozen geospatial backbone   →   pixel-level disaster footprint",
            fontsize=11, color=EDGE)
    # status pill
    _rect(ax, L_x + L_w - 1.55, L1_y + layer_h - 0.55, 1.30, 0.36,
          fc="white", ec=TEAL, lw=1.0)
    ax.text(L_x + L_w - 0.90, L1_y + layer_h - 0.37,
            "VALIDATED", ha="center", fontsize=9, fontweight="bold",
            color=TEAL)
    # key metrics inline
    ax.text(L_x + 0.25, L1_y + 0.45,
            "F1 = 0.849  on USA hold-out   ·   "
            "F1 = 0.828  averaged across 10 leave-one-region-out runs",
            fontsize=10.5, fontweight="bold", color=EDGE)
    ax.text(L_x + 0.25, L1_y + 0.18,
            "U-Net (Sentinel-1 + Sentinel-2)  ·  AlphaEarth pre + post + Sentinel-1",
            fontsize=9, color=BODY_GREY, style="italic")

    # --- Layer 2 (middle): reasoner, implemented (focal) ---
    _rect(ax, L_x, L2_y, L_w, layer_h, fc=NAVY_FILL, ec=NAVY, lw=2.0)
    ax.text(L_x + 0.25, L2_y + layer_h - 0.40,
            "Layer 2   ·   Neuro-symbolic reasoner",
            fontsize=14, fontweight="bold", color=NAVY)
    ax.text(L_x + 0.25, L2_y + layer_h - 0.80,
            "Graph algorithms + LLM planner   →   "
            "answers to 10 emergency questions",
            fontsize=11, color=EDGE)
    _rect(ax, L_x + L_w - 1.75, L2_y + layer_h - 0.55, 1.50, 0.36,
          fc="white", ec=NAVY, lw=1.0)
    ax.text(L_x + L_w - 1.00, L2_y + layer_h - 0.37,
            "IMPLEMENTED", ha="center", fontsize=9, fontweight="bold",
            color=NAVY)
    # Example questions in a single horizontal line for visual rhythm
    ax.text(L_x + 0.25, L2_y + 0.45,
            "Which hospitals are flooded?  ·  "
            "Which villages lost road access?  ·  "
            "Top-5 roads to clear?",
            fontsize=10.5, color=EDGE)
    ax.text(L_x + 0.25, L2_y + 0.18,
            "Live demo on USA chip: 58 / 2,053 buildings, 10.1 / 232 km roads, "
            "3 isolated communities  (Fig. 1d)",
            fontsize=9, color=BODY_GREY, style="italic")

    # --- Layer 3 (top): RL policy, planned ---
    _rect(ax, L_x, L3_y, L_w, layer_h, fc=GREY_FILL, ec=GREY_EDGE, lw=1.4,
          dashed=True)
    ax.text(L_x + 0.25, L3_y + layer_h - 0.40,
            "Layer 3   ·   Reinforcement-learning policy",
            fontsize=14, fontweight="bold", color=GREY_EDGE)
    ax.text(L_x + 0.25, L3_y + layer_h - 0.80,
            "Meta-RL across a curated atlas of ≥ 30 historical disasters",
            fontsize=11, color=EDGE)
    _rect(ax, L_x + L_w - 1.55, L3_y + layer_h - 0.55, 1.30, 0.36,
          fc="white", ec=GREY_EDGE, lw=1.0)
    ax.text(L_x + L_w - 0.90, L3_y + layer_h - 0.37,
            "PLANNED", ha="center", fontsize=9, fontweight="bold",
            color=GREY_EDGE)
    ax.text(L_x + 0.25, L3_y + 0.45,
            "Schedule perception · select labels · issue alerts · dispatch responders",
            fontsize=10.5, color=EDGE)
    ax.text(L_x + 0.25, L3_y + 0.18,
            "Reward = labels saved  +  response-time reduction  +  lives saved",
            fontsize=9, color=BODY_GREY, style="italic")

    # =====================================================================
    # OUTPUTS (right)
    # =====================================================================
    out_x = 11.2
    out_w = 2.4
    out_y = inp_y
    out_h = inp_h
    _rect(ax, out_x, out_y, out_w, out_h, fc="white", ec=GREY_EDGE, lw=0.9)
    ax.text(out_x + out_w / 2, out_y + out_h - 0.32,
            "OUTPUTS", ha="center", fontsize=10, fontweight="bold",
            color=BODY_GREY)

    outputs = [
        ("Briefing",       "one-page summary"),
        ("Dispatch report", "structured JSON"),
        ("Impact map",     "georeferenced GeoTIFF"),
        ("Action plan",    "ranked decisions"),
        ("Atlas entry",    "for self-improvement"),
        ("Time-to-answer", "minutes, not days"),
    ]
    top_o = out_y + out_h - 0.95
    bot_o = out_y + 0.30
    step_o = (top_o - bot_o) / (len(outputs) - 1)
    for i, (head, sub) in enumerate(outputs):
        y = top_o - i * step_o
        ax.text(out_x + out_w / 2, y, head,
                ha="center", fontsize=11.5, fontweight="bold", color=EDGE)
        ax.text(out_x + out_w / 2, y - 0.27, sub,
                ha="center", fontsize=9, color=BODY_GREY)

    # =====================================================================
    # ARROWS — only essential flows, big and obvious
    # =====================================================================
    # Inputs → Layer 1
    _arrow(ax, inp_x + inp_w, L1_y + layer_h / 2, L_x, L1_y + layer_h / 2,
           lw=1.6, color=GREY_EDGE)
    # Inputs → Layer 2 (graph data: OSM, WorldPop, JRC)
    _arrow(ax, inp_x + inp_w, L2_y + layer_h / 2, L_x, L2_y + layer_h / 2,
           lw=1.6, color=GREY_EDGE)
    # Layer 1 → Layer 2 (internal)
    _arrow(ax, L_x + L_w / 2, L1_y + layer_h, L_x + L_w / 2, L2_y,
           lw=1.8, color=EDGE)
    # Layer 2 → Layer 3 (internal)
    _arrow(ax, L_x + L_w / 2, L2_y + layer_h, L_x + L_w / 2, L3_y,
           lw=1.8, color=EDGE)
    # Layer 2 → Outputs (briefings / reports / maps)
    _arrow(ax, L_x + L_w, L2_y + layer_h / 2, out_x, L2_y + layer_h / 2,
           lw=1.6, color=GREY_EDGE)
    # Layer 3 → Outputs (action plan)
    _arrow(ax, L_x + L_w, L3_y + layer_h / 2, out_x, L3_y + layer_h / 2,
           lw=1.6, color=GREY_EDGE)

    # =====================================================================
    # FOOTER RIBBON — the headline claim, big and clear
    # =====================================================================
    foot_y = 0.45
    foot_h = 0.75
    _rect(ax, 0.4, foot_y, 13.2, foot_h, fc=NAVY, ec=NAVY, lw=0)
    ax.text(1.2, foot_y + foot_h / 2 + 0.05,
            "End-to-end metric:",
            fontsize=10.5, color="#cad7eb", va="center")
    ax.text(1.2, foot_y + foot_h / 2 - 0.18,
            "time to answer the 10-question UN-OCHA questionnaire",
            fontsize=9, color="#9fb3d8", va="center", style="italic")

    ax.text(7.0, foot_y + foot_h / 2 + 0.05,
            "Manual workflow   1–3 days",
            fontsize=11.5, color="white", ha="center", va="center",
            fontweight="bold")
    ax.text(7.0, foot_y + foot_h / 2 - 0.20,
            "Dispatcher target",
            fontsize=9, color="#9fb3d8", ha="center", va="center")

    ax.annotate("", xy=(9.6, foot_y + foot_h / 2),
                xytext=(8.2, foot_y + foot_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="white",
                                lw=2, mutation_scale=18))

    ax.text(10.6, foot_y + foot_h / 2 + 0.05,
            "30 minutes",
            fontsize=13, color="white", ha="left", va="center",
            fontweight="bold")
    ax.text(13.05, foot_y + foot_h / 2,
            "≥ 100×", fontsize=18, color="white", ha="right", va="center",
            fontweight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG_TINT, pad_inches=0.25)
    plt.close()
    return out_path


if __name__ == "__main__":
    print(f"Saved: {render()}")
