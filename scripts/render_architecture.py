"""Render the three-layer GeoDisaster-FM Dispatcher architecture figure.

A clean Nature-blog-style schematic showing how Layers 1-3 connect to
the upstream inputs (Sentinel + AlphaEarth) and downstream responder
deliverables (the 10 emergency questions).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def render(out_path: str | Path = "outputs/figures/fig0_architecture.png") -> Path:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Color palette (consistent with blog CSS)
    accent_blue = "#2453a8"
    layer_done  = "#cfe2f3"   # light blue
    layer_partial = "#e0e6f3" # paler
    layer_plan  = "#f0f0f3"   # light grey
    arrow_color = "#5a6577"
    text_color  = "#1a202c"
    muted       = "#5a6577"

    # Title
    ax.text(7, 8.5, "GeoDisaster-FM Dispatcher — three-layer agent for global disaster response",
            ha="center", fontsize=14, fontweight="bold", color=text_color)
    ax.text(7, 8.1, "From raw satellite imagery to actionable emergency answers in 30 minutes instead of 1–3 days",
            ha="center", fontsize=10, color=muted, style="italic")

    # ----- INPUTS (left) -----
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 1.5), 1.8, 5, boxstyle="round,pad=0.06",
        facecolor="#fdf3e6", edgecolor="#c08538", linewidth=1.5,
    ))
    ax.text(1.2, 6.2, "INPUTS", ha="center", fontsize=10,
            fontweight="bold", color="#7a5614")
    inputs = [
        ("Sentinel-1", "SAR VV+VH, dB"),
        ("Sentinel-2", "13-band optical"),
        ("AlphaEarth", "64-d annual emb."),
        ("OSM", "roads / buildings\nfacilities"),
        ("WorldPop", "100 m population"),
        ("JRC GSW", "permanent water"),
    ]
    for i, (name, desc) in enumerate(inputs):
        y = 5.6 - i * 0.65
        ax.text(0.45, y, "▸", color="#c08538", fontsize=10, fontweight="bold")
        ax.text(0.65, y, f"{name}", fontsize=8.5, fontweight="bold", color=text_color)
        ax.text(0.65, y - 0.18, desc, fontsize=7, color=muted)

    # ----- LAYER 1 (bottom-middle) ✓ done -----
    L1_y = 1.7
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.7, L1_y), 8.7, 1.2, boxstyle="round,pad=0.05",
        facecolor=layer_done, edgecolor=accent_blue, linewidth=2.0,
    ))
    ax.text(2.95, L1_y + 0.96, "Layer 1 · Perception", fontsize=11.5,
            fontweight="bold", color=accent_blue)
    ax.text(2.95, L1_y + 0.72,
            "Frozen geospatial backbone (U-Net + Sentinel-2, or AlphaEarth + S1) → pixel-level disaster footprint",
            fontsize=9, color=text_color)
    ax.text(2.95, L1_y + 0.42,
            "Validated: F1 = 0.849 on USA hold-out · avg F1 = 0.828 across 10 leave-one-region-out runs",
            fontsize=8.5, color=muted, style="italic")
    ax.text(11.0, L1_y + 0.6, "✓ DONE", ha="center", fontsize=10,
            fontweight="bold", color="#1c7f4f",
            bbox=dict(boxstyle="round,pad=0.3", fc="#dcecdc", ec="#1c7f4f"))

    # ----- LAYER 2 (middle) ◐ partial -----
    L2_y = 3.4
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.7, L2_y), 8.7, 1.7, boxstyle="round,pad=0.05",
        facecolor=layer_partial, edgecolor=accent_blue, linewidth=2.0,
    ))
    ax.text(2.95, L2_y + 1.46, "Layer 2 · Neuro-symbolic reasoner",
            fontsize=11.5, fontweight="bold", color=accent_blue)
    ax.text(2.95, L2_y + 1.20,
            "Graph algorithms over OSM (NetworkX) + LLM-as-planner over Datalog query templates",
            fontsize=9, color=text_color)
    # Question chips
    questions = [
        "Q1 hospitals in flood?",
        "Q3 affected buildings?",
        "Q4 blocked roads (km)?",
        "Q5 isolated communities?",
        "Q7 roads to clear first?",
        "Q9 population disconnected?",
    ]
    for i, q in enumerate(questions):
        row, col = i // 3, i % 3
        x = 2.95 + col * 2.85
        y = L2_y + 0.7 - row * 0.32
        ax.text(x, y, q, fontsize=7.5, color=text_color,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#c0c8d4"))
    ax.text(11.0, L2_y + 0.85, "◐ DEMO\nLIVE", ha="center", fontsize=9,
            fontweight="bold", color="#2453a8",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8edf9", ec="#2453a8"))

    # ----- LAYER 3 (top) · planned -----
    L3_y = 5.6
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.7, L3_y), 8.7, 1.7, boxstyle="round,pad=0.05",
        facecolor=layer_plan, edgecolor="#7a8190", linewidth=2.0, linestyle="--",
    ))
    ax.text(2.95, L3_y + 1.46, "Layer 3 · RL policy",
            fontsize=11.5, fontweight="bold", color="#5a6577")
    ax.text(2.95, L3_y + 1.20,
            "Meta-RL across a curated atlas of ≥30 historical disasters · PPO with action-space:",
            fontsize=9, color=text_color)
    actions = [
        "task imagery", "ask label", "issue alert", "dispatch responder",
    ]
    for i, a in enumerate(actions):
        x = 2.95 + i * 2.05
        ax.text(x, L3_y + 0.78, f"⊳ {a}", fontsize=8, color=text_color,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#a8b0bf"))
    ax.text(2.95, L3_y + 0.38,
            "Reward = time-saved + lives-saved + labels-not-wasted",
            fontsize=8.5, color=muted, style="italic")
    ax.text(11.0, L3_y + 0.85, "· PLANNED", ha="center", fontsize=10,
            fontweight="bold", color="#5a6577",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f3", ec="#7a8190"))

    # ----- Vertical data-flow arrows between layers -----
    for y0, y1 in [(L1_y + 1.2, L2_y), (L2_y + 1.7, L3_y)]:
        ax.annotate("", xy=(7, y1), xytext=(7, y0),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.8))
    ax.text(7.18, L1_y + 1.4, "pixel mask\n+ uncertainty", fontsize=7.5, color=muted)
    ax.text(7.18, L2_y + 1.8, "graph state\n+ answers", fontsize=7.5, color=muted)

    # ----- Input arrows to Layer 1 -----
    for i, name in enumerate(["Sentinel-1", "Sentinel-2", "AlphaEarth"]):
        ax.annotate("", xy=(2.7, L1_y + 0.85 - i * 0.15),
                    xytext=(2.1, 5.6 - i * 0.65),
                    arrowprops=dict(arrowstyle="->", color=arrow_color,
                                    lw=1.0, alpha=0.5))

    # ----- Input arrows to Layer 2 (OSM, WorldPop, JRC) -----
    for i, name in enumerate(["OSM", "WorldPop", "JRC GSW"]):
        ax.annotate("", xy=(2.7, L2_y + 0.5 - i * 0.18),
                    xytext=(2.1, 5.6 - (i + 3) * 0.65),
                    arrowprops=dict(arrowstyle="->", color=arrow_color,
                                    lw=1.0, alpha=0.5))

    # ----- OUTPUTS (right) -----
    ax.add_patch(mpatches.FancyBboxPatch(
        (11.9, 1.5), 1.9, 5, boxstyle="round,pad=0.06",
        facecolor="#dcecdc", edgecolor="#1c7f4f", linewidth=1.5,
    ))
    ax.text(12.85, 6.2, "OUTPUTS", ha="center", fontsize=10,
            fontweight="bold", color="#1c5e3a")
    outputs = [
        ("Briefing", "1-page text"),
        ("Dispatch report", "structured JSON"),
        ("Impact map", "GeoTIFF + overlays"),
        ("Action plan", "ranked decisions"),
        ("Atlas entry", "for self-improve"),
    ]
    for i, (name, desc) in enumerate(outputs):
        y = 5.6 - i * 0.75
        ax.text(12.0, y, "▸", color="#1c5e3a", fontsize=10, fontweight="bold")
        ax.text(12.2, y, f"{name}", fontsize=8.5, fontweight="bold", color=text_color)
        ax.text(12.2, y - 0.18, desc, fontsize=7, color=muted)

    # ----- Output arrows from Layer 2 + Layer 3 to OUTPUTS box -----
    for i, y in enumerate([L2_y + 1.0, L3_y + 1.0]):
        ax.annotate("", xy=(11.9, 4.7 - i * 0.8), xytext=(11.4, y),
                    arrowprops=dict(arrowstyle="->", color=arrow_color,
                                    lw=1.0, alpha=0.5))

    # ----- Bottom footer -----
    ax.text(7, 0.8,
            "End-to-end metric: time-to-answer on the 10-question UN OCHA emergency questionnaire",
            ha="center", fontsize=10, fontweight="bold", color=accent_blue)
    ax.text(7, 0.4,
            "Baseline (manual expert workflow): 1–3 days     ·     Our target with Layers 1+2+3: 30 minutes",
            ha="center", fontsize=9, color=text_color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path


if __name__ == "__main__":
    print(f"Saved: {render()}")
