"""Fig 0 — GeoDisaster-FM Dispatcher (DeepSeek-R1 / Nature pipeline style).

Design principle: a single horizontal pipeline. Each stage is a large
box with ONE verb (Perception / Reasoning / Decision) and one short
descriptor; inputs and outputs are category-level abstractions, not
specific dataset names. Status badges and metrics live below, not
inside, the stage boxes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Nimbus Sans L", "DejaVu Sans",
]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["axes.unicode_minus"] = False


# Palette
INPUT_FILL   = "#e9eef5"
INPUT_EDGE   = "#7e8a9d"

TEAL_FILL    = "#cde9ec"
TEAL_EDGE    = "#0f6a72"

NAVY_FILL    = "#cbd8ee"
NAVY_EDGE    = "#1a3d6e"

GREY_FILL    = "#ececef"
GREY_EDGE    = "#7d838c"

OUT_FILL     = "#dce8df"
OUT_EDGE     = "#3a6b48"

BG           = "#fbfbfc"
EDGE         = "#1a202c"
BODY_GREY    = "#525861"


def _box(ax, x, y, w, h, *, fc, ec, lw=1.6, dashed=False):
    rec = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    if dashed:
        rec.set_linestyle((0, (6, 3)))
    ax.add_patch(rec)


def _pill(ax, cx, y, text, *, fg, bg, w=1.3, h=0.32):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=bg, edgecolor=fg, linewidth=0.9,
    ))
    ax.text(cx, y + h / 2, text, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=fg)


def _arrow(ax, x0, y0, x1, y1, *, lw=2.0, color=EDGE):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=lw, color=color, shrinkA=4, shrinkB=4, zorder=4,
    ))


def render(out_path: str | Path = "outputs/figures/fig0_architecture.png") -> Path:
    fig = plt.figure(figsize=(15, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.set_facecolor(BG)
    ax.axis("off")

    # =====================================================================
    # TITLE
    # =====================================================================
    ax.text(7.5, 7.05,
            "GeoDisaster-FM Dispatcher",
            ha="center", fontsize=24, fontweight="bold", color=EDGE)
    ax.text(7.5, 6.60,
            "From satellite imagery to emergency decisions in minutes",
            ha="center", fontsize=13, color=BODY_GREY)

    # =====================================================================
    # FIVE-STAGE HORIZONTAL PIPELINE
    # =====================================================================
    y_box = 3.30
    h_box = 2.05
    box_w = 2.45
    gap   = 0.55
    x0    = 0.40
    centres = [x0 + i * (box_w + gap) + box_w / 2 for i in range(5)]

    # Layout convention inside each box:
    #   header_y = y_box + h_box - 0.50   (big stage name)
    #   body_y   = y_box + 0.70           (single descriptor line / chips)
    header_y = y_box + h_box - 0.50
    body_y   = y_box + 0.70

    # ----- INPUT (stage 0) -----
    _box(ax, x0, y_box, box_w, h_box, fc=INPUT_FILL, ec=INPUT_EDGE, lw=1.4)
    ax.text(centres[0], header_y,
            "Input", ha="center", fontsize=15,
            fontweight="bold", color=INPUT_EDGE)
    ax.text(centres[0], body_y + 0.45,
            "satellite",
            ha="center", fontsize=11, color=EDGE)
    ax.text(centres[0], body_y + 0.15,
            "maps",
            ha="center", fontsize=11, color=EDGE)
    ax.text(centres[0], body_y - 0.15,
            "demographics",
            ha="center", fontsize=11, color=EDGE)

    # ----- LAYER 1: PERCEPTION -----
    _box(ax, x0 + 1 * (box_w + gap), y_box, box_w, h_box,
         fc=TEAL_FILL, ec=TEAL_EDGE, lw=1.8)
    ax.text(centres[1], header_y,
            "Perception", ha="center", fontsize=17,
            fontweight="bold", color=TEAL_EDGE)
    ax.text(centres[1], body_y + 0.15,
            "Layer 1",
            ha="center", fontsize=10, color=BODY_GREY,
            fontweight="bold")

    # ----- LAYER 2: REASONING (focal) -----
    _box(ax, x0 + 2 * (box_w + gap), y_box, box_w, h_box,
         fc=NAVY_FILL, ec=NAVY_EDGE, lw=2.4)
    ax.text(centres[2], header_y,
            "Reasoning", ha="center", fontsize=17,
            fontweight="bold", color=NAVY_EDGE)
    ax.text(centres[2], body_y + 0.15,
            "Layer 2",
            ha="center", fontsize=10, color=BODY_GREY,
            fontweight="bold")

    # ----- LAYER 3: DECISION -----
    _box(ax, x0 + 3 * (box_w + gap), y_box, box_w, h_box,
         fc=GREY_FILL, ec=GREY_EDGE, lw=1.4, dashed=True)
    ax.text(centres[3], header_y,
            "Decision", ha="center", fontsize=17,
            fontweight="bold", color=GREY_EDGE)
    ax.text(centres[3], body_y + 0.15,
            "Layer 3",
            ha="center", fontsize=10, color=BODY_GREY,
            fontweight="bold")

    # ----- OUTPUT (stage 4) -----
    _box(ax, x0 + 4 * (box_w + gap), y_box, box_w, h_box,
         fc=OUT_FILL, ec=OUT_EDGE, lw=1.4)
    ax.text(centres[4], header_y,
            "Output", ha="center", fontsize=15,
            fontweight="bold", color=OUT_EDGE)
    ax.text(centres[4], body_y + 0.45,
            "briefing",
            ha="center", fontsize=11, color=EDGE)
    ax.text(centres[4], body_y + 0.15,
            "map",
            ha="center", fontsize=11, color=EDGE)
    ax.text(centres[4], body_y - 0.15,
            "action plan",
            ha="center", fontsize=11, color=EDGE)

    # ----- Arrows between stages -----
    for i in range(4):
        x_start = x0 + (i + 1) * box_w + i * gap
        x_end   = x_start + gap
        _arrow(ax, x_start, y_box + h_box / 2, x_end, y_box + h_box / 2,
               lw=2.2, color=EDGE)

    # =====================================================================
    # STATUS / METRIC ROW under each stage
    # =====================================================================
    y_status = y_box - 0.55
    # Input: no status
    # Layer 1
    _pill(ax, centres[1], y_status, "VALIDATED", fg=TEAL_EDGE, bg="white", w=1.25)
    ax.text(centres[1], y_status - 0.35,
            "F1 = 0.85   (10-region avg 0.83)",
            ha="center", fontsize=9.5, color=EDGE, fontweight="bold")
    # Layer 2
    _pill(ax, centres[2], y_status, "IMPLEMENTED", fg=NAVY_EDGE, bg="white", w=1.45)
    ax.text(centres[2], y_status - 0.35,
            "10 emergency questions answered",
            ha="center", fontsize=9.5, color=EDGE, fontweight="bold")
    # Layer 3
    _pill(ax, centres[3], y_status, "PLANNED", fg=GREY_EDGE, bg="white", w=1.05)
    ax.text(centres[3], y_status - 0.35,
            "≥ 30 historical disasters atlas",
            ha="center", fontsize=9.5, color=BODY_GREY, fontweight="bold")
    # Output
    ax.text(centres[4], y_status - 0.15,
            "1 minute / question",
            ha="center", fontsize=9.5, color=BODY_GREY,
            fontweight="bold", style="italic")

    # =====================================================================
    # BOTTOM RIBBON — headline claim
    # =====================================================================
    foot_y = 0.55
    foot_h = 0.95
    _box(ax, 0.40, foot_y, 14.20, foot_h, fc=NAVY_EDGE, ec=NAVY_EDGE, lw=0)

    ax.text(1.10, foot_y + foot_h / 2 + 0.13,
            "End-to-end response time",
            fontsize=11, color="#c4d4ec", va="center")
    ax.text(1.10, foot_y + foot_h / 2 - 0.20,
            "raw imagery → responder answers",
            fontsize=9, color="#94aacb", va="center", style="italic")

    # Baseline value
    ax.text(6.40, foot_y + foot_h / 2,
            "1–3 days",
            fontsize=20, color="white", ha="center", va="center",
            fontweight="bold")
    ax.text(6.40, foot_y + foot_h / 2 - 0.34,
            "manual expert workflow",
            fontsize=9, color="#94aacb", ha="center", va="center")

    # Arrow
    _arrow(ax, 7.40, foot_y + foot_h / 2, 9.60, foot_y + foot_h / 2,
           lw=2.6, color="white")

    # Target value
    ax.text(10.90, foot_y + foot_h / 2,
            "30 minutes",
            fontsize=20, color="white", ha="center", va="center",
            fontweight="bold")
    ax.text(10.90, foot_y + foot_h / 2 - 0.34,
            "Layer 1 + 2 + 3 (target)",
            fontsize=9, color="#94aacb", ha="center", va="center")

    # Improvement factor
    ax.text(13.85, foot_y + foot_h / 2 + 0.12,
            "≥ 100×",
            fontsize=24, color="white", ha="center", va="center",
            fontweight="bold")
    ax.text(13.85, foot_y + foot_h / 2 - 0.30,
            "faster",
            fontsize=10, color="#c4d4ec", ha="center", va="center")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG, pad_inches=0.25)
    plt.close()
    return out_path


if __name__ == "__main__":
    print(f"Saved: {render()}")
