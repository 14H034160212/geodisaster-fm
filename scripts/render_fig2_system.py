"""Fig 2 — System architecture of the calibration-centric agent.

A clean methods/system diagram matching the current manuscript: a frozen
perception model whose pixel ranking transfers across events, an
active-calibration stage that spends four labels to recover the
event-optimal threshold tau*, a neuro-symbolic reasoning layer over
OpenStreetMap, and the decision-level briefing. The H2 message — the
only per-event learning is the one-parameter threshold — is annotated
on the calibration stage.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

fig, ax = plt.subplots(figsize=(13.5, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

# palette
C_INPUT = "#e8eef5"; C_PERC = "#cfe3f3"; C_CAL = "#1f5fbe"; C_REA = "#d8ecdc"; C_OUT = "#e9e4f5"
EDGE = "#33415c"

def box(x, w, color, title, lines, title_color="#16243b", body_color="#2a2a2a", y=22, h=12, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                facecolor=color, edgecolor=EDGE, linewidth=1.2, alpha=alpha))
    ax.text(x + w/2, y + h - 2.2, title, ha="center", va="top", fontsize=12.5,
            fontweight="bold", color=title_color)
    ax.text(x + w/2, y + h - 5.6, lines, ha="center", va="top", fontsize=8.6,
            color=body_color, linespacing=1.45)

def arrow(x0, x1, y=28, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=EDGE))
    if label:
        ax.text((x0+x1)/2, y+1.4, label, ha="center", va="bottom", fontsize=7.6,
                color="#444", style="italic")

# stage boxes
box(1.5, 17, C_INPUT, "Event imagery",
    "Sentinel-1 / Sentinel-2\nor HLS, event-day\n(no labels)", y=22, h=13)
box(22, 18, C_PERC, "Frozen perception",
    "pre-trained segmenter\n(U-Net or foundation)\n→ per-pixel scores\n\\textit{ranking transfers}".replace("\\textit{","").replace("}",""),
    y=22, h=13)
box(43.5, 19, C_CAL, "Active calibration",
    "pick 4 chips $\\rightarrow$ fit $\\tau^*$\n(MDP / PPO; §Methods)\n$\\rightarrow$ calibrated mask",
    title_color="white", body_color="white", y=22, h=13)
box(65.5, 16.5, C_REA, "Neuro-symbolic\nreasoning",
    "OSM buildings / roads /\nhospitals; graph\nconnectivity", y=22, h=13)
box(84, 14.5, C_OUT, "Decision\nbriefing",
    "10 UN-OCHA answers\nJSON + Markdown\n~minutes/event", y=22, h=13)

# arrows between
arrow(18.7, 21.7, y=28.5)
arrow(40.2, 43.2, y=28.5)
arrow(62.7, 65.2, y=28.5, label=r"$\tau^*$")
arrow(82.2, 83.7, y=28.5)

# H2 annotation under the calibration box
ax.add_patch(FancyBboxPatch((43.5, 7), 19, 11, boxstyle="round,pad=0.3,rounding_size=1.0",
                            facecolor="#fdf3e3", edgecolor="#c98a23", linewidth=1.0))
ax.text(53, 16.5, "The only per-event learning", ha="center", va="top",
        fontsize=8.2, fontweight="bold", color="#9a6a12")
ax.text(53, 13.2,
        "One scalar $\\tau$, 4 labels $\\rightarrow$\n99% of the full-pool oracle.\nNo retraining, no new\nrepresentation (H2).",
        ha="center", va="top", fontsize=7.8, color="#5c4410", linespacing=1.4)
ax.add_patch(FancyArrowPatch((53, 18), (53, 22), arrowstyle="-|>", mutation_scale=13,
                             lw=1.3, color="#c98a23"))

# bottom speed bar
ax.add_patch(FancyBboxPatch((1.5, 0.5), 97, 4.4, boxstyle="round,pad=0.2,rounding_size=0.8",
                            facecolor="#16243b", edgecolor="none"))
ax.text(4, 2.7, "End-to-end machine time (event imagery → responder briefing):",
        ha="left", va="center", fontsize=8.8, color="#cfe0f5")
ax.text(78, 2.7, "0.031 s / chip  ·  minutes / event", ha="left", va="center",
        fontsize=9.4, color="white", fontweight="bold")

ax.text(50, 40, "A four-label calibration agent for cross-disaster mapping",
        ha="center", va="top", fontsize=13.5, fontweight="bold", color="#16243b")

fig.tight_layout()
out = Path("outputs/figures/fig2_system_architecture.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved {out}")
