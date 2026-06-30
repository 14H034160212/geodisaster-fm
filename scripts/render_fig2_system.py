"""Fig 2 — System architecture of the calibration-centric agent.

Nature-standard methods/system schematic: a frozen perception model whose
per-pixel ranking transfers across events, an active-calibration stage that
spends four labels to recover the event-optimal threshold tau*, a
neuro-symbolic reasoning layer over OpenStreetMap, and the decision-level
briefing. The H2 message — the only per-event learning is the one-parameter
threshold, in contrast to the trained-once-then-frozen perception model — is
encoded visually beneath the pipeline. No in-figure title (the caption carries
it); restrained palette; uniform sans-serif type; thin rules; 300 dpi.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---- Nature-style typography: prefer a Helvetica-like sans, fall back gracefully ----
import matplotlib.font_manager as fm
_avail = {f.name for f in fm.fontManager.ttflist}
_prefs = [f for f in ("Helvetica", "Arial", "Nimbus Sans", "TeX Gyre Heros")
          if f in _avail]
# Prepend the preferred faces so font.family="sans-serif" actually resolves to one.
matplotlib.rcParams["font.sans-serif"] = _prefs + matplotlib.rcParams["font.sans-serif"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["svg.fonttype"] = "none"

# ---- restrained, colour-blind-friendly palette ----
C_INPUT = "#eef2f6"; C_PERC = "#dce7f2"; C_CAL = "#2c6fb5"
C_REA   = "#e1ecdc"; C_OUT  = "#ece6f3"
EDGE    = "#54606e"      # uniform thin edge
INK     = "#1c2733"      # primary text
SUB     = "#445063"      # secondary text
GOLD_BG = "#fbf2e0"; GOLD_ED = "#c08a2a"; GOLD_TX = "#6f4e11"
FROZEN  = "#6b7785"

TITLE_FS = 8.0; BODY_FS = 6.4; SMALL_FS = 5.9; BADGE_FS = 6.6

fig, ax = plt.subplots(figsize=(7.2, 3.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

BOX_Y, BOX_H = 30.0, 12.0
MID = BOX_Y + BOX_H / 2

def stage(x, w, color, n, title, lines, light=False):
    tx = "white" if light else INK
    bx = "#e8eef5" if light else SUB
    ax.add_patch(FancyBboxPatch((x, BOX_Y), w, BOX_H,
                 boxstyle="round,pad=0.3,rounding_size=0.9",
                 facecolor=color, edgecolor=EDGE, linewidth=0.8))
    # numbered stage marker, just above the top-left corner (outside the box)
    ax.add_patch(Circle((x + 2.2, BOX_Y + BOX_H + 1.9), 1.5,
                 facecolor="white", edgecolor=EDGE, linewidth=0.9, zorder=5))
    ax.text(x + 2.2, BOX_Y + BOX_H + 1.9, str(n), ha="center", va="center",
            fontsize=BADGE_FS, fontweight="bold", color=INK, zorder=6)
    ax.text(x + w / 2, BOX_Y + BOX_H - 2.0, title, ha="center", va="top",
            fontsize=TITLE_FS, fontweight="bold", color=tx)
    ax.text(x + w / 2, BOX_Y + BOX_H - 4.9, lines, ha="center", va="top",
            fontsize=BODY_FS, color=bx, linespacing=1.5)

def arrow(x0, x1, label=None):
    ax.add_patch(FancyArrowPatch((x0, MID), (x1, MID), arrowstyle="-|>",
                 mutation_scale=11, lw=1.0, color=EDGE,
                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, MID + 1.2, label, ha="center", va="bottom",
                fontsize=BODY_FS, color=INK)

# ---- five pipeline stages ----
stage(1.0, 17.0, C_INPUT, 1, "Event imagery",
      "Sentinel-1 / Sentinel-2\nor HLS, event-day\n(no labels)")
stage(21.5, 17.5, C_PERC, 2, "Frozen perception",
      "pre-trained segmenter\n(U-Net or foundation)\nper-pixel scores;\nranking transfers")
stage(42.5, 18.0, C_CAL, 3, "Active calibration",
      "pick 4 chips, fit $\\tau^{*}$\n(MDP, PPO; Methods)\ncalibrated binary mask", light=True)
stage(64.0, 16.5, C_REA, 4, "Neuro-symbolic\nreasoning",
      "OSM buildings, roads,\nhospitals; graph\nconnectivity")
stage(84.0, 15.0, C_OUT, 5, "Decision briefing",
      "10 UN-OCHA answers\nJSON + Markdown\n~minutes / event")

arrow(18.0, 21.5)
arrow(39.0, 42.5)
arrow(60.5, 64.0, label=r"$\tau^{*}$")
arrow(80.5, 84.0)

# ---- H1/H2 visual contrast beneath the pipeline ----
# perception = trained once, then frozen (not per-event)
ax.annotate("", xy=(30.25, 28.6), xytext=(30.25, 26.8),
            arrowprops=dict(arrowstyle="-", lw=0.8, color=FROZEN))
ax.text(30.25, 26.4, "trained once, then frozen\n(shared across all events)",
        ha="center", va="top", fontsize=SMALL_FS, color=FROZEN, style="italic",
        linespacing=1.4)

# calibration = the only per-event learning (H2 callout)
ax.add_patch(FancyArrowPatch((51.5, 29.7), (51.5, 24.4), arrowstyle="-|>",
             mutation_scale=10, lw=1.0, color=GOLD_ED))
ax.add_patch(FancyBboxPatch((37.5, 10.5), 28.0, 13.0,
             boxstyle="round,pad=0.3,rounding_size=0.8",
             facecolor=GOLD_BG, edgecolor=GOLD_ED, linewidth=0.9))
ax.text(51.5, 22.6, "The only per-event learning", ha="center", va="top",
        fontsize=BODY_FS + 0.4, fontweight="bold", color=GOLD_ED)
ax.text(51.5, 19.7,
        "one scalar $\\tau$ from 4 labels recovers\n"
        "$\\approx$99% of the full-pool oracle,\n"
        "with no retraining or new representation (H2)",
        ha="center", va="top", fontsize=SMALL_FS, color=GOLD_TX, linespacing=1.5)

# ---- machine-time summary strip ----
ax.add_patch(FancyBboxPatch((1.0, 2.0), 98.0, 3.8,
             boxstyle="round,pad=0.15,rounding_size=0.6",
             facecolor="#1c2733", edgecolor="none"))
ax.text(3.5, 3.9, "End-to-end machine time, event imagery to responder briefing",
        ha="left", va="center", fontsize=SMALL_FS + 0.3, color="#c4d3e6")
ax.text(96.5, 3.9, "0.031 s / chip  ·  minutes / event", ha="right", va="center",
        fontsize=BODY_FS + 0.6, color="white", fontweight="bold")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
out = Path("outputs/figures/fig2_system_architecture.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
plt.close()
print(f"Saved {out} and {out.with_suffix('.pdf')}")
