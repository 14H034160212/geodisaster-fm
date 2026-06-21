"""Render the bridge figure linking the two works:
Paper 1 (GeoDisaster-FM, Nature Communications) -> Paper 2 (Embodied UAV Disaster Digital Twin).

Output: outputs/figures/fig_twin_bridge.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("outputs/figures/fig_twin_bridge.png")

FG = "#1a1a1a"; MUTED = "#6b7280"; LINE = "#cfcabc"
BLUE = "#1f5fbe"; GREEN = "#1c7f4f"; SAND = "#fbfaf6"
P1_FILL = "#eaf1fb"; P2_FILL = "#e8f3ec"; SHARED_FILL = "#f4f1e6"


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.03):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1))


def txt(ax, x, y, s, size=10, weight="normal", color=FG, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


fig, ax = plt.subplots(figsize=(12, 6.8))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
fig.patch.set_facecolor(SAND); ax.set_facecolor(SAND)

# ---- Title ----
txt(ax, 6, 6.7, "From Calibrated Perception to Embodied Decision", size=15, weight="bold")
txt(ax, 6, 6.32, "GeoDisaster-FM 的两阶段路线:第一篇打地基,第二篇推到具身决策", size=10, color=MUTED)

# ---- Paper 1 box ----
box(ax, 0.4, 2.7, 5.1, 3.2, P1_FILL, BLUE, lw=2)
txt(ax, 2.95, 5.55, "论文 1 · GeoDisaster-FM", size=12.5, weight="bold", color=BLUE)
txt(ax, 2.95, 5.18, "Nature Communications(投稿中)", size=9.5, color=MUTED)
txt(ax, 2.95, 4.72, "Perception + Calibration  (Layer 1–3)", size=10.5, weight="bold")
for i, line in enumerate([
    "• 跨灾害瓶颈 = 校准,不是表征",
    "• 4 个标签 ≈ full-pool oracle (99%)",
    "• 18 真实事件 · 3 灾种 · 3 基础模型",
    "• Layer 3 PPO:在真实数据上做决策",
]):
    txt(ax, 2.95, 4.35 - i * 0.36, line, size=9.5, ha="center")

# ---- Paper 2 box ----
box(ax, 6.5, 2.7, 5.1, 3.2, P2_FILL, GREEN, lw=2)
txt(ax, 9.05, 5.55, "论文 2 · Disaster Digital Twin", size=12.5, weight="bold", color=GREEN)
txt(ax, 9.05, 5.18, "Embodied UAV(新方向)", size=9.5, color=MUTED)
txt(ax, 9.05, 4.72, "Action + Embodiment  (Layer 4)", size=10.5, weight="bold")
for i, line in enumerate([
    "• 真实数据做数字孪生真值图层",
    "• T1 巡检 · T2 避水导航 · T3 搜救 · T4 优先级",
    "• rule / vision / VLM / RL agent 对比",
    "• 物理模拟补动态淹没层(非 UE5 仿真)",
]):
    txt(ax, 9.05, 4.35 - i * 0.36, line, size=9.5, ha="center")

# ---- Bridge arrow ----
arr = FancyArrowPatch((5.55, 4.3), (6.45, 4.3), arrowstyle="-|>", mutation_scale=26,
                      lw=2.4, color=FG)
ax.add_patch(arr)
txt(ax, 6.0, 4.62, "复用 + 扩展", size=9, weight="bold", color=FG)

# ---- Shared foundation band ----
box(ax, 0.4, 0.5, 11.2, 1.7, SHARED_FILL, LINE, lw=1.4)
txt(ax, 6, 1.9, "共享地基 / Shared foundation", size=11, weight="bold", color="#7a5a1f")
txt(ax, 3.0, 1.42, "真实数据(已就绪)", size=9.5, weight="bold")
txt(ax, 3.0, 1.08, "xBD · Sen1Floods11 · HLS Burn-Scars", size=9, color=MUTED)
txt(ax, 3.0, 0.78, "+ RescueNet(真实 UAV)· OSM · DEM", size=9, color=MUTED)
txt(ax, 6.0, 1.42, "Layer 3 PPO 决策器", size=9.5, weight="bold")
txt(ax, 6.0, 1.0, "校准选择 → 具身优先级/搜索", size=9, color=MUTED)
txt(ax, 9.0, 1.42, "物理模拟", size=9.5, weight="bold")
txt(ax, 9.0, 1.0, "GeoClaw + Physics-Informed Video Diffusion", size=8.7, color=MUTED)

# light connectors from boxes down to shared band
for x0 in (2.95, 9.05):
    ax.add_patch(FancyArrowPatch((x0, 2.68), (x0, 2.22), arrowstyle="-", lw=1.0,
                                 color=LINE, linestyle=(0, (4, 3))))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=SAND)
print(f"Saved {OUT}")
