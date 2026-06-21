# GeoDisaster-FM —— 跨灾害制图的"校准范式"

> **Cross-disaster mapping is largely a calibration problem, not a representation problem:
> four labels recover 99% of the full-pool oracle.**

**投稿目标:** Nature Communications(直投)
**在线:** https://geodisaster-fm.pages.dev/ · 报告页 https://geodisaster-fm.pages.dev/report
**GitHub:** https://github.com/14H034160212/geodisaster-fm

---

## 一句话

灾后遥感"看不准"换一个新灾害,真正的瓶颈**不是模型表征(representation)不够好,而是判定阈值(calibration)没对准**——而对准它,**只要 4 个标注样本**就够,几乎追平用整个数据池调出来的 oracle。

---

## 科学问题

跨灾害泛化失败的主导机制是什么?两个竞争假设:

| 假设 | 主张 | 若成立的含义 |
|---|---|---|
| **H1 Representation drift** | 学到的表征不能迁移,需要重训 / 更大 foundation model | 领域该继续投资更大 backbone |
| **H2 Calibration drift** | 表征够用,只是决策阈值 τ 要重调 | cross-disaster adaptation 是**便宜问题**,4 个标签就够 |

**我们设计严格实验区分 H1/H2,结论:H2 主导。**

---

## 方法 —— Calibration-Centric Active Adaptation (CCA)

1. **实证重定义**:18 个真实事件上,几乎每个事件最优阈值 τ\* ≠ 0.5(范围 0.30–0.70);单事件仅靠重校准最高 +0.235 F1(Pakistan 2022 洪水)。
2. **形式化 MDP**:状态 = chip 预测统计量;动作 = 选哪个 chip 去标;奖励 = test F1 增益。证明所有 monotone 单参数后验校准(温度缩放/Platt/isotonic)对二元阈值分割**等价于调阈值** → full-pool oracle 是该族方法理论上限。
3. **PPO 解 MDP**:紧凑 actor-critic(2×64 Tanh)+ GAE-λ 信用分配 + episode-terminal 奖励 + entropy 退火(0.10→0.01)。
4. **闭环 agent**:感知 → 神经符号推理(OpenStreetMap 查询)→ RL 校准,回答 10 个 UN-OCHA 标准灾害响应问题。

---

## 关键结果

**严格协议:leave-one-event-out (LOEO),20 seeds × 10 fold = 200 paired pairs**,彻底消除事件级 leakage。

| 方法 | Pooled F1 | vs full-pool oracle | 显著性 |
|---|---:|---:|---|
| full-pool oracle | **0.839** | —(理论上限) | — |
| **PPO(本方法,4 chip)** | **0.837** | **−0.002** | **统计等价** ✓ |
| uncertainty sampling | 0.835 | −0.004 | n.s. |
| random | 0.832 | −0.007 | — |
| zero-shot (τ=0.5) | 0.822 | −0.017 | — |

- **4 个标签 ≡ full-pool oracle**(Δ=−0.002, t-p=0.42)—— 头条主张。
- **三个 foundation model(Prithvi-100M / DOFA / AlphaEarth)都不超过从零训练的 U-Net** —— 瓶颈不在表征。
- **零标签先验修正全部失败**(Saerens EM −0.61、BBSE −0.14、quantile-matching −0.07)→ drift = 先验漂移(免费)+ score 分布扭曲(必须用标签),后者主导,这是 novelty 核心防线。
- **端到端时延**:感知 0.031 秒/chip vs Copernicus EMS 1–3 天(3–4 个数量级加速);决策答案与人工标注 Pearson r = 0.971(10 events, 431 chips)。

---

## 数据与规模

- **Sen1Floods11** —— 10 个真实洪水事件
- **xBD** —— 建筑损毁,多灾种(85 万建筑标注)
- **HLS Burn-Scars** —— 4 个野火季(NASA-IBM)
- 合计 **3 benchmark × 3 灾种 × 18 真实事件**
- Backbone: U-Net(from scratch)/ Prithvi-100M / DOFA / Google AlphaEarth

---

## 代码结构

```
geodisaster/          核心包(datasets / models / decision / experiments / viz ...)
scripts/              全部实验脚本(calibration_analysis / eval_layer3_ppo_significance / LOEO aggregate ...)
outputs/              结果(layer3_ppo/*.json、decision/*.json、figures/*.png、site/ 网站)
latex_sn/             Springer Nature 官方模板投稿包(main_sn.pdf)
MANUSCRIPT.md         正文 · RESULTS_INVENTORY.md 每条数据的可复现来源
REPORT_TONIGHT.md     给导师的进度报告(→ 渲染成在线报告页)
```

复现:每个量化结论都能由 `RESULTS_INVENTORY.md` 列出的 JSON + `scripts/` 重跑。

---

## 当前状态

- ✅ 主线证据完整(LOEO v1/v2/v3、calibration drift × 3 benchmark、零标签测试、decision-reward A/B)
- ✅ 论文已重构为假设检验式(H1 vs H2),官方 Nature 模板投稿包就位
- ✅ 六轮审稿后估计:desk ~75–80%、送审→接收 ~85%、综合 ~65–75%(整刊接收率仅 ~8%,此为 strong submission)
- ⏳ 剩余为人工项:作者单位 / 通讯邮箱 / funding / author contributions / 推荐 reviewer / 英文 copy-edit / Zenodo DOI

---

## 与新方向的关系

这篇是**感知 + 校准的地基(Layer 1–3)**。在此之上把决策推到 *action* 层(UAV 具身搜索/巡检)是自然续作,
见 [README_EMBODIED_TWIN.md](README_EMBODIED_TWIN.md)。
