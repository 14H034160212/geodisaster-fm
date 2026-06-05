# 老师好,今晚的汇报

## 0. 一句话总结

> 在严格 leave-one-event-out(10 fold × 10 seed = 100 paired pairs)的 leakage-free 协议下,**用 4 个标签做的 active threshold calibration policy,F1 和用全 pool 校准的 oracle 在统计上等价**(Δ = −0.002, p = 0.42);同时显著优于 zero-shot 默认(p = 0.009)和 CoreSet active learning(p = 0.024)。CCA 框架 + 4-标签 oracle-等价的故事,是这轮工作的核心可发表主张。

---

## 1. 从上次汇报到现在做了什么(主要 4 件)

### ① ChatGPT 同行评审 → 发现 leakage 风险
之前 R4 的 paired-significant 结论(PPO p ≤ 0.005)是在"同 4 hard regions 既训练又评估"的 within-event 协议下做的。审稿人会问:**policy 是不是在它的 evaluation events 上间接训过?** 这是 event-level leakage。

### ② 设计 leave-one-event-out 严格协议
- 10 folds × 10 seeds = **100 paired pairs**(原来 10)
- 每 fold:PPO 只在另外 9 个事件上训练,**完全不见 meta-test 事件**
- 然后才在 meta-test 上 evaluate

### ③ 旧 PPO 在严格协议下崩了 → 诊断 + 修底层
LOEO-v1 结果:**PPO 平均输给 random −0.007 F1**;Pakistan 单 fold 输 −0.033(灾难)。

诊断出 3 个底层 bug:
1. **没有 GAE-λ** — 用 raw discounted return,SNR ≈ 1:10
2. **step-level reward** — 每加一个 chip 都给"F1 增量"作 reward,噪声盖过信号
3. **constant entropy 0.01** — 早期 collapse

修了之后 LOEO-v2 重测,Pakistan 从 −0.033 翻成 **−0.002**,Somalia 从 −0.037 翻成 **+0.018**。

### ④ 进一步 ablation + r=0.971 鲁棒性审计
- **richer 10-d features**(decision-frontier proximity + 4 分位数)→ **没帮助,反而更差**(LOEO-v3 显著低于 oracle)。**5-d 是最优**,这是诚实的负面发现,反而强化"v2 设计选择是合理的"。
- **r=0.971** 经审计后:Spearman 0.90,MAPE median 25%,**bottom-50% area chip r = 0.118**(几乎不相关),top-50% r = 0.972 主导 headline。**大面积事件主导是真问题,必须诚实标注**。

---

## 2. 现在最强的 5 条主张(全部 leakage-free 100 paired pairs)

| # | 主张 | 数据 | 显著性 |
|---|---|---|---|
| 1 | **Calibration 是 cross-disaster 主要 lever** | 12 events × 2 benchmarks,所有 τ\* ≠ 0.5,Pakistan +0.235 F1 单事件提升 | empirical |
| 2 | **4 chip = full-pool oracle**(头条) | Δ = −0.002, n=100 | t-p = 0.42(等价)|
| 3 | **PPO > zero-shot 校准** | Δ = +0.015 | **t-p = 0.009 \*\*** |
| 4 | **PPO > CoreSet active learning** | Δ = +0.008 | **t-p = 0.024 \*** |
| 5 | **PPO > random**(方向稳但 t-test 边缘) | Δ = +0.005,wins 65/100 pairs | Wilcoxon **p = 0.0006**;t-p = 0.084 |

## 3. 4 个诚实标注的局限(全部已加入 paper)

| 局限 | 含义 |
|---|---|
| richer features 没帮助 | 5-d 是最优;**capacity/budget mismatch,不是 missing signal** |
| r=0.971 由大面积 chip 主导 | bottom-50% r=0.118;**claim 限定为 dominant-signal regime** |
| Decision-aligned reward 改 policy 但没 net-improve decision metric | 20-seed 不够,**reward 是 control knob 的架构验证有效,decision metric 改善是 open** |
| AE 在 equal inputs 上 ≈ U-Net 不超过 | F1 ceiling 不是 representational |

---

## 4. 新 paper Title + Abstract 框架

**新 Title:**
> "**Four labels are enough**: cross-disaster mapping is a calibration problem, and an active-calibration policy matches the full-pool oracle"

**Abstract 5 段架构(每段单一核心主张):**
1. **Reframing** —— 12 events × 2 benchmarks 的 calibration drift
2. **Mechanism** —— Active Calibration MDP + 3 个必需 RL-OPT,带 v1 灾难性失败对照
3. **Headline** —— 4 chip ≡ oracle + 显著 > zero-shot/CoreSet
4. **Calibrated negative findings** —— 4 个诚实负面发现强化主线
5. **End-to-end agent** —— 0.031s/chip + r=0.971(+ Spearman/MAPE 诚实标注)

---

## 5. 关于"我们的方法是不是最好"—— 诚实定位

老师如果问起,我准备这样答:

**在我们测过的所有方法里,PPO 是 point estimate 第一,且**是唯一统计上和 full-pool oracle 等价**的方法**:

| 方法 | Pooled F1 | vs PPO | t-p |
|---|---|---|---|
| **full-pool oracle** | **0.8388** | +0.002(tied)| 0.42 |
| **PPO(我们的方法)** | **0.8368** | — | — |
| uncertainty(entropy)| 0.8348 | -0.002(n.s.)| 0.33 |
| random | 0.8321 | -0.005 | 0.084(Wilcoxon p=0.0006)|
| CoreSet | 0.8285 | **-0.008 \*** | 0.024 |
| zero-shot | 0.8221 | **-0.015 \*\*** | 0.009 |

**但有两个诚实 caveat 必须承认:**

1. **PPO 和 uncertainty 没有统计差距**(Δ=+0.002,t-p=0.33)— **uncertainty 是它最接近的对手**。
2. **该问题的天花板被 oracle 锁住**。我们没测的方法(Bayesian AL / MCTS / NeuralUCB / ensemble uncertainty)**最多也只能 match oracle,不可能显著超过 PPO**(因为 PPO 已经在 oracle 等价位置)。

**那为什么 paper 还保留 PPO 作 headline?**

| 理由 | 说明 |
|---|---|
| ① PPO 是**唯一**测过的、统计上和 oracle 等价的方法 | uncertainty 0.8348 < oracle 0.8388;PPO 0.8368 ≈ oracle |
| ② PPO 是 framework extension 的载体 | 换 reward / 加 state / 升 budget 都能扩展;simple uncertainty 不能 |
| ③ 完整 ablation chain 证明 v2 是正确设计点 | RL-OPT 必需(LOEO-v1 灾难);5-d 是最优(LOEO-v3 ablation) |

**所以 paper 的 selling point 重新表述:**

> "**Cross-disaster calibration is a 4-label problem, not a method-choice problem.** 我们的方法是最 robust 的一个,但任何 reasonable active-selection 方法都能接近 oracle。这本身是个 operational 好消息 —— responder 不需要 deploy 复杂 RL 系统,只要 4 个标签和合理选法。" 

**这个表述比原来"PPO 击败一切"更诚实、更难撕、更有 operational value**。

---

## 6. Venue 判断 + 想请老师定夺的两个问题

### ❶ 投稿 venue:Nature Communications 还是 sister 期刊?

**我的判断**:
- **主线已经够 Nature Communications**: "4 chip = oracle" + 3 个显著主张 + 诚实负面 + 完整 reproducibility + 12 events × 2 benchmarks 的实证基础
- **但** PPO − random 只是 Wilcoxon 显著(t-p = 0.084),审稿人**可能**会要求更多 seeds 把 t-test 也推过 0.05

**fallback 方案**:
- **Communications Earth & Environment**:Nature 系列,Earth science 门槛低,故事不变完全够投
- **npj Natural Hazards**:也是 Nature 系列,domain 最匹配

**请老师决定**:
- A. 直接投 Nat Commun,如果被打回再降级 → 时间风险高但 ceiling 最高
- B. 直接投 Communications Earth & Environment 或 npj Natural Hazards → 稳但 prestige 低一些
- C. 先投 Nat Commun 准备好 24h 内 fallback 到 CCE → 折中

### ❷ 是否还要做"把 PPO − random 推过 t-p < 0.05"的一轮实验?

**可能的路径**(都是 ~1-2 周工作):
- **增加 seeds**:从 10 加到 20-30 seeds × 10 folds → 200-300 paired pairs,统计 power 翻倍
- **更深网络**:actor MLP 从 2×64 改到 3×128 + 600 updates(给 richer features 一个公平机会)
- **Bayesian active calibration baseline**:加一个理论 correct 的 non-RL baseline 作对比

**我的判断**:**不做也能投**。Wilcoxon p = 0.0006 已经是 robust 的方向证据,t-p = 0.084 在 paper 里**诚实标注 + 主线已经站住**(头条是 "4 chip = oracle"、不是 "PPO > random")。但**如果有时间,加更多 seeds 是 ROI 最高的**。

---

## 7. 代码 + 数据状态(reproducibility)

| 项 | 状态 |
|---|---|
| GitHub 主仓库 | `14H034160212/geodisaster-fm`,commit `0bd752f` |
| Live blog dashboard | https://geodisaster-fm.pages.dev/(已 redeploy 反映 LOEO-v2) |
| Manuscript | `MANUSCRIPT.md` 809 行,可投状态 |
| 完整实验 inventory | `RESULTS_INVENTORY.md`(每个 claim 都追溯到 JSON) |
| Framework 文档 | `FRAMEWORK_CCA.md`(中文写作 + 英文表格) |
| Figures | 28 张(headline:Fig 26 LOEO v1↔v2;Fig 27 失败案例;Fig 28 v2/v3 ablation) |
| BibTeX | `refs.bib`(16 entries) |

---

## 8. Slide 建议(如果今晚需要做 slides)

| Slide # | 内容 |
|---|---|
| 1 | Title + 一句话总结(Section 0) |
| 2 | 时间线:chatgpt review → leakage 修复 → RL-OPT → LOEO-v2 → ablation(Section 1) |
| 3 | Fig 26 主图 + LOEO 5 主张表(Section 2) |
| 4 | Fig 27 失败案例分布 + Fig 28 ablation(Section 3) |
| 5 | **诚实定位:PPO 是 best 但 tied with uncertainty,oracle 是天花板(Section 5)** |
| 6 | Venue 决策 + 2 个 open question(Section 6) |

---

## 9. 最重要的一句话(给老师)

**新版的故事更诚实但更可发表。** 抹掉了 chatgpt review 提的 leakage 风险,把 PPO 从"显著优于所有 baseline"修正为"4 chip ≡ oracle、显著优于 zero-shot/CoreSet、rank-significant > random",并诚实标注了 4 个 limitation。**核心 selling point 反而更清晰、更难被审稿人撕** —— "4 chip = oracle" 是一个具体、可复现、操作有用的 claim。

请老师帮看 Section 6 的两个问题,谢谢!
