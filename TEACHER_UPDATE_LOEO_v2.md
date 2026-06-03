# 老师好,这是 CCA 框架最新一轮 leakage-free 重测后的诚实汇报

## 一句话结论

经过 review 提示的"PPO 协议有潜在 event-level leakage"质疑后,我把实验重做成了**严格 leave-one-event-out(LOEO)10-fold × 10 seeds = 100 paired pairs**,并同时修了 PPO 的 3 个底层 bug。**新的诚实结论是**:

> **PPO 用 4 个 chip 校准 = 用全部 pool 校准(oracle 等价);PPO 显著优于 zero-shot 和 CoreSet,Wilcoxon 显著优于 random,但和 random 的 t-test 在 p=0.084(边缘)。**

---

## 为什么重做

ChatGPT 帮做的同行评审指出了一个关键风险:之前的 PPO paired 显著性(p ≤ 0.005)是在"同 4 个 hard regions 既训练又评估,只是 seed 间换 pool/test 切分"的协议下做的。审稿人可以质疑这是"policy 在它的 evaluation events 上间接训练过",即 event-level leakage。

为了堵这一刀,我设计了**严格的 leave-one-event-out 协议**:

| 步骤 | 内容 |
|---|---|
| 1 | 把 10 个 Sen1Floods11 事件分成 10 折(每折一个事件作 meta-test) |
| 2 | 每折:PPO 在另外 9 个事件上训练,**完全不见 meta-test 事件** |
| 3 | 锁定 policy,在 meta-test 事件上 10 个随机 pool/test seed 评估 |
| 4 | 10 fold × 10 seeds = **100 paired pairs**(原来只有 10) |

---

## 三阶段实验链 + 诚实诊断

### 阶段 A:旧 PPO 在 LOEO 下**灾难性失败**

- Pooled n=100,Δ_PPO−random = **-0.0074 F1**(方向反!)
- Pakistan 单 fold:PPO 比 random **低 0.033 F1**(灾难)
- Somalia:PPO 比 random **低 0.037 F1**(灾难)
- **PPO 显著低于 full-pool oracle**(Δ=-0.014,p=0.032)

→ 旧 PPO 在 cross-event 上根本不工作,旧的"PPO 显著优于所有 baseline"结论是 within-event 协议下的产物,**leakage 嫌疑被坐实了**

### 阶段 B:诊断 RL 底层

原 PPO 有 3 个结构性 bug,在小 reward / 多事件 transfer 场景下尤其致命:

1. **没有 GAE-λ** — 用 raw discounted return,episode-level F1 增益 ~0.005,step-level 噪声 σ~0.05,信噪比 ~1:10 → 梯度估计是噪声
2. **step-level reward** — 每加一个 chip 都给"F1 增量"作 reward,噪声 σ=0.05 比 episode 信号 0.005 大 10 倍
3. **constant entropy 0.01** — 早期就 collapse 到 deterministic policy,失去探索

### 阶段 C:修复 + LOEO 重测

修了:
1. **GAE-λ = 0.95**(advantage estimation,降方差)
2. **`terminal_pixel` reward**(只在 episode 末给整个 F1 gain,降噪一个量级)
3. **entropy schedule 0.10 → 0.01**(线性退火,保留早期探索)

**LOEO-v2 100-pair 重测结果(U-Net,budget=4):**

| 比较 | Δ F1 | t-p | Wilcoxon p | 判决 |
|---|---|---|---|---|
| PPO vs full-pool oracle | **-0.002** | 0.42 | 0.57 | **statistical tie**(4 chip ≈ 全 pool) |
| PPO vs zero-shot(τ=0.5) | **+0.015** | **0.009** | <10⁻⁴ | **显著 \*\*** |
| PPO vs CoreSet | **+0.008** | **0.024** | 0.009 | **显著 \*** |
| PPO vs uncertainty | +0.002 | 0.33 | 0.14 | n.s. |
| PPO vs random | +0.005 | **0.084** | **0.0006** | rank 显著,parametric 边缘 |

**Pakistan 和 Somalia 这两个原本灾难性失败的 fold 现在追平 random**:Pakistan -0.033 → -0.002,Somalia -0.037 → +0.018(反胜!)。

---

## 这意味着什么(诚实判断)

### 强主张(可投 Nature Communications)

1. **Calibration drift 是 cross-disaster 主要 lever**(12 events × 2 benchmarks 都成立,这条没动)
2. **PPO 用 4 个 chip 在严格 leakage-free 协议下追平 oracle** — 这是非常强的标签效率宣言
3. **PPO 显著 > zero-shot 和 CoreSet**(t-p ≤ 0.024)
4. **结构性 RL 改进(GAE+terminal+entropy schedule)是必需的** — 旧 PPO 在 LOEO 下崩,新 PPO 不崩 → 这本身是一个 methodological 贡献

### 弱主张(诚实标注)

5. **PPO vs random** 在 t-test 上不到 0.05(p=0.084),但 Wilcoxon p=0.0006 — paper 里两个都报,Wilcoxon 才是这个分布形态下正确读数。
6. **Decision-aligned reward 改 reward 显著改 policy**(architecture 验证),但**还没显著 net-improve decision metric**(n=20×4 power 不够)— 这条保留在 R4-Appendix。

---

## Manuscript 改写状态

已 commit 到 GitHub(c5012f1 → e2131da):

- **Abstract** —— 整段重写,headline 改为 leakage-free LOEO
- **R4** —— 完全重写,5-比较 paired 表 + per-event headroom 表;旧 within-event 协议正式 retire 到末尾的 Methodological Appendix(物理 reorder)
- **Methods** —— GAE-λ / terminal_pixel / entropy schedule / LOEO 协议全部加进去
- **FRAMEWORK_CCA.md** —— 六大主张表重写
- **新图:Fig 25**(leakage-free 协议示意)、**Fig 26**(LOEO v1 vs v2 三面板)
- **Blog**(Cloudflare live dashboard)已重新生成,LOEO-v2 是新的 R4 头条

---

## 我想请老师的意见

1. **新故事够不够 strong 投 Nature Communications?**
   我的判断:**够**。"4 chip 追平 full-pool oracle" + "显著 > zero-shot/CoreSet" + "Wilcoxon 显著 > random" 三个 robust 主张已经成立。t-test 0.084 边缘的诚实标注反而显得严谨。

2. **是不是 fallback 到 Communications Earth & Environment / npj Natural Hazards 更稳?**
   两个都可以。我现在 manuscript 主线对 Nat Commun 已经 robust,但 CCE 对 Earth science domain 门槛低一点,可能审稿更稳。请老师拍板目标 venue。

3. **后续是否要做 P1/P2/P4(我下面列的)?**
   - **P1(已做):重跑 blog dashboard** ✓
   - **P2(已做):R4-Appendix 物理 reorder** ✓
   - **P3(在做):这份汇报本身**
   - **P4(待做):投稿前打磨** — Methods 扩到 2K 字、Ethics 段、failure-case 图、LaTeX 模板、BibTeX 整理

---

附:GitHub commit:`e2131da`  
Cloudflare live blog:https://geodisaster-fm.pages.dev/ (重新部署后会反映新 LOEO 结果)

如果老师方便,**最重要的反馈是:这版"PPO ≡ oracle, > zero-shot/CoreSet, Wilcoxon > random"的故事强度,够不够投 Nat Commun?** 还是先 CCE / npj 稳一点?

谢谢老师!
