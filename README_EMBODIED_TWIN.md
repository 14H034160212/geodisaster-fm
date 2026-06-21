# Open-Data Disaster Digital Twin for Embodied UAV (新方向 / 续作)

> **Open-Data Disaster Digital Twin for Embodied UAV Damage Inspection and Search-and-Rescue**
> 面向多灾种防灾减灾的开源数据驱动灾害数字孪生与无人机具身智能评估框架

**状态:** 构思阶段(白老师 2026-06-21 提出)· 尚无代码
**关系:** [GeoDisaster-FM](README.md) 的续作——把"感知+校准的地基"推到 *action / 具身决策* 层

---

## 一句话

不做"预测灾害本身",而是回答:**灾害发生后,如何用开源真实数据快速搭一个可交互的"灾害数字孪生",
让 UAV/机器人 agent 在里面做搜索、巡检、灾损识别和救援优先级决策。**

---

## 核心思路:真实数据 + 物理模拟,而不是 photorealistic 仿真

> **关键决策(白老师"用真实数据 vs 自己模拟"那一问):**
> 现有具身搜救 benchmark(ESARBench / RescueBench)都是**纯仿真(UE5+AirSim),不用真实灾害数据**——
> 照搬就丢掉了"用我们真实数据、容易发表"的优势,且工程量大、与已有 benchmark 同质化。
> **本方向不去搭 photorealistic 3D 世界**,而是:

1. 把**真实灾害数据**做成数字孪生的**静态真值图层**(损伤/淹没/道路/人口);
2. 用**物理模拟**(GeoClaw / Physics-Informed Video Diffusion)补**动态淹没图层**——这是"模拟用得有道理"的范例;
3. 让 agent 在这个孪生上做**决策/规划级具身任务**(先去哪看、怎么飞避水、找谁、怎么排优先级),
   而不是 photorealistic 视觉导航 → 工程量降到可控,真实数据优势保住。

---

## 数据图层

| 图层 | 来源 | 真实/模拟 |
|---|---|---|
| 建筑损伤(静态真值) | **xBD**(85 万建筑、多灾种)—— GeoDisaster-FM 已在用 | 真实 |
| 洪水淹没(静态真值) | **Sen1Floods11**(10 个真实洪水)—— 已有 | 真实 |
| 野火烧痕 | **HLS Burn-Scars** —— 已有 | 真实 |
| **UAV 低空感知锚点** | **RescueNet**(飓风 Michael 真实 UAV 影像 4494 张+语义分割) | 真实 |
| 洪水/海啸**动态**演进 | **GeoClaw** / **Physics-Informed Video Diffusion** | 物理模拟 |
| 建筑/道路/POI、地形 | OpenStreetMap、DEM | 真实 |

> **RescueNet 是唯一接近"真实具身感知"的数据**,让 agent 不只在卫星图上决策、还有真实无人机低空视角——发表的强锚点。

---

## 具身任务(沿用白老师文档的四个)

| Task | 内容 | 主要指标 |
|---|---|---|
| **T1 Damage Inspection** | 优先巡检高损伤建筑 | 单位时间严重损伤建筑召回率 |
| **T2 Flood-Aware Navigation** | 避开洪水/阻断道路找可通行路线 | 路径长度、危险区进入次数 |
| **T3 Search & Rescue Clue** | 据视觉线索+损伤+道路推理受困者位置 | 搜索成功率、平均发现时间 |
| **T4 Risk-Aware Prioritization** | 据损伤+人口+连通性+水深排救援优先级 | NDCG / Spearman |

---

## Agent 对比(直接复用现有 RL)

- rule(网格扫描 / 最近邻 / 损伤概率排序)
- vision(先识损伤再规划)
- VLM/MLLM agent(图+地图+文字任务)
- **RL/IL agent —— 复用 GeoDisaster-FM 现成的 Layer 3 PPO**(从"主动校准"扩到"具身风险感知优先级/搜索规划")

---

## 与现有项目的衔接(最大卖点)

GeoDisaster-FM 本质已是"真实多灾种数据上,感知 → 神经符号推理 → RL 决策"的闭环 agent;
**Layer 3 PPO 就是一个 agent 在真实灾害数据上做决策**。本方向 = 把它从 chip 选择扩到具身搜索/巡检——
**基础设施已有,不是从零开始**。

---

## 白老师发的参考论文 —— 定位

| 论文 | 定位 | 用法 |
|---|---|---|
| **ESARBench / RescueBench** | 纯仿真具身搜救 benchmark | **任务设计参考**(探索→线索→定位→交接、风险优先级);**不照搬引擎** |
| **Physics-Informed Video Diffusion**(浅水方程约束) | 动态淹没图层的快速物理一致生成器 | **重点深入用**——把"物理模拟"接进数字孪生 |
| **DRAW2ACT**(机械臂轨迹条件视频生成) | 偏机器人操作,离 UAV 搜救较远 | **暂不进主线**(远期可做"生成 UAV demo 视频训 policy") |

---

## 落地建议(务实版)

- **灾种**:先做 **洪水/飓风 + 建筑损伤**(数据最齐,白老师文档的"最稳组合")。
- **环境**:先做轻量级 **2.5D 网格/图 决策环境**(HAZARD challenge 风格),**别一上来搭 UE5**。
- **消融**:有无数字孪生 / 有无洪水图层 / 有无损伤优先级。
- **投稿**:**International Journal of Disaster Risk Reduction(最稳)**;具身味做重可冲 RA-L 或 ICRA/IROS workshop。

---

## 论文骨架

1. Introduction —— 灾害响应从静态制图走向可交互可决策的数字孪生;现有 embodied benchmark 少面向真实灾害
2. Related Work —— digital twins / RS damage assessment / UAV disaster response / embodied SAR benchmarks
3. Open-Data Disaster Digital Twin Construction(xBD / RescueNet / OSM / DEM / GeoClaw 图层)
4. Embodied Disaster Response Tasks(T1–T4)
5. Agent Baselines & Metrics
6. Experiments(agent 对比 + 消融)
7. Discussion(可迁移性、部署限制、数据偏差、多灾种扩展)
8. Conclusion

---

## 源文档

白老师 2026-06-21 提供的材料在 [`20260621_bai/`](20260621_bai/):
中文构思文档、HAZARD Challenge、ESARBench(2605.01371)、RescueBench(2606.01848)、
DRAW2ACT(2512.14217)、Physics-Informed Video Diffusion。
