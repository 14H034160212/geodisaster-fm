# 研究构思 / Proposal

## Open-Data Disaster Digital Twin for Embodied UAV Damage Inspection and Search-and-Rescue
### 面向多灾种防灾减灾的开源数据驱动灾害数字孪生与无人机具身智能评估框架

**提出人:** 包启明 · **致:** 白老师 · **日期:** 2026-06-21
**定位:** 现有 GeoDisaster-FM(投 Nature Communications)的续作——把"感知+校准的地基"推到具身决策(action)层

---

### 1. 问题与立意

不做"预测灾害本身",而是回答:**灾后如何用开源真实数据快速搭一个可交互的"灾害数字孪生",
让 UAV/机器人 agent 在其中执行搜索、巡检、灾损识别与救援优先级决策。** 这比单纯做遥感分类更有新意,
也直接对接国际上正在兴起的"具身智能 + 防灾减灾"方向(ESARBench、RescueBench、HAZARD Challenge)。

### 2. 关键技术决策(回应"真实数据 vs 自己模拟")

现有具身搜救 benchmark(ESARBench/RescueBench)**都是纯仿真(UE5+AirSim),不用任何真实灾害数据**。
照搬会丢掉"用真实数据、容易发表"的优势,且工程量大、与已有工作同质化。**本方案不去搭 photorealistic 3D 世界**,而是:

1. 用**真实灾害数据**做数字孪生的**静态真值图层**(建筑损伤 / 淹没 / 道路 / 人口);
2. 用**物理模拟**(GeoClaw + 白老师的 Physics-Informed Video Diffusion)补**动态淹没图层**——"模拟用得有道理"的范例;
3. agent 在孪生上做**决策/规划级具身任务**(先去哪看、怎么飞避水、找谁、如何排序),
   而非 photorealistic 视觉导航 → **工程量可控、真实数据优势保住**。

### 3. 数据图层(全部开源)

| 图层 | 来源 | 真实/模拟 |
|---|---|---|
| 建筑损伤(静态) | **xBD**(85 万建筑、多灾种)| 真实 |
| 洪水淹没(静态) | **Sen1Floods11**(10 个真实洪水)| 真实 |
| 野火烧痕 | **HLS Burn-Scars** | 真实 |
| **UAV 低空感知锚点** | **RescueNet**(飓风 Michael 真实 UAV 4494 张+语义分割)| 真实 |
| 洪水/海啸**动态**演进 | **GeoClaw** / **Physics-Informed Video Diffusion** | 物理模拟 |
| 建筑/道路/POI、地形 | OpenStreetMap、DEM | 真实 |

> xBD / Sen1Floods11 / HLS **已在 GeoDisaster-FM 中就绪**;RescueNet 是唯一接近"真实具身感知"的数据,为发表强锚点。

### 4. 具身任务与 Agent

- **任务:** T1 Damage Inspection · T2 Flood-Aware Navigation · T3 Search-and-Rescue Clue · T4 Risk-Aware Prioritization
- **Agent 对比:** rule / vision / VLM-MLLM / **RL(复用 GeoDisaster-FM 现成 Layer 3 PPO,从"主动校准"扩到"具身优先级/搜索规划")**
- **指标:** 搜索成功率、平均发现时间、覆盖率、危险区进入次数、严重损伤召回、优先级排序 NDCG/Spearman

### 5. 与现有项目的衔接(最大可行性优势)

GeoDisaster-FM 本质已是"真实多灾种数据上,感知→神经符号推理→RL 决策"的闭环;
**Layer 3 PPO 就是一个 agent 在真实灾害数据上做决策**。本方案 = 把它从 chip 选择扩到具身搜索/巡检——
**基础设施已有,不从零开始**(详见 [README_EMBODIED_TWIN.md](README_EMBODIED_TWIN.md) 与衔接图 `outputs/figures/fig_twin_bridge.png`)。

### 6. 预期贡献(三点)

1. **开源数据驱动的多灾种灾害数字孪生构建框架**——统一卫星灾损、UAV 灾后语义、OSM/DEM、水动力模拟图层。
2. **把数字孪生从静态地图扩展到具身智能评估环境**——不止输出 damage/flood map,而是让 agent 在其中搜索、巡检、导航、排序。
3. **面向防灾减灾的 embodied AI 任务与指标体系**——评价的是发现时间、巡检效率、避险能力、高风险目标覆盖率,而非分类精度。

### 7. 时间线与里程碑(约 6 个月)

| 阶段 | 时间 | 里程碑 | 交付 |
|---|---|---|---|
| **M0 立项与数据对齐** | 2026-07(第 1–3 周)| xBD/RescueNet/Sen1Floods11/OSM/DEM 对齐到统一坐标网格;选定先做"洪水/飓风+建筑损伤" | 数据 catalog + 图层规范 |
| **M1 数字孪生 v0** | 2026-07–08 | 轻量级 2.5D 网格/图决策环境(HAZARD 风格);静态真值图层贯通 | 可加载的孪生环境 + 可视化 |
| **M2 动态图层** | 2026-08 | GeoClaw 洪水/海啸淹没接入;探索 PIVD 作为快速代理 | 动态淹没图层 + 物理一致性验证 |
| **M3 任务与 baseline** | 2026-09 | T1–T4 任务定义;rule/vision/VLM 三类 baseline 跑通 | 任务接口 + baseline 结果表 |
| **M4 RL agent(复用 PPO)** | 2026-09–10 | Layer 3 PPO 迁移到具身优先级/搜索;四任务对比 + 消融(有无孪生/洪水层/损伤优先级)| 完整实验结果 |
| **M5 写作与投稿** | 2026-11–12 | 论文成稿 + 图表 + 开源代码 | 投 IJDRR(初稿)|

> 关键风险与缓解:**真实 UAV 灾害数据稀缺** → 以 RescueNet 为锚 + 物理模拟补动态层;**具身工程量** → 不搭 UE5,先做 2.5D 决策环境。

### 8. 投稿目标

- **首选(最稳):** *International Journal of Disaster Risk Reduction*(接受"系统框架+数据实验+应用价值")
- **备选(具身味重时):** *IEEE RA-L* / ICRA · IROS · NeurIPS-CVPR embodied workshop
- 参考对标:NHESS、Remote Sensing、ISPRS Int. J. Geo-Information

---

**源材料:** 白老师 2026-06-21 提供的文档在 [`20260621_bai/`](20260621_bai/)(中文构思、HAZARD Challenge、ESARBench、RescueBench、DRAW2ACT、Physics-Informed Video Diffusion)。
