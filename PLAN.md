# GeoDisaster-FM 实施计划

> 依据 `Nature_disaster_GeoFM_proposal.docx`（方向 A: GeoDisaster-FM）与 `Nature_GeoFM_Disaster_Data_Requirements.docx` 拆解。
> 技术栈：Python + PyTorch + Google Earth Engine。

## 当前状态

- 项目目录：`/data/qbao775/nature_geofm_disaster`
- 已有：两份 docx 提案
- 缺失：代码骨架、数据、模型权重、GEE 凭证
- 团队：当前是单人 + AI 协作起步阶段

## 总体路线（与文档 8 周计划对齐）

| 优先级 | 阶段 | 对应文档 | 文档周次 |
|--------|------|----------|----------|
| P0 | 仓库骨架 + 灾害事件清单 | 提案 §12 周 1 | W1 |
| P1 | 数据管线（GEE + 标签 + 辅助数据） | 数据需求 §2,§3,§8 / 提案 §7 | W2 |
| P2 | 数据集类 + 训练评测框架 | 提案 §8 | W3 |
| P3 | 模型库（监督 + Vision FM + RS FM + AlphaEarth 主方法） | 提案 §7 表 3 | W4–5 |
| P4 | 实验驱动（few-shot / 跨域 / 跨事件 / 跨灾种 / 跨年份） | 提案 §8 / 数据需求 §5 | W6 |
| P5 | 决策指标（建筑 / 道路 / 人口 / 设施 / 孤立社区） | 提案 §3 任务 3, §8 / 数据需求 §3.4 | W7 |
| P6 | 论文图表 + 复现 manifest | 提案 §9 表 4 | W8 |

## 详细任务清单（按优先级）

### P0 — 仓库骨架 ⭐ 必须最先做

- **P0.1 Repo skeleton**：`pyproject.toml`、`requirements.txt`、目录结构、`.gitignore`、`configs/` 默认 YAML。
- **P0.2 Disaster event catalog**：覆盖日本近年代表性多灾种事件的 manifest（CSV+YAML），字段对齐数据需求 §6 表格。

### P1 — 数据管线

GEE 下载器（输出 GeoTIFF 或 COG，按事件 AOI + 时间窗）：

- **P1.1 AlphaEarth annual embeddings**（10 m，64 维），主表征。
- **P1.2 Sentinel-1 SAR**（VV/VH，灾前灾后对），洪水主信号。
- **P1.3 Sentinel-2 + DEM/slope/curvature/hydrology**，光学 + 地形先验。
- **P1.4 OSM + WorldPop + 关键设施**（道路、建筑、人口、医院、避难所、学校）。
- **P1.5 全球公开训练集**：xBD、OpenEarthMap、Sen1Floods11、（可选）GlobalRoadNet、WHU Building、LoveDA。
- **P1.6 日本官方标签摄入**：GSI 浸水范围、JAXA 滑坡 inventory、MLIT 通行止め、自治体灾害调查 → polygon→raster。

### P2 — 数据集与训练评测框架

- **P2.1 Patch tiling + multi-source align**：按事件 AOI 切 patch，做地理配准、缺失值掩码、灾前/灾后对齐。
- **P2.2 DataModule + train/eval CLI**：基于 pytorch-lightning，配置驱动（Hydra/YAML）。
- **P2.3 Metrics**：IoU、F1、AUPRC、Brier、calibration、决策指标接入。

### P3 — 模型库

- **P3.1 传统监督**：U-Net、DeepLabV3+、SegFormer。
- **P3.2 通用视觉 FM**：DINOv2、SAM/SAM2（frozen backbone + 轻量 head）。
- **P3.3 遥感 FM**：SatMAE、Prithvi、RemoteCLIP、CrossEarth。
- **P3.4 AlphaEarth 主方法**：AE embedding + LR / RF / XGBoost / MLP / lightweight seg head。
- **P3.5 多模态融合**：AE + SAR + DEM + 道路/建筑/人口先验。

### P4 — 实验驱动

- **P4.1 Few-shot sweep**：0.1% / 1% / 5% / 10% / 50% / 100% 标签量曲线。
- **P4.2 Cross-region**：日本某些都道府县/流域训练 → 未见区域测试。
- **P4.3 Cross-event**：一次灾害事件训练 → 另一次事件测试。
- **P4.4 Cross-hazard**：洪水 → 滑坡 / 台风影响测试。
- **P4.5 Global-to-Japan transfer**：xBD/OpenEarthMap 训练 → 日本测试。
- **P4.6 Temporal transfer**：灾前年表征训练 → 灾后年事件测试。
- **P4.7 Ablation**：AE only / SAR only / DEM only / AE+SAR / AE+SAR+DEM / +道路+建筑+人口。

### P5 — 决策指标（从像素到应急）

- **P5.1 暴露度**：受影响建筑数、受影响道路长度、受影响人口。
- **P5.2 关键设施暴露**：医院、避难所、学校、变电站。
- **P5.3 道路可达性 + 孤立社区**：道路图 + 中断概率 → 图连通性 + Dijkstra 替代路径 + 救援优先级。

### P6 — 图表与复现

- **P6.1 Fig 1 范式对比**、**Fig 2 日本多灾种地图**、**Fig 3 few-shot 曲线**、**Fig 4 跨域矩阵**、**Fig 5 决策影响图**。
- **P6.2 Extended Data**：消融、失败模式、计算成本、灾种细分。
- **P6.3 复现 manifest**：seed、版本、checkpoint、数据 hash。

## 与方向对应关系

| 路线 | 任务覆盖 |
|------|----------|
| 方向 A（最推荐） | P0–P6 全覆盖 |
| 方向 B（benchmark） | P2、P3、P4、P6 |
| 方向 C（few-shot） | P3、P4.1、P6 |
| 方向 D（滑坡） | P1（DEM/坡度/降雨）、P3、P4.2、P4.4 |
| 方向 E（洪水恢复） | P1.2、P3.5、P4.6、P5 |
| 方向 F（道路可达性） | P5.3 |
| 方向 G（AlphaEarth 边界） | P4.7 ablation |
| 方向 H（LLM agent） | 后续扩展，不进入第一篇 |
| 方向 I（基础设施韧性） | P5 扩展 |
| 方向 J（Review） | 不依赖代码 |

## 三个最小数据包（来自数据需求 §8）

- **Package A 洪水**：3–5 个事件，Sentinel-1 灾前/灾后，官方淹没范围，DEM/河流/建筑/道路。
- **Package B 滑坡**：3+ 强降雨/地震诱发事件，1,000+ polygon，灾后影像，DEM/坡度/降雨/地质。
- **Package C 道路/建筑**：2–3 个事件，高分灾前/灾后，建筑 footprint 或道路网。

实现层面三个包共用同一 schema 和同一 pipeline；切换只是 catalog 子集 + 数据源组合不同。

## 风险与对应（来自提案 §11）

| 风险 | 代码层规避 |
|------|------------|
| AlphaEarth 10 m annual 对小时级/单体建筑不适用 | 主任务限定洪水/滑坡/暴露度；细粒度交给 SAR/高分 |
| 标签质量参差 | `data/labels/` 里强制带 `confidence` 字段 + 多源交叉验证脚本 |
| 模型比较易变 benchmark | 把"标注效率规律 / domain gap 结构"作为 first-class 产物输出 |
| 区域性太强 | P4.5 global-to-Japan 必做 |

---

实现顺序：从 **P0.1** 开始，每完成一个 P 级 checkpoint 我会暂停汇报进度，等你确认或调整后再继续。

## 先在公开数据集上做实验

在等 GSI 浸水范围 polygon 期间，下面两步都可以**立刻在你本机上跑**，验证 pipeline。

### A. 合成数据 smoke test（30 秒，无下载）

```bash
geodisaster smoke
```

会做：
1. 在 `/tmp/geodisaster_smoke_xxx/` 下生成 2 个合成 event × 8 个 patch（AlphaEarth-shape, S1-shape, DEM-shape, S2-shape, 带圆形 flood 标签）
2. 对 3 个核心模型族（`alphaearth_head` / `multi_modal_fusion` / `smp_unet`）跑 forward pass 形状校验
3. 测 Welford normalize-stats（与 numpy 全量聚合对比到 1e-9）
4. 验证 BinaryConfusion / AUPRC / ECE / Focal BCE loss 的可微性
5. 2-epoch toy Lightning 训练，跑通 fit + test 闭环

任何一步失败就报告失败的 family/step。**这是验收新代码的第一步。**

### B. Sen1Floods11 真公开数据训练（一两小时，需要 ~5 GB 磁盘）

[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)：446 个全球 11 个洪水事件的人工标注 Sentinel-1 chips，无注册门槛。

```bash
# 1. 下载 Sen1Floods11 v1.1（约 5 GB）
mkdir -p data/external/sen1floods11
gsutil -m cp -r gs://sen1floods11/v1.1 data/external/sen1floods11/

# 2. 转成我们的 patch 格式 + 自动生成 catalog
geodisaster ingest-sen1floods11 \
    --root data/external/sen1floods11 \
    --out-patches data/processed/patches \
    --out-catalog data/catalog/sen1floods11_events.yaml
# 可选：--with-alphaearth 同时为每个 chip 拉 AlphaEarth 64-d embedding（要 GEE auth，慢）

# 3. 算 normalize stats（仅训练 region）
geodisaster compute-stats \
    --catalog data/catalog/sen1floods11_events.yaml \
    --event sen1floods11_Bolivia --event sen1floods11_Ghana \
    --event sen1floods11_India --event sen1floods11_Mekong \
    --event sen1floods11_Nigeria --event sen1floods11_Paraguay \
    --event sen1floods11_Somalia --event sen1floods11_Sri-Lanka \
    --source sentinel1 \
    --out data/processed/norm_stats_sen1floods11.yaml

# 4. 跨区域训练：8 个 region 训练，Pakistan 验，Spain + USA 测
geodisaster train \
    --catalog data/catalog/sen1floods11_events.yaml \
    --model-config configs/model/unet.yaml \
    --train-events sen1floods11_Bolivia --train-events sen1floods11_Ghana \
    --train-events sen1floods11_India --train-events sen1floods11_Mekong \
    --train-events sen1floods11_Nigeria --train-events sen1floods11_Paraguay \
    --train-events sen1floods11_Somalia --train-events sen1floods11_Sri-Lanka \
    --val-events sen1floods11_Pakistan \
    --test-events sen1floods11_Spain --test-events sen1floods11_USA \
    --stats data/processed/norm_stats_sen1floods11.yaml

# 5. Few-shot sweep
geodisaster run-few-shot \
    --catalog data/catalog/sen1floods11_events.yaml \
    --model-config configs/model/unet.yaml \
    --stats data/processed/norm_stats_sen1floods11.yaml

# 6. 出图
geodisaster make-figures \
    --catalog data/catalog/sen1floods11_events.yaml \
    --few-shot-csv "U-Net=outputs/few_shot_unet/few_shot_results.csv" \
    --reproducibility
```

这条路径**完全公开 + 无 API 门槛**，跑下来直接证明：
- pipeline 真的能从原始公开数据走到 Nature Fig 3 风格的 few-shot 曲线
- 模型库（至少 U-Net + AlphaEarth 主方法）能在真实 SAR 数据上收敛
- cross-region 协议有可比数字
- 后面接到日本数据上只是换 catalog + 加 AlphaEarth/DEM 等 source

## 端到端运行顺序

```bash
# 0. 环境
bash scripts/setup_env.sh
earthengine authenticate && earthengine set_project <gcp-project>

# 1. 数据
geodisaster download-gee  --data-config configs/data/japan_flood.yaml
geodisaster fetch-osm     --data-config configs/data/japan_flood.yaml
# 把 GSI 浸水 polygon 放到 data/external/gsi/<event_id>/ 后：
geodisaster ingest-labels --raw-root data/external/gsi
geodisaster tile-dataset

# 2. 归一化统计（必跑，否则不同 source 量级差距会主导 loss）
geodisaster compute-stats \
    --event jp_typhoon_hagibis_2019 --event jp_west_japan_floods_2018 \
    --source alphaearth --source sentinel1 --source dem \
    --out data/processed/norm_stats.yaml

# 3. 训练 / 实验
geodisaster train          --model-config configs/model/alphaearth_mlp.yaml
geodisaster run-few-shot   --model-config configs/model/alphaearth_mlp.yaml
geodisaster run-cross-domain --model-config configs/model/alphaearth_mlp.yaml

# 4. 应急决策指标 + 图
geodisaster decision-metrics --event jp_kyushu_floods_2020 \
    --impact-mask outputs/.../prediction.tif \
    --buildings data/raw/jp_kyushu_floods_2020/jp_kyushu_floods_2020_buildings.gpkg \
    --roads data/raw/jp_kyushu_floods_2020/jp_kyushu_floods_2020_roads.gpkg \
    --population data/raw/jp_kyushu_floods_2020/jp_kyushu_floods_2020_worldpop_2019.tif
geodisaster make-figures --reproducibility
```

注意：
- `download-gee` 大 AOI 会自动按 GEE 50 MB / 32k 像素限制切分下载、再 mosaic（AlphaEarth 64 维与 Sentinel-1 2 维各自用合适的 chunk 大小，自动从 `n_bands_hint` 推算）。
- `compute-stats` 必须只跑训练事件，否则会泄漏验证/测试分布。
- AlphaEarth 是 annual embedding：灾前年应取 `event.year - 1` 才真正不含灾后信号；当前 catalog 的 `pre_window` 与事件同年，跨年实验前先在 [alphaearth.py](geodisaster/data/gee/alphaearth.py) 把 `_pick_year` 调成 `event.year - 1`。
