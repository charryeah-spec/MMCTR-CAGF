# 方案设计说明

本文档记录本项目的建模思路与模块划分，便于复现与答辩说明。

## 1. 总体思路：级联而非联合

官方赛题将这一设计与工业在线推理的低延迟需求直接关联：Task 1&2 要求嵌入模型与 CTR 模型**分开训练**，不允许联合端到端优化。物品嵌入离线生成，精排模型独立训练和更新，因此本项目把问题定义为“表示—排序器接口”研究。

```
行为序列 + 官方嵌入  →  [离线] 混合嵌入生成  →  item_info_behavior_fusion.parquet
                                                      ↓
用户点击日志 + 物品特征  →  [在线] DIN / Transformer_DCN 训练  →  pCTR 预测
```

## 2. Task 1：混合物品嵌入

在固定 DIN 评估器下，优化 128 维 `item_emb_d128`。融合四路信号：

| 来源 | 方法 | 作用 |
|------|------|------|
| 官方 `item_info` | 基线嵌入 | 保留 BERT+CLIP 语义 |
| RQ 残差量化 | `residual_quantizer.py` | 嵌入去噪与流形平滑 |
| 行为共现 | `build_behavior_emb.py` | 从用户序列挖掘协同信号 |
| 多版本 MM | `item_emb.parquet` 均值 | 降低单一路径方差 |

默认融合权重（可在 `build_hybrid_emb.py` 调整）：

- 官方 0.40 · RQ 0.20 · 行为 0.25 · MM 集成 0.15

融合后 L2 归一化，写回 `item_info_behavior_fusion.parquet`，**不修改 DIN 源码**。

### 行为共现图

- 数据：`item_seq.parquet`（约 600 万条用户序列）
- 滑窗 w=5，距离衰减权重 1/d
- 稀疏矩阵 + TruncatedSVD → 128 维
- 边数上限 1500 万，防止内存溢出

### RQ 残差量化

- 3 层 K-Means，每层 5 个码本
- 对官方嵌入做多层残差量化后重建
- 全程离线，与 CTR 训练解耦

## 3. Task 2：Transformer_DCN

Task 2 使用官方 `item_info.parquet`，优化 CTR 结构。本项目复现并集成公开第一名 Task 2 方案的 **Transformer + DCN v2** 架构：

1. **Target-Aware Transformer**：序列每位置拼接候选物品嵌入，捕获序列-候选交互
2. **DCN v2 + 并行 DNN**：显式特征交叉 + 隐式非线性
3. **MLP 输出**：二分类 pCTR

实现见 `src/Transformer_DCN.py`，超参见 `config/Transformer_DCN_microlens_mmctr_tuner_config_01.yaml`。

## 4. 合规设计

`scripts/check_compliance.py` 自动校验：

- Task 1：DIN 未改；tuner 仅 5 个键；使用融合嵌入
- Task 2：官方 item_info；特征列不变；非 DIN 模型
- Task 1&2：融合嵌入存在；嵌入与 CTR 分阶段训练

完整的官方规则—代码映射见 `docs/OFFICIAL_TASKS_AND_COMPLIANCE.md`。

## 5. 模块归属

| 模块 | 性质 |
|------|------|
| `src/embedding/*` | 本项目设计与实现 |
| `scripts/*`（除官方入口外） | 本项目编排与工具 |
| `src/DIN.py`、`mmctr_dataloader.py` | 官方 FuxiCTR 基线 |
| `src/Transformer_DCN.py` | 参考开源方案复现，见 REFERENCES.md |
| `run_*.py`、`prediction.py` | FuxiCTR 官方训练入口 |

## 6. 消融实验

Task 1 嵌入消融由 `scripts/run_ablation.py` 驱动，预设见 `src/embedding/ablation_presets.py`：

- **A0**：官方 `item_info.parquet`
- **A1–A3**：官方 + 单组件，等权 0.5/0.5
- **A4**：四源融合，权重 0.40 / 0.20 / 0.25 / 0.15
- **A5**：CAGF 逐物品置信度门控，使用留一源自监督目标

DIN 超参在消融中固定，仅替换 `item_info` 路径，保证单因素对照。

CAGF 的置信度特征、目标函数、合规性与验收标准见 `docs/CAGF.md`。
