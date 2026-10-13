# Task 1 嵌入消融 — 结果汇总（A0–A5）

> 验证集 AUC 来自 AutoDL 训练；嵌入文件见 `data/ablation_embeddings/`。

## 表 1：主结果（按 Best AUC 降序）

| 排名 | ID | 配置 | 官方 | RQ | 行为SVD | MM均值 | CAGF | Best AUC | Δ vs A0 | 相对提升 | Epoch |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **A1** | 官方+RQ | ✓ | ✓ | — | — | — | **0.895484** | **+0.028603** | **+3.30%** | 12 |
| 2 | **A5** | 置信度门控融合 | ✓ | ✓ | ✓ | ✓ | ✓ | **0.891218** | **+0.024337** | **+2.81%** | 14 |
| 3 | A3 | 官方+MM均值 | ✓ | — | — | ✓ | — | 0.890280 | +0.023399 | +2.70% | 11 |
| 4 | A4 | 固定四源融合 | ✓ | ✓ | ✓ | ✓ | — | 0.889229 | +0.022348 | +2.58% | 16 |
| 5 | A2 | 官方+行为SVD | ✓ | — | ✓ | — | — | 0.870555 | +0.003674 | +0.42% | 14 |
| 6 | A0 | 官方基线 | ✓ | — | — | — | — | 0.866881 | — | — | 14 |

**基线 A0 = 0.866881**

## 嵌入文件

| 实验 | 路径 |
|------|------|
| A0 | `data/ablation_embeddings/A0_baseline_item_info.parquet` |
| A1 | `data/ablation_embeddings/A1_rq_item_info.parquet` |
| A2 | `data/ablation_embeddings/A2_svd_item_info.parquet` |
| A3 | `data/ablation_embeddings/A3_mm_item_info.parquet` |
| A4 | `data/ablation_embeddings/A4_full_item_info.parquet` |
| A5 | `data/MicroLens_1M_x1/ablation/item_info_A5.parquet` |

## 结论

- **最优单模块：A1（官方+RQ）**，AUC **0.895484**
- CAGF A5 达到 **0.891218**，比固定融合 A4 高 **0.001989**，但仍比 A1 低 **0.004266**
- A5 的单种子结果尚不能视为统计显著提升；Task 1 提交仍可优先用 A1
- 行为 SVD 单独增益有限（+0.37 pp）

A5 的公开证据包括 `results/cagf_A5_downstream.csv`、`results/cagf_A5_summary.json` 与 `evidence/CAGF_DIAGNOSTICS.md`。完整日志和 checkpoint 因体积原因不进入公开仓库。最优 checkpoint 的 logloss 为 2.254988，表明排序 AUC 改善不等于概率校准良好。

CSV 明细：`results/ablation_summary.csv`
