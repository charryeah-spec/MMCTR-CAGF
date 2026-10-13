# 实验流程

本文档描述从数据检查到提交的完整操作步骤。

## 前置条件

```bash
conda activate mmctr
cd <项目根目录>
python scripts/check_data.py
```

## 赛道一：Task 1（嵌入优化 + DIN）

```bash
# 构建混合嵌入
bash scripts/build_hybrid_embedding.sh

# 训练 DIN（固定 CTR 结构）
bash scripts/run_task1_behavior.sh 0
tail -f logs/task1_din.log

# 预测
python scripts/find_best_expid.py DIN_behavior_fusion_task1_config.csv
bash scripts/predict_task1_din.sh <BEST_EXPID> 0

# 合规检查
python scripts/check_compliance.py --task task1 --submission submission/prediction.csv
```

## 赛道二：Task 2（CTR 优化 + 官方嵌入）

```bash
bash scripts/run_task2_tdcn.sh 0
tail -f logs/task2_tdcn.log

python scripts/find_best_expid.py Transformer_DCN_microlens_mmctr_tuner_config_01.csv
bash scripts/predict_task2_tdcn.sh <BEST_EXPID> 0

python scripts/check_compliance.py --task task2 --submission submission/prediction.csv
```

可选超参网格搜索：

```bash
bash scripts/run_task2_sweep.sh 0
```

## 赛道三：Task 1&2（级联，推荐）

```bash
bash scripts/build_hybrid_embedding.sh
bash scripts/run_task12_behavior_tdcn.sh 0
tail -f logs/task12_behavior_tdcn.log

python scripts/find_best_expid.py Transformer_DCN_behavior_fusion_config.csv
bash scripts/predict_task12.sh <BEST_EXPID> 0

python scripts/check_compliance.py --task task12 --submission submission/prediction.csv
```

提交文件：`submission/<experiment_id>.zip`

## 常用工具

| 命令 | 说明 |
|------|------|
| `python scripts/check_data.py` | 检查数据文件是否齐全 |
| `python scripts/find_best_expid.py <csv>` | 从训练日志 CSV 提取最优实验 ID |
| `python scripts/check_compliance.py --task all` | 全赛道合规自检 |
| `bash scripts/run_ablation.sh 0` | **Task1 嵌入消融 A0–A5**（全流程） |
| `bash scripts/run_ablation.sh 0 --embed-only` | 仅构建消融嵌入，不训练 |
| `bash scripts/run_cagf_ablation.sh 0` | 聚焦对比 A1 / A4 / A5 |
| `python scripts/summarize_ablation.py` | 从已有结果汇总消融表 |

## Task 1 嵌入消融（A0–A5）

固定 DIN 超参，仅改变物品嵌入来源，用于报告中的单因素对照。

| 实验 | 官方 | RQ | 行为 SVD | 多版本均值 | 目的 |
|------|:----:|:--:|:--------:|:----------:|------|
| A0 | ✓ | — | — | — | 官方基线 |
| A1 | ✓ | ✓ | — | — | 结构平滑 |
| A2 | ✓ | — | ✓ | — | 协同信号 |
| A3 | ✓ | — | — | ✓ | 集成降方差 |
| A4 | ✓ | ✓ | ✓ | ✓ | 完整融合 |
| A5 | ✓ | ✓ | ✓ | ✓ | CAGF 逐物品自适应门控 |

```bash
# 全流程（6 组 DIN，建议 GPU 环境）
bash scripts/run_ablation.sh 0

# 只补跑 A2
bash scripts/run_ablation.sh 0 A2

# 仅生成嵌入文件（不训练）
bash scripts/run_ablation.sh 0 --embed-only

# 聚焦对比：RQ-only vs 固定融合 vs CAGF
bash scripts/run_cagf_ablation.sh 0
```

产物：

- 嵌入：`data/ablation_embeddings/`（报告用命名）或 `data/MicroLens_1M_x1/ablation/`（脚本生成）
- 汇总：`results/ablation_summary.csv`、`results/ablation_summary.md`
- 配置：`config/ablation_A*/`（自动生成）

A5 还会产生逐物品权重 `results/cagf_A5_item_weights.parquet`、训练摘要 `results/cagf_A5_summary.json` 和门控权重 `checkpoints/cagf_A5_gate.pt`。其中大文件由 `.gitignore` 排除。固定 DIN 单种子结果为 **0.891218 AUC**（A4: 0.889229，A1: 0.895484），详见 `results/cagf_A5_downstream.csv` 与 `docs/CAGF.md`。

如果已有最优 checkpoint，可避免重复训练，直接评估：

```bash
python scripts/evaluate_checkpoint.py \
  --config config/ablation_A5 \
  --expid DIN_ablation_A5_001 \
  --gpu 0 \
  --out results/cagf_A5_downstream.csv
```

## 训练参数参考

| 模型 | batch_size | learning_rate | 单 epoch 耗时（4090） |
|------|------------|---------------|----------------------|
| DIN | 8192 | 1e-3 | ~3 min |
| Transformer_DCN | 128 | 5e-4 | ~10 min |

早停：监控验证集 AUC，DIN patience=3，Transformer_DCN patience=5。

## 日志位置

```
logs/build_behavior_emb.log
logs/build_hybrid_emb.log
logs/task1_din.log
logs/task2_tdcn.log
logs/task12_behavior_tdcn.log
```
