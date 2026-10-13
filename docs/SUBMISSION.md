# 作业提交说明

## 建议纳入提交包的内容

| 类别 | 路径 |
|------|------|
| 源代码 | `src/`、`scripts/`、`config/`、`run_*.py`、`prediction.py` |
| 文档 | `docs/`、`README.md` |
| 实验结果 | `results/ablation_summary.csv` |
| 赛题预测 | 由 `prediction.py` 生成；公开仓库不包含预测数据 |
| 课程报告 / PPT | 自行附上 |

## 可不打包（体积大）

- `data/`（约 1.3 GB）— 注明下载地址即可  
- `checkpoints/`（约 1 GB）— 可按 README 复现  
- `embeddings/` — 可由脚本重新生成  

## 核心结论

| 指标 | AUC |
|------|-----|
| 官方基线 | 0.8655 |
| 消融 A1（官方+RQ） | 0.895484 |
| 级联验证峰值 | 0.972976 |
| Codabench Task 1&2 | 0.9817（公开截图显示值） |

## 复现

见 [`WORKFLOW.md`](WORKFLOW.md) 与根目录 `README.md`。
