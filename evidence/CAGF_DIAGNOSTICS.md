# CAGF full-data diagnostic evidence

Generated from `results/cagf_A5_summary.json` and `results/cagf_A5_item_weights.parquet` with seed `20242025`.

| Check | Retained result |
|---|---:|
| Challenge items represented | 91,718 |
| Behavior source available | 91,297 |
| MM ensemble available | 91,717 |
| Distinct four-decimal weight vectors | 91,716 |
| Mean gate entropy | 1.2306 |
| Non-finite output values | 0 |
| Missing-source nonzero weights | 0 |

| Source | Mean weight | Across-item std | P05 | P95 |
|---|---:|---:|---:|---:|
| Official | 0.4294 | 0.0711 | 0.3223 | 0.5561 |
| RQ | 0.2276 | 0.0561 | 0.1418 | 0.3251 |
| Behavior | 0.0788 | 0.0200 | 0.0541 | 0.1159 |
| MM ensemble | 0.2643 | 0.0543 | 0.1745 | 0.3545 |

These diagnostics establish adaptivity, numerical stability, and correct masking. The subsequent fixed-DIN GPU run reached 0.891218 validation AUC, compared with 0.889229 for A4 and 0.895484 for A1. The A5–A4 difference is based on one seed and is not a statistical-significance claim. Compact evidence is retained in `results/cagf_A5_downstream.csv` and `results/cagf_A5_summary.json`; full logs and checkpoints remain outside the public repository.
