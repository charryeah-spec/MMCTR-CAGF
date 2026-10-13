# Evidence and Attribution

This page defines which claims are directly verifiable, which are recorded experiment results, and which code came from upstream work.

## Official protocol evidence

| Fact | Primary source |
|---|---|
| Challenge motivation: decouple representation and CTR modeling for low-latency online inference | [Official challenge page](https://erel-mir.github.io/challenge/mmctr-track2/) |
| MicroLens-1M contains about 1M users and 91K items | [Official challenge page](https://erel-mir.github.io/challenge/mmctr-track2/) |
| Task 1 output is a 128-D item embedding evaluated with a fixed CTR model | [Official Task 1 rules](https://erel-mir.github.io/challenge/mmctr-track2/) |
| Task 2 fixes organizer embeddings/features and allows one neural CTR model | [Official Task 2 rules](https://erel-mir.github.io/challenge/mmctr-track2/) |
| Task 1&2 requires two separately trained models and no additional features | [Official cascaded-task rules](https://erel-mir.github.io/challenge/mmctr-track2/) |
| Only five specified Task 1 DIN hyperparameters may be tuned; baseline validation AUC is 0.8655 | [Official FuxiCTR baseline](https://github.com/reczoo/WWW2025_MMCTR_Challenge) |
| Published first-place Task 2 solution reports 0.9839 leaderboard AUC | [Xu et al., 2025](https://arxiv.org/abs/2505.03543) |

## Claim ledger

| Claim | Evidence | Confidence / wording |
|---|---|---|
| A0 validation AUC = 0.866881 | [`../results/ablation_summary.csv`](../results/ablation_summary.csv) | Recorded controlled experiment |
| A1 validation AUC = 0.895484 | Same CSV and [`../results/ablation_summary.md`](../results/ablation_summary.md) | Recorded controlled experiment |
| A1 lift = +0.028603 / +3.30% | Same CSV; recomputed by health check | Directly reproducible from recorded values |
| A5 validation AUC = 0.891218 | [`../results/cagf_A5_downstream.csv`](../results/cagf_A5_downstream.csv) and [`../results/cagf_A5_summary.json`](../results/cagf_A5_summary.json) | Recorded fixed-seed experiment |
| A5−A4 = +0.001989; A5−A1 = -0.004266 | [`../results/ablation_summary.csv`](../results/ablation_summary.csv) | Arithmetic comparison; no significance claim |
| Cascaded validation AUC = 0.972976 | [`SUBMISSION.md`](SUBMISSION.md) | Recorded project result; raw training log is not in the public package |
| Codabench test AUC = 0.9817 displayed | [`../evidence/codabench_result.png`](../evidence/codabench_result.png) | Direct screenshot evidence |

Validation and test AUC values come from different splits and are intentionally not presented as a direct improvement pair.

The published 0.9839 result belongs to the first-place **Task 2** submission. This repository's 0.9817 screenshot records **Task 1&2**. They are not treated as a same-track ranking comparison.

## Ownership boundary

### Designed and implemented for this project

- `src/embedding/build_behavior_emb.py`
- `src/embedding/build_hybrid_emb.py`
- `src/embedding/fuse_item_info.py`
- `src/embedding/ablation_presets.py`
- `src/embedding/confidence_features.py`
- `src/embedding/confidence_gated_fusion.py`
- `scripts/run_ablation.py` and the A0–A5 orchestration
- `scripts/check_compliance.py`
- `scripts/project_healthcheck.py`
- Training/prediction shell orchestration and project documentation

CAGF/A5 is evidenced by its implementation, full-data gate summary, fixed-DIN result CSV, and retained diagnostics. Full logs, checkpoints, and per-item weight files are excluded from the public repository. The single-seed improvement over A4 must not be described as statistically significant, and A1 remains the strongest Task 1 variant.

### Upstream official baseline

- `src/DIN.py`
- `src/mmctr_dataloader.py`
- `run_expid.py`, `run_param_tuner.py`, and the original prediction entry point
- FuxiCTR framework and MicroLens challenge configuration conventions

Source: [FuxiCTR](https://github.com/reczoo/FuxiCTR) and the [official WWW 2025 MM-CTR baseline](https://github.com/reczoo/WWW2025_MMCTR_Challenge).

### Reproduced reference architecture

- `src/Transformer_DCN.py`
- Residual-quantization idea, independently implemented for the offline pipeline

Source: the Apache-2.0 [open WWW 2025 MM-CTR solution](https://github.com/pinskyrobin/WWW2025_MMCTR) and its [published technical report](https://arxiv.org/abs/2505.03543). The repository does not claim the Transformer_DCN architecture as original work.

## Safe public description

> An independent, course-based reproduction and extension of the WWW 2025 MM-CTR benchmark protocol, adding embedding-side interventions, controlled ablation, three-track orchestration, and compliance verification around an attributed reference ranker.

Avoid describing the work as an official award-winning submission, a new Transformer–DCN architecture, or a production deployment.

## Public-package verification

Run from the repository root:

```bash
python3 scripts/project_healthcheck.py \
  --json evidence/healthcheck.json
```

This checks syntax, core files, controlled result arithmetic, CAGF diagnostics, public-package hygiene, common credential patterns, and attribution language. It does not rerun GPU training or validate excluded prediction rows.
