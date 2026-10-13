# Official Task Mapping and Compliance

This project is a **course-based reproduction and extension study** built on the official WWW 2025 EReL@MIR Multimodal CTR Prediction Challenge protocol. It does not claim an official award or final competition rank.

## Official benchmark context

The organizers motivate the benchmark by the low-latency requirements of industrial online recommendation. They separate expensive offline item representation from online CTR inference and use MicroLens-1M, which contains approximately **1 million users and 91 thousand items** with content features such as titles and cover images. Training data are public, test labels are private, and the official evaluation metric is ROC-AUC.

Sources:

- [Official challenge description](https://erel-mir.github.io/challenge/mmctr-track2/)
- [Official FuxiCTR baseline](https://github.com/reczoo/WWW2025_MMCTR_Challenge)
- [Codabench benchmark](https://www.codabench.org/competitions/5372/)

## Three-task protocol

| Track | Official variable | Official invariant | This repository |
|---|---|---|---|
| Task 1: item embedding | 128-D multimodal item representation | Fixed downstream CTR model; only five specified DIN hyperparameters may be tuned; no external data | RQ reconstruction, behavior-SVD, multimodal mean, fixed fusion, and CAGF in the A0–A5 protocol |
| Task 2: CTR prediction | Neural CTR architecture and hyperparameters | Organizer-provided embeddings and feature schema; one end-to-end neural model; no ensemble or tree model | Reproduction and integration of a target-aware Transformer + DCN v2 + DNN |
| Task 1&2: cascade | Both offline representation and CTR model | No additional features; two separate, disjoint training stages rather than joint training | Offline embedding artifact followed by an independently trained Transformer–DCN ranker |

All three tracks output pCTR predictions for the private test set. The retained submission artifact used three score columns—`Task1`, `Task2`, and `Task1&2`—and 379,142 sequential IDs. Prediction rows are excluded from this public repository; the generator and compliance checks remain available.

## Rule-to-code traceability

| Rule | Enforcement / evidence |
|---|---|
| Task 1 output is 128-dimensional | `build_hybrid_emb.py` rejects any non-128-D source |
| Learned fusion stays separate from DIN | CAGF trains with label-free leave-one-source-out reconstruction and writes a Parquet artifact before DIN |
| Task 1 uses fixed DIN | `check_compliance.py::check_task1_config` verifies the base model |
| Only five Task 1 hyperparameters are tunable | `TASK1_ALLOWED_TUNER` and config validation |
| Task 2 uses official embeddings | `check_task2_config` requires the official `item_info.parquet` |
| Task 2 feature inputs remain fixed | Exact feature-column set comparison |
| Task 2 is one neural model | One `Transformer_DCN` experiment graph; no prediction ensemble path |
| Task 1&2 adds no features | Exact feature-column set comparison |
| Task 1&2 stages are separate | Offline embedding builder writes a Parquet artifact before ranker training |
| Submission is valid pCTR | Row IDs, column names, finite values, `[0,1]` bounds, and ZIP CRC checks |

## Reproducibility levels

- **Lightweight verification:** `python3 scripts/project_healthcheck.py` checks source syntax, recorded ablation arithmetic, CAGF diagnostics, repository hygiene, credential patterns, and attribution without GPU dependencies.
- **Rule verification:** `python scripts/check_compliance.py --task all` checks data, feature, model, and track constraints in the configured ML environment.
- **Experiment reproduction:** `bash scripts/run_ablation.sh 0` rebuilds the Task 1 representations and trains the controlled A0–A5 suite.

The repository does not claim that the current lightweight check reproduces GPU training; it verifies the public artifact package and separates that claim from full experiment reproduction.

## Benchmark context—not a ranking claim

The official FuxiCTR baseline reports a best validation AUC of **0.8655**. The published first-place Task 2 report records **0.9776 validation AUC** for its full model and **0.9839 leaderboard AUC**. This project records **0.972976 validation AUC** and **0.9817 Codabench-displayed test AUC** for its cascaded Task 1&2 submission.

These figures come from different tracks and experiment artifacts; they provide context only and must not be used to infer a competition rank. The academically defensible contribution is the controlled embedding-side extension, negative-result analysis, and reproducible three-track pipeline.
