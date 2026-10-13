# Project Case Study

## 1. Problem

Click-through-rate prediction for short-video recommendation has two coupled components: the quality of the item representation and the ranking model that consumes it. The official WWW 2025 EReL@MIR benchmark separates them to reflect a practical systems constraint: content-heavy representation learning can run offline, while the CTR model must support low-latency online inference.

The protocol uses MicroLens-1M—approximately one million users and 91 thousand items with titles and cover images—and evaluates pCTR against private test labels using ROC-AUC. It makes the representation/ranker interface measurable through three tracks:

- Task 1 evaluates a submitted item embedding with a fixed DIN ranker.
- Task 2 evaluates a custom CTR ranker using the official item embedding.
- Task 1&2 evaluates the complete cascaded system.

Task 1 fixes the CTR evaluator, Task 2 fixes the organizer embedding and input schema, and Task 1&2 requires two disjoint training stages with no additional features. I therefore treated the assignment as a **reproduction-and-extension study under an externally defined experimental protocol**, rather than an unconstrained model-building exercise.

### Research questions

1. Can structured reconstruction make frozen multimodal item embeddings more useful to a fixed ranker?
2. Does adding collaborative behavior to content embeddings consistently improve the representation?
3. How strongly does the downstream ranker determine whether a multimodal representation is useful?

## 2. System design

### Offline item representation

I implemented four candidate signals:

1. **Official multimodal embedding** preserves the content representation supplied with the challenge.
2. **Residual quantization (RQ)** sequentially fits K-Means codebooks to reconstruction residuals. The reconstructed vector acts as a structured, lower-variance approximation.
3. **Behavior co-occurrence SVD** builds a sparse item graph from user sequences, weights neighbors by inverse distance within a sliding window, and reduces it to 128 dimensions.
4. **Multi-version multimodal mean** averages available embedding variants to reduce single-model variance.

Each source is L2-normalized before weighted fusion. The public implementation now aligns every source by `item_id` and fails fast on duplicate IDs, missing required coverage, non-128-dimensional vectors, NaN/Inf values, or invalid weights.

### Original extension: Confidence-Aware Gated Fusion

The fixed A4 weights cannot represent item-level reliability. I therefore implemented CAGF, a small offline gate whose inputs combine intrinsic source confidence and cross-source geometry. It learns per-item weights through leave-one-source-out reconstruction: each source is hidden from the mixture in turn and predicted from the other available sources. Semantic anchoring, confidence calibration, an entropy floor, and weak batch balancing prevent trivial collapse. No CTR labels or DIN gradients are used, so the embedding model remains separate from the evaluator.

CAGF is registered as A5 with the same DIN protocol. The full 91,718-item representation and diagnostics are complete. Source-weight standard deviations range from 0.0200 to 0.0711, 91,716 distinct weight vectors remain after four-decimal rounding, and every missing source receives zero weight. A5 reached 0.891218 validation AUC: +0.001989 over fixed fusion A4 and -0.004266 versus RQ-only A1.

### Online ranking

The ranking stage combines:

- a target-aware Transformer over historical item features;
- DCN v2 for explicit high-order feature interactions;
- a parallel DNN for implicit nonlinear interactions;
- a final MLP for pCTR prediction.

This architecture is a reproduced and adapted implementation of the published first-place Task 2 approach, not claimed as my original invention. My contribution is the embedding-side extension, its controlled evaluation, and integration into a rule-compliant three-track pipeline.

## 3. Experimental method

I used a controlled A0–A5 protocol. Across all six experiments, the downstream DIN configuration, seed, early-stopping rule, and training data remain fixed; only the item representation changes.

| Variant | Representation | Validation AUC | Delta vs A0 |
|---|---|---:|---:|
| A0 | Official baseline | 0.866881 | — |
| A1 | Official + RQ | **0.895484** | **+0.028603** |
| A2 | Official + behavior SVD | 0.870555 | +0.003674 |
| A3 | Official + multimodal mean | 0.890280 | +0.023399 |
| A4 | Four-source fusion | 0.889229 | +0.022348 |
| A5 | Confidence-Aware Gated Fusion | **0.891218** | **+0.024337** |

The official FuxiCTR repository reports a 0.8655 best validation AUC for its baseline. My controlled A0 reproduction reached 0.866881; all intervention deltas below are computed against that local controlled run. The full table is machine-readable in [`../results/ablation_summary.csv`](../results/ablation_summary.csv).

## 4. Findings

**RQ was the strongest individual representation intervention.** A1 improved validation AUC by 0.028603, a 3.30% relative lift over the controlled A0 baseline.

**More components were not automatically better.** A4 combined every signal but trailed A1 by 0.006255 AUC. Equal-scale normalization does not ensure that the sources are equally reliable or geometrically compatible.

**Adaptive weighting improved the multi-source control, but did not beat the simpler RQ representation.** A5 exceeded A4 by 0.001989 AUC and A3 by 0.000938, while remaining 0.004266 below A1. This supports the item-specific reliability hypothesis within one recorded seed, but does not establish statistical significance.

**Behavior co-occurrence added only a small standalone gain.** Its +0.003674 lift suggests that naive linear fusion underuses collaborative structure; a learned gate, confidence-aware weighting, or graph objective would be a stronger next experiment.

**The downstream model changes the value of a representation.** The cascaded Transformer–DCN system reached a recorded validation AUC of 0.972976 and a Codabench-displayed test AUC of 0.9817. I interpret this as evidence of representation–ranker compatibility, not as proof that one representation is universally superior.

### External benchmark context

The published first-place Task 2 report records 0.9776 validation AUC for its full model and 0.9839 on its leaderboard. Those values are useful context for the reproduced ranker, but my 0.9817 score belongs to the cascaded Task 1&2 submission and therefore cannot establish a direct rank. I make no award, placement, or SOTA claim.

## 5. Engineering decisions

- Sparse co-occurrence construction avoids a dense item-by-item matrix.
- Edge pruning bounds memory usage during graph construction.
- Configuration-driven experiments separate modeling choices from execution.
- Compliance checks encode challenge constraints before packaging a submission.
- A rule-to-code traceability table connects every official track invariant to a configuration or validation step.
- The project health check verifies source syntax, ablation evidence, probability ranges, sequential IDs, ZIP integrity, and attribution without requiring the GPU stack.
- Large datasets and checkpoints remain outside the public package; scripts and download instructions preserve reproducibility.

## 6. Limitations and next experiments

- Reported runs use one fixed seed per controlled variant; repeated seeds and confidence intervals are still needed.
- A5 has one fixed-seed downstream result; repeated seeds and confidence intervals are required before claiming a stable CAGF gain over A4.
- The best A5 checkpoint achieved logloss 2.254988 and training became unstable after epoch 14, so ranking gains should not be confused with good calibration or optimization stability.
- RQ uses a small global codebook. Product quantization or larger codebooks may preserve more local geometry.
- The current behavior graph uses co-occurrence rather than direction, timestamps, or negative feedback.
- Latency, throughput, calibration, and memory were not benchmarked, so this is an offline research prototype rather than a production deployment claim.
- The repository reproduces a published ranking architecture; novelty is limited to the representation-side experiments, integration, and analysis.

## 7. Reproduction

```bash
pip install -r requirements.txt
python scripts/check_data.py
bash scripts/run_ablation.sh 0
python scripts/project_healthcheck.py
```

The complete command sequence is documented in [`WORKFLOW.md`](WORKFLOW.md).
