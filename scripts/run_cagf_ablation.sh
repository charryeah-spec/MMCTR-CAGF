#!/usr/bin/env bash
# Focused comparison: RQ-only (A1) vs fixed fusion (A4) vs adaptive CAGF (A5).
set -euo pipefail

GPU="${1:-0}"
shift || true

python scripts/run_ablation.py \
  --gpu "$GPU" \
  --codes A1 A4 A5 \
  --out results/cagf_ablation_summary.csv \
  "$@"
