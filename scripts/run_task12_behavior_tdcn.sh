#!/usr/bin/env bash
# Task 1&2：混合嵌入 + Transformer_DCN 级联训练
# 用法: bash scripts/run_task12_behavior_tdcn.sh [GPU_ID]
set -euo pipefail
GPU="${1:-0}"
mkdir -p logs

bash scripts/build_hybrid_embedding.sh

nohup python run_param_tuner.py \
  --config config/Transformer_DCN_behavior_fusion_config.yaml \
  --gpu "$GPU" > logs/task12_behavior_tdcn.log 2>&1 &
echo "[INFO] PID=$!  日志: logs/task12_behavior_tdcn.log"
echo "训练完成后:"
echo "  python scripts/find_best_expid.py Transformer_DCN_behavior_fusion_config.csv"
echo "  bash scripts/predict_task12.sh <BEST_EXPID> $GPU"
echo "  python scripts/check_compliance.py --task task12 --submission submission/prediction.csv"
