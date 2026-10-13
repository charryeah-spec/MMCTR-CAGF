#!/usr/bin/env bash
# Task 1：混合嵌入 + DIN 训练
# 用法: bash scripts/run_task1_behavior.sh [GPU_ID]
set -euo pipefail
GPU="${1:-0}"
mkdir -p logs

bash scripts/build_hybrid_embedding.sh

echo "[Task 1] 启动 DIN 训练..."
nohup python run_param_tuner.py \
  --config config/DIN_behavior_fusion_task1_config.yaml \
  --gpu "$GPU" > logs/task1_din.log 2>&1 &
echo "[INFO] PID=$!  日志: logs/task1_din.log"
echo "训练完成后:"
echo "  python scripts/find_best_expid.py DIN_behavior_fusion_task1_config.csv"
echo "  bash scripts/predict_task1_din.sh <BEST_EXPID> $GPU"
echo "  python scripts/check_compliance.py --task task1 --submission submission/prediction.csv"
