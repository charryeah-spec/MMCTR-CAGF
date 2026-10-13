#!/usr/bin/env bash
# Task 2：官方嵌入 + Transformer_DCN 训练
# 用法: bash scripts/run_task2_tdcn.sh [GPU_ID]
set -euo pipefail
GPU="${1:-0}"
mkdir -p logs

nohup python run_param_tuner.py \
  --config config/Transformer_DCN_microlens_mmctr_tuner_config_01.yaml \
  --gpu "$GPU" > logs/task2_tdcn.log 2>&1 &
echo "[INFO] PID=$!  日志: logs/task2_tdcn.log"
echo "训练完成后:"
echo "  python scripts/find_best_expid.py Transformer_DCN_microlens_mmctr_tuner_config_01.csv"
echo "  bash scripts/predict_task2_tdcn.sh <BEST_EXPID> $GPU"
echo "  python scripts/check_compliance.py --task task2 --submission submission/prediction.csv"
