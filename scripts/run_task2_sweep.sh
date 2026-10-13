#!/usr/bin/env bash
# Task 2 超参网格搜索
# 用法: bash scripts/run_task2_sweep.sh [GPU_ID]
set -euo pipefail
GPU="${1:-0}"
mkdir -p logs

nohup python run_param_tuner.py \
  --config config/Transformer_DCN_microlens_mmctr_tuner_config_sweep.yaml \
  --gpu "$GPU" > logs/task2_sweep.log 2>&1 &
echo "[INFO] PID=$!  日志: logs/task2_sweep.log"
echo "完成后: python scripts/find_best_expid.py Transformer_DCN_microlens_mmctr_tuner_config_sweep.csv"
