#!/usr/bin/env bash
# Task 1&2 测试集预测
# 用法: bash scripts/predict_task12.sh <EXPERIMENT_ID> [GPU_ID]
set -euo pipefail
[ "$#" -ge 1 ] || { echo "用法: bash scripts/predict_task12.sh <EXPERIMENT_ID> [GPU]"; exit 1; }
EXPID="$1"
GPU="${2:-0}"
python prediction.py --config config/Transformer_DCN_behavior_fusion_config --expid "$EXPID" --gpu "$GPU" --task-col "Task1&2"
ls -lh submission/
head submission/prediction.csv
