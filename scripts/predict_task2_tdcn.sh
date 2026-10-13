#!/usr/bin/env bash
# Task 2 测试集预测
# 用法: bash scripts/predict_task2_tdcn.sh <EXPERIMENT_ID> [GPU_ID]
set -euo pipefail
[ "$#" -ge 1 ] || { echo "用法: bash scripts/predict_task2_tdcn.sh <EXPERIMENT_ID> [GPU]"; exit 1; }
EXPID="$1"
GPU="${2:-0}"
python prediction.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid "$EXPID" --gpu "$GPU" --task-col Task2
ls -lh submission/
head submission/prediction.csv
