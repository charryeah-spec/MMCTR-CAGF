#!/usr/bin/env bash
# Task 1 嵌入消融实验 A0–A5（构建嵌入 + DIN 训练 + 结果汇总）
# 用法:
#   bash scripts/run_ablation.sh [GPU_ID]           # 全流程
#   bash scripts/run_ablation.sh 0 A1 A4 A5         # 对比 RQ / 固定融合 / CAGF
#   bash scripts/run_ablation.sh 0 --embed-only     # 只构建嵌入
set -euo pipefail

GPU="${1:-0}"
shift || true

EMBED_ONLY=false
REBUILD_BEHAVIOR=false
CODES=()
EXTRA=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --embed-only) EMBED_ONLY=true ;;
    --rebuild-behavior) REBUILD_BEHAVIOR=true ;;
    A0|A1|A2|A3|A4|A5) CODES+=("$1") ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

CMD=(python scripts/run_ablation.py --gpu "$GPU")
[ "${#CODES[@]}" -gt 0 ] && CMD+=(--codes "${CODES[@]}")
[ "$REBUILD_BEHAVIOR" = true ] && CMD+=(--rebuild-behavior)
[ "$EMBED_ONLY" = true ] && CMD+=(--skip-train)

echo "[INFO] ${CMD[*]} ${EXTRA[*]:-}"
"${CMD[@]}" "${EXTRA[@]}"

if [ "$EMBED_ONLY" = false ]; then
  if [ "${#CODES[@]}" -gt 0 ]; then
    python scripts/summarize_ablation.py --codes "${CODES[@]}"
  else
    python scripts/summarize_ablation.py
  fi
fi
