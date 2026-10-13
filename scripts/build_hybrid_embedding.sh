#!/usr/bin/env bash
# 离线构建混合物品嵌入（行为 SVD → 四源融合）
set -euo pipefail
mkdir -p logs embeddings

echo "[1/2] 行为共现图 + SVD..."
python src/embedding/build_behavior_emb.py \
  --seq_path data/item_seq.parquet \
  --item_info data/MicroLens_1M_x1/item_info.parquet \
  --out embeddings/behavior_item_emb_128.parquet \
  --window 5 --dim 128 --max_edges 15000000 --svd_iter 15 \
  2>&1 | tee logs/build_behavior_emb.log

echo "[2/2] RQ + 行为 + 多版本嵌入融合..."
python src/embedding/build_hybrid_emb.py \
  --item_info data/MicroLens_1M_x1/item_info.parquet \
  --behavior_emb embeddings/behavior_item_emb_128.parquet \
  --item_emb_path data/item_emb.parquet \
  --out data/MicroLens_1M_x1/item_info_behavior_fusion.parquet \
  --use_rq --use_mm_ensemble \
  2>&1 | tee logs/build_hybrid_emb.log

ls -lh data/MicroLens_1M_x1/item_info_behavior_fusion.parquet
echo "[完成] 混合嵌入已写入 item_info_behavior_fusion.parquet"
python scripts/check_compliance.py --task task12 || true
