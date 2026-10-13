#!/usr/bin/env python3
"""将自定义嵌入与官方 item_info 按权重线性融合。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def to_mat(s): return np.asarray(s.tolist(), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_info", default="data/MicroLens_1M_x1/item_info.parquet")
    ap.add_argument("--custom_emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--official_weight", type=float, default=0.7)
    ap.add_argument("--custom_weight", type=float, default=0.3)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    item_info = pd.read_parquet(args.item_info)
    custom = pd.read_parquet(args.custom_emb)
    if "item_id" in custom.columns:
        custom = custom.sort_values("item_id").reset_index(drop=True)
    off = normalize(to_mat(item_info["item_emb_d128"]), norm="l2", axis=1)
    cus = normalize(to_mat(custom["item_emb_d128"]), norm="l2", axis=1)
    if off.shape != cus.shape: raise ValueError(f"shape mismatch {off.shape} vs {cus.shape}")
    fused = normalize(args.official_weight*off + args.custom_weight*cus, norm="l2", axis=1).astype(np.float32)
    out_df = item_info.copy()
    out_df["item_emb_d128"] = [r.tolist() for r in fused]
    out_df.to_parquet(args.out, index=False)
    print("[OK] saved", args.out, fused.shape)
if __name__ == "__main__":
    main()
