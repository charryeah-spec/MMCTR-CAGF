#!/usr/bin/env python3
"""从用户行为序列构建物品共现图，经 SVD 得到 128 维行为嵌入（Task 1）。"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def infer_seq_col(df, preferred="item_seq"):
    if preferred in df.columns:
        return preferred
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    raise ValueError(f"Cannot infer sequence column: {list(df.columns)}")

def parse_seq(x):
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist() if int(v) > 0]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x if int(v) > 0]
    if pd.isna(x):
        return []
    return [int(v) for v in str(x).replace(",", " ").split() if int(v) > 0]

def build_edges(seq_iter, window=5, max_edges=15000000):
    counts = defaultdict(float)
    for n, seq in enumerate(seq_iter, 1):
        L = len(seq)
        for i, src in enumerate(seq):
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if i == j:
                    continue
                dst = seq[j]
                if src == dst:
                    continue
                dist = abs(i - j)
                counts[(src, dst)] += 1.0 / dist
                counts[(dst, src)] += 1.0 / dist
        if n % 200000 == 0:
            print(f"[INFO] rows={n}, edges={len(counts)}")
            if len(counts) > max_edges:
                counts = defaultdict(float, sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_edges])
                print(f"[INFO] pruned edges={len(counts)}")
    if len(counts) > max_edges:
        counts = defaultdict(float, sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_edges])
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_path", default="data/item_seq.parquet")
    ap.add_argument("--item_info", default="data/MicroLens_1M_x1/item_info.parquet")
    ap.add_argument("--out", default="embeddings/behavior_item_emb_128.parquet")
    ap.add_argument("--seq_col", default="item_seq")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max_edges", type=int, default=15000000)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--svd_iter", type=int, default=15)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    item_info = pd.read_parquet(args.item_info)
    num_items = len(item_info)
    seq_df = pd.read_parquet(args.seq_path)
    seq_col = infer_seq_col(seq_df, args.seq_col)
    if args.max_rows > 0:
        seq_df = seq_df.head(args.max_rows)
    print("[INFO] num_items", num_items, "seq_rows", len(seq_df), "seq_col", seq_col)
    edges = build_edges((parse_seq(x) for x in seq_df[seq_col]), args.window, args.max_edges)
    rows, cols, vals, raw_vals = [], [], [], []
    for (i, j), v in edges.items():
        if 0 <= i < num_items and 0 <= j < num_items:
            rows.append(i)
            cols.append(j)
            vals.append(np.log1p(v))
            raw_vals.append(v)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(num_items, num_items), dtype=np.float32)
    mat = mat + sparse.eye(num_items, dtype=np.float32) * 1e-3
    svd = TruncatedSVD(n_components=args.dim, n_iter=args.svd_iter, random_state=20242025)
    emb = normalize(svd.fit_transform(mat), norm="l2", axis=1).astype(np.float32)
    support = np.zeros(num_items, dtype=np.float32)
    np.add.at(support, np.asarray(rows, dtype=np.int64), np.asarray(raw_vals, dtype=np.float32))
    pd.DataFrame({
        "item_id": np.arange(num_items, dtype=np.int64),
        "item_emb_d128": [r.tolist() for r in emb],
        "behavior_support": support,
    }).to_parquet(args.out, index=False)
    print(
        "[OK] saved", args.out,
        "var", float(svd.explained_variance_ratio_.sum()),
        "supported_items", int((support > 0).sum()),
    )
if __name__ == "__main__":
    main()
