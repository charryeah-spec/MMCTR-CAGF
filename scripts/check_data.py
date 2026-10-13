#!/usr/bin/env python3
"""检查 MicroLens-1M 数据文件是否齐全，并打印基本规模信息。"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "data/MicroLens_1M_x1/train.parquet",
    "data/MicroLens_1M_x1/valid.parquet",
    "data/MicroLens_1M_x1/test.parquet",
    "data/MicroLens_1M_x1/item_info.parquet",
    "data/item_emb.parquet",
    "data/item_feature.parquet",
    "data/item_seq.parquet",
]


def main() -> None:
    ok = True
    for rel in REQUIRED:
        path = ROOT / rel
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"[OK] {rel}  ({size_mb:.2f} MB)")
        else:
            print(f"[缺失] {rel}")
            ok = False

    if not ok:
        raise SystemExit(1)

    train = pd.read_parquet(ROOT / "data/MicroLens_1M_x1/train.parquet")
    item_info = pd.read_parquet(ROOT / "data/MicroLens_1M_x1/item_info.parquet")
    print(f"\n训练集: {train.shape[0]:,} 条, 列: {list(train.columns)}")
    print(f"物品表: {item_info.shape[0]:,} 个, 嵌入维度: {len(item_info['item_emb_d128'].iloc[1])}")


if __name__ == "__main__":
    main()
