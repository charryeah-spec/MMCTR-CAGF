#!/usr/bin/env python3
"""从 FuxiCTR 训练结果 CSV 中解析并输出验证集 AUC 最高的 experiment_id。"""
from __future__ import annotations

import re
import sys

import pandas as pd


def parse_result_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    rows = []
    for _, row in df.iterrows():
        text = " ".join(map(str, row.tolist()))
        exp_match = re.search(r"\[exp_id\]\s+([^,\s]+)", text)
        auc_match = re.search(r"AUC[:=]\s*([0-9.]+)", text, re.I)
        loss_match = re.search(r"logloss[:=]\s*([0-9.]+)", text, re.I)
        if exp_match and auc_match:
            rows.append({
                "experiment_id": exp_match.group(1),
                "AUC": float(auc_match.group(1)),
                "logloss": float(loss_match.group(1)) if loss_match else None,
            })
    if not rows:
        print(df.tail(10).to_string(index=False))
        raise SystemExit("[ERROR] 无法从 CSV 解析实验结果，请确认训练已完成")
    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python scripts/find_best_expid.py <result_csv>")
        sys.exit(1)
    ranked = parse_result_csv(sys.argv[1])
    print(ranked.head(10).to_string(index=False))
    print("\n[BEST_EXPID]", ranked.iloc[0]["experiment_id"])


if __name__ == "__main__":
    main()
