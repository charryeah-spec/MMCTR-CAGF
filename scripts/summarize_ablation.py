#!/usr/bin/env python3
"""从消融训练产物中汇总 A0–A5 验证集 AUC（无需重新训练）。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/embedding"))

from ablation_presets import ORDER, PRESETS  # noqa: E402


def parse_result_csv(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        exp = re.search(r"\[exp_id\]\s+([^,\s]+)", line)
        auc = re.search(r"AUC[:=]\s*([0-9.]+)", line, re.I)
        loss = re.search(r"logloss[:=]\s*([0-9.]+)", line, re.I)
        if exp and auc:
            rows.append({
                "experiment_id": exp.group(1),
                "AUC": float(auc.group(1)),
                "logloss": float(loss.group(1)) if loss else None,
            })
    return rows[-1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument(
        "--out",
        default="results/ablation_summary.csv",
        help="汇总表输出路径",
    )
    args = ap.parse_args()

    summary = []
    for code in args.codes:
        preset = PRESETS[code]
        parsed = parse_result_csv(ROOT / f"ablation_{code}.csv")
        if not parsed:
            print(f"[WARN] {code}: 未找到 ablation_{code}.csv，跳过")
            continue
        summary.append({
            "实验": code,
            "名称": preset.name,
            "官方": "✓",
            "RQ": "✓" if preset.use_rq else "—",
            "行为SVD": "✓" if preset.use_behavior else "—",
            "多版本均值": "✓" if preset.use_mm else "—",
            "CAGF": "✓" if preset.use_gated else "—",
            "验证集AUC": parsed["AUC"],
            "logloss": parsed["logloss"],
            "experiment_id": parsed["experiment_id"],
        })

    if not summary:
        raise SystemExit("[ERROR] 没有可汇总的结果，请先运行 scripts/run_ablation.py")

    df = pd.DataFrame(summary)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\n[OK] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
