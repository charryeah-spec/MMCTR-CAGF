#!/usr/bin/env python3
"""按 A0–A5 预设构建全部消融用 item_info 嵌入文件。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/embedding"))

from ablation_presets import ORDER, PRESETS, OFFICIAL_ITEM_INFO
from build_hybrid_emb import fuse_embeddings


def ensure_behavior_embedding(force: bool = False) -> None:
    out = ROOT / "embeddings/behavior_item_emb_128.parquet"
    if out.exists() and not force:
        print(f"[SKIP] 行为嵌入已存在: {out}")
        return
    print("[INFO] 构建行为共现 SVD 嵌入...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/embedding/build_behavior_emb.py"),
            "--seq_path", str(ROOT / "data/item_seq.parquet"),
            "--item_info", str(OFFICIAL_ITEM_INFO),
            "--out", str(out),
        ],
        check=True,
        cwd=ROOT,
    )


def build_one(code: str) -> Path:
    preset = PRESETS[code]
    item_path = preset.item_info_path
    if not preset.needs_build:
        if not item_path.exists():
            raise FileNotFoundError(f"官方 item_info 缺失: {item_path}")
        print(f"[A0] 使用官方嵌入: {item_path}")
        return item_path

    print(f"[{code}] {preset.name} -> {item_path}")
    if preset.use_gated:
        from confidence_gated_fusion import build_cagf_embedding

        build_cagf_embedding(
            OFFICIAL_ITEM_INFO,
            item_path,
            behavior_emb_path=ROOT / "embeddings/behavior_item_emb_128.parquet",
            item_emb_path=ROOT / "data/item_emb.parquet",
            diagnostics_path=ROOT / "results/cagf_A5_item_weights.parquet",
            summary_path=ROOT / "results/cagf_A5_summary.json",
            checkpoint_path=ROOT / "checkpoints/cagf_A5_gate.pt",
        )
        return item_path

    fuse_embeddings(
        OFFICIAL_ITEM_INFO,
        item_path,
        behavior_emb=ROOT / "embeddings/behavior_item_emb_128.parquet",
        item_emb_path=ROOT / "data/item_emb.parquet",
        use_rq=preset.use_rq,
        use_behavior=preset.use_behavior,
        use_mm_ensemble=preset.use_mm,
        equal_weights=preset.equal_weights and not preset.use_full_weights,
        official_weight=0.40,
        rq_weight=0.20,
        behavior_weight=0.25,
        mm_weight=0.15,
    )
    return item_path


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 Task1 嵌入消融所需的 item_info 文件")
    ap.add_argument(
        "--codes",
        nargs="+",
        default=ORDER,
        choices=ORDER,
        help="要构建的实验编号，默认 A0–A5",
    )
    ap.add_argument("--rebuild-behavior", action="store_true", help="强制重建行为 SVD 嵌入")
    args = ap.parse_args()

    need_behavior = any(PRESETS[c].use_behavior for c in args.codes)
    if need_behavior:
        ensure_behavior_embedding(force=args.rebuild_behavior)

    built = []
    for code in args.codes:
        built.append((code, build_one(code)))

    print("\n[完成] 消融嵌入文件:")
    for code, path in built:
        print(f"  {code}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
