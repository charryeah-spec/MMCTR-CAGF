#!/usr/bin/env python3
"""Task 1 嵌入消融：生成配置、训练 DIN、汇总验证集 AUC（A0–A5）。"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/embedding"))

from ablation_presets import ORDER, PRESETS  # noqa: E402

# 固定 DIN 超参，保证消融对照公平（仅嵌入不同）
DIN_HPARAMS = {
    "model": "DIN",
    "dataset_id": "MicroLens_1M_x1",
    "loss": "binary_crossentropy",
    "metrics": ["AUC", "logloss"],
    "task": "binary_classification",
    "optimizer": "adam",
    "learning_rate": 1.0e-3,
    "embedding_regularizer": 1.0e-6,
    "net_regularizer": 0,
    "net_dropout": 0.1,
    "batch_size": 8192,
    "embedding_dim": 64,
    "dnn_hidden_units": [1024, 512, 256],
    "dnn_activations": "ReLU",
    "attention_hidden_units": [512, 256],
    "attention_hidden_activations": "ReLU",
    "attention_output_activation": None,
    "attention_dropout": 0.1,
    "din_use_softmax": False,
    "batch_norm": True,
    "epochs": 100,
    "shuffle": True,
    "seed": 20242025,
    "monitor": "AUC",
    "monitor_mode": "max",
    "max_len": 64,
    "accumulation_steps": 1,
    "early_stop_patience": 3,
    "num_workers": 3,
    "model_root": "./checkpoints/",
    "pickle_feature_encoder": True,
    "save_best_only": True,
    "eval_steps": None,
    "debug_mode": False,
    "group_id": "user_id",
    "use_features": None,
    "feature_specs": None,
    "feature_config": None,
    "verbose": 1,
}

FEATURE_COLS = [
    {"name": "user_id", "active": True, "dtype": "int", "type": "meta"},
    {"name": "item_seq", "active": True, "dtype": "int", "type": "meta"},
    {"name": "likes_level", "active": True, "dtype": "int", "type": "categorical", "vocab_size": 11},
    {"name": "views_level", "active": True, "dtype": "int", "type": "categorical", "vocab_size": 11},
    {
        "name": "item_id", "active": True, "dtype": "int", "type": "categorical",
        "vocab_size": 91718, "source": "item",
    },
    {
        "name": "item_tags", "active": True, "dtype": "int", "type": "sequence",
        "max_len": 5, "vocab_size": 11740, "source": "item",
    },
    {
        "name": "item_emb_d128", "active": True, "dtype": "float", "type": "embedding",
        "source": "item", "embedding_dim": 128,
    },
]


def rel_item_info(path: Path) -> str:
    return "./" + path.relative_to(ROOT).as_posix()


def write_ablation_config(code: str) -> tuple[Path, str]:
    preset = PRESETS[code]
    config_dir = ROOT / f"config/ablation_{code}"
    config_dir.mkdir(parents=True, exist_ok=True)

    expid = f"DIN_ablation_{code}_001"
    model_cfg = {expid: dict(DIN_HPARAMS)}
    dataset_cfg = {
        "MicroLens_1M_x1": {
            "data_root": "./data/",
            "data_format": "parquet",
            "train_data": "./data/MicroLens_1M_x1/train.parquet",
            "valid_data": "./data/MicroLens_1M_x1/valid.parquet",
            "test_data": "./data/MicroLens_1M_x1/test.parquet",
            "item_info": rel_item_info(preset.item_info_path),
            "rebuild_dataset": False,
            "feature_cols": FEATURE_COLS,
            "label_col": {"name": "label", "dtype": "float"},
        }
    }

    with open(config_dir / "model_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False, allow_unicode=True)
    with open(config_dir / "dataset_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_cfg, f, sort_keys=False, allow_unicode=True)

    return config_dir, expid


def parse_auc_from_csv(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for line in text.splitlines():
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


def train_one(code: str, gpu: int, log_dir: Path) -> dict:
    config_dir, expid = write_ablation_config(code)
    log_file = log_dir / f"ablation_{code}.log"
    cmd = [
        sys.executable, "run_expid.py",
        "--config", str(config_dir),
        "--expid", expid,
        "--gpu", str(gpu),
    ]
    print(f"\n[{code}] 训练 DIN: {' '.join(cmd)}")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{code} 训练失败，见日志 {log_file}")

    result = parse_auc_from_csv(ROOT / f"ablation_{code}.csv")
    if not result:
        raise RuntimeError(f"{code} 未解析到 AUC，见 {log_file}")
    result["code"] = code
    result["config_dir"] = str(config_dir.relative_to(ROOT))
    result["log_file"] = str(log_file.relative_to(ROOT))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Task1 嵌入消融实验（A0–A5）")
    ap.add_argument("--codes", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-embed", action="store_true", help="跳过嵌入构建")
    ap.add_argument("--skip-train", action="store_true", help="仅构建嵌入，不训练")
    ap.add_argument("--rebuild-behavior", action="store_true")
    ap.add_argument("--out", default="results/ablation_summary.csv")
    args = ap.parse_args()

    log_dir = ROOT / "logs/ablation"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_embed:
        embed_cmd = [
            sys.executable, "scripts/build_ablation_embeddings.py",
            "--codes", *args.codes,
        ]
        if args.rebuild_behavior:
            embed_cmd.append("--rebuild-behavior")
        subprocess.run(embed_cmd, cwd=ROOT, check=True)

    if args.skip_train:
        print("[INFO] 已跳过训练阶段")
        return

    rows = []
    for code in args.codes:
        rows.append(train_one(code, args.gpu, log_dir))

    import pandas as pd

    summary = []
    for row in rows:
        preset = PRESETS[row["code"]]
        summary.append({
            "实验": row["code"],
            "名称": preset.name,
            "官方": "✓",
            "RQ": "✓" if preset.use_rq else "—",
            "行为SVD": "✓" if preset.use_behavior else "—",
            "多版本均值": "✓" if preset.use_mm else "—",
            "CAGF": "✓" if preset.use_gated else "—",
            "验证集AUC": row["AUC"],
            "logloss": row["logloss"],
            "experiment_id": row["experiment_id"],
        })

    df = pd.DataFrame(summary)
    out_csv = ROOT / args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n========== 消融结果 ==========")
    print(df.to_string(index=False))
    print(f"\n[OK] 已保存 {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
