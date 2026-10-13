#!/usr/bin/env python3
"""MM-CTR 赛题合规自检：数据格式、配置约束、提交文件校验。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

TASK1_ALLOWED_TUNER = {
    "embedding_regularizer",
    "net_regularizer",
    "net_dropout",
    "learning_rate",
    "batch_size",
}
OFFICIAL_FEATURE_COLS = {
    "user_id", "item_seq", "likes_level", "views_level",
    "item_id", "item_tags", "item_emb_d128", "label",
}
OFFICIAL_ITEM_INFO = "data/MicroLens_1M_x1/item_info.parquet"


def norm_data_path(p: str) -> str:
    return str(Path(p).as_posix()).lstrip("./")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def model_from_config(cfg: dict) -> str:
    base = load_yaml(ROOT / cfg["base_config"] / "model_config.yaml")
    return base[cfg["base_expid"]]["model"]


def check_data() -> list[str]:
    errors = []
    required = [
        "data/MicroLens_1M_x1/train.parquet",
        "data/MicroLens_1M_x1/valid.parquet",
        "data/MicroLens_1M_x1/test.parquet",
        OFFICIAL_ITEM_INFO,
        "data/item_seq.parquet",
        "data/item_emb.parquet",
    ]
    for rel in required:
        if not (ROOT / rel).exists():
            errors.append(f"Missing required dataset file: {rel}")
    item_info = ROOT / OFFICIAL_ITEM_INFO
    if item_info.exists():
        df = pd.read_parquet(item_info)
        if list(df.columns) != ["item_id", "item_tags", "item_emb_d128"]:
            errors.append(f"Unexpected item_info columns: {df.columns.tolist()}")
        dim = len(df["item_emb_d128"].iloc[1])
        if dim != 128:
            errors.append(f"item_emb_d128 must be 128-dim, got {dim}")
    return errors


def feature_cols_from_config(cfg: dict) -> set[str]:
    ds = cfg.get("dataset_config", {}).get(cfg.get("dataset_id", "MicroLens_1M_x1"), {})
    cols = {c["name"] for c in ds.get("feature_cols", [])}
    label = ds.get("label_col", {}).get("name")
    if label:
        cols.add(label)
    return cols


def check_task1_config(path: Path) -> list[str]:
    errors = []
    cfg = load_yaml(path)
    model = model_from_config(cfg)
    if model != "DIN":
        errors.append(f"Task1 must use fixed DIN model, got {model} in {path}")

    tuner = set(cfg.get("tuner_space", {}).keys())
    extra = tuner - TASK1_ALLOWED_TUNER
    if extra:
        errors.append(f"Task1 config has disallowed tuner keys {extra}; only {TASK1_ALLOWED_TUNER}")

    ds = cfg.get("dataset_config", {}).get("MicroLens_1M_x1", {})
    item_info = ds.get("item_info", "")
    if norm_data_path(item_info) == OFFICIAL_ITEM_INFO:
        errors.append(
            "Task1 custom-embedding config should not use official item_info only; "
            "build hybrid embedding first or use DIN_microlens_mmctr_tuner_config_01.yaml for baseline."
        )
    cols = feature_cols_from_config(cfg)
    if cols != OFFICIAL_FEATURE_COLS:
        errors.append(f"Task1 feature columns changed: {cols}")
    return errors


def check_task2_config(path: Path) -> list[str]:
    errors = []
    cfg = load_yaml(path)
    model = model_from_config(cfg)
    if model != "Transformer_DCN":
        errors.append(
            f"This Task2 reproduction must use one Transformer_DCN neural model, got {model}"
        )

    ds = cfg.get("dataset_config", {}).get("MicroLens_1M_x1", {})
    if norm_data_path(ds.get("item_info", "")) != OFFICIAL_ITEM_INFO:
        errors.append(
            f"Task2 must use official embeddings only: {OFFICIAL_ITEM_INFO}, "
            f"got {ds.get('item_info')}"
        )
    cols = feature_cols_from_config(cfg)
    if cols != OFFICIAL_FEATURE_COLS:
        errors.append(f"Task2 input features changed: {cols}")
    return errors


def check_task12_config(path: Path) -> list[str]:
    errors = []
    cfg = load_yaml(path)
    model = model_from_config(cfg)
    if model != "Transformer_DCN":
        errors.append(
            f"Task1&2 ranker must be the separately trained Transformer_DCN, got {model}"
        )
    ds = cfg.get("dataset_config", {}).get("MicroLens_1M_x1", {})
    if norm_data_path(ds.get("item_info", "")) == OFFICIAL_ITEM_INFO:
        errors.append("Task1&2 should use fused item_info, not official item_info only")
    cols = feature_cols_from_config(cfg)
    if cols != OFFICIAL_FEATURE_COLS:
        errors.append(f"Task1&2 added/removed features: {cols}")
    fusion = ROOT / "data/MicroLens_1M_x1/item_info_behavior_fusion.parquet"
    if not fusion.exists():
        errors.append("Task1&2 fused item_info missing; run scripts/build_hybrid_embedding.sh")
    return errors


TASK_PRED_COLS = {
    "task1": "Task1",
    "task2": "Task2",
    "task12": "Task1&2",
}


def check_submission(path: Path, task: str = "") -> list[str]:
    errors = []
    if not path.exists():
        return [f"Submission file not found: {path}"]
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c != "ID"]
    if task:
        expected = [TASK_PRED_COLS[task]]
        if list(df.columns) != ["ID", expected[0]]:
            errors.append(f"Columns must be ['ID','{expected[0]}'], got {list(df.columns)}")
        pred_cols = expected
    elif not pred_cols:
        errors.append("Submission must include at least one prediction column")
    else:
        allowed = set(TASK_PRED_COLS.values())
        unknown = [c for c in pred_cols if c not in allowed]
        if unknown:
            errors.append(f"Unknown prediction columns: {unknown}; allowed {sorted(allowed)}")
    test_n = len(pd.read_parquet(ROOT / "data/MicroLens_1M_x1/test.parquet"))
    if len(df) != test_n:
        errors.append(f"Prediction rows {len(df)} != test rows {test_n}")
    if not df["ID"].equals(pd.Series(range(len(df)))):
        errors.append("ID column must be 0..N-1")
    for col in pred_cols:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            errors.append(f"{col} column contains NaN")
        if ((df[col] < 0) | (df[col] > 1)).any():
            errors.append(f"{col} predictions should be probabilities in [0,1]")
    return errors


def main():
    ap = argparse.ArgumentParser(description="MM-CTR compliance checker")
    ap.add_argument("--task", choices=["task1", "task2", "task12", "all"], default="all")
    ap.add_argument("--submission", default="", help="Path to submission/prediction.csv")
    args = ap.parse_args()

    errors: list[str] = []
    print("[CHECK] dataset files & embedding format")
    errors.extend(check_data())

    if args.task in ("task1", "all"):
        print("[CHECK] Task1 config")
        errors.extend(check_task1_config(ROOT / "config/DIN_behavior_fusion_task1_config.yaml"))

    if args.task in ("task2", "all"):
        print("[CHECK] Task2 config")
        errors.extend(check_task2_config(ROOT / "config/Transformer_DCN_microlens_mmctr_tuner_config_01.yaml"))

    if args.task in ("task12", "all"):
        print("[CHECK] Task1&2 config")
        errors.extend(check_task12_config(ROOT / "config/Transformer_DCN_behavior_fusion_config.yaml"))

    if args.submission:
        print("[CHECK] submission format")
        sub_task = args.task if args.task in TASK_PRED_COLS else ""
        errors.extend(check_submission(ROOT / args.submission, task=sub_task))

    if errors:
        print("\n[FAIL] compliance issues:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("\n[PASS] compliance checks passed for", args.task)


if __name__ == "__main__":
    main()
