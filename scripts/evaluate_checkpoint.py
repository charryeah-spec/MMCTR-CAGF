#!/usr/bin/env python3
"""Evaluate a saved FuxiCTR checkpoint without repeating model training."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from fuxictr.features import FeatureMap  # noqa: E402
from fuxictr.pytorch.dataloaders import RankDataLoader  # noqa: E402
from fuxictr.utils import load_config, print_to_list, set_logger  # noqa: E402

import src as model_zoo  # noqa: E402
from src.mmctr_dataloader import MMCTRDataLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="FuxiCTR config directory")
    parser.add_argument("--expid", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", help="defaults to the model checkpoint path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    params = load_config(args.config, args.expid)
    params["gpu"] = args.gpu
    set_logger(params)

    data_dir = Path(params["data_root"]) / params["dataset_id"]
    feature_map = FeatureMap(params["dataset_id"], str(data_dir))
    feature_map.load(str(data_dir / "feature_map.json"), params)

    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    checkpoint = args.checkpoint or model.checkpoint
    model.load_weights(checkpoint)

    params["data_loader"] = MMCTRDataLoader
    _, valid_gen = RankDataLoader(feature_map, stage="train", **params).make_iterator()
    result = model.evaluate(valid_gen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", *result.keys()])
        writer.writeheader()
        writer.writerow({"experiment_id": args.expid, **result})
    print(f"[OK] {args.expid}: {print_to_list(result)}")
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
