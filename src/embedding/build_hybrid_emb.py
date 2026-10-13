#!/usr/bin/env python3
"""四源加权融合生成混合物品嵌入，供 DIN / Transformer_DCN 离线读取（Task 1 & 1&2）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_quantizer import ResidualQuantizer


def to_mat(series: pd.Series, *, source: str = "embedding") -> np.ndarray:
    """Convert a list-valued embedding column into a validated dense matrix."""
    try:
        matrix = np.asarray(series.tolist(), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} cannot be converted to a dense float matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != 128:
        raise ValueError(f"{source} must have shape (n_items, 128), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source} contains NaN or infinite values")
    return matrix


def align_embeddings(
    table: pd.DataFrame,
    target_ids: np.ndarray,
    *,
    source: str,
    require_full_coverage: bool,
) -> tuple[np.ndarray, float]:
    """Align an embedding table to item_info order instead of assuming row order."""
    required = {"item_id", "item_emb_d128"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"{source} missing columns: {sorted(missing_columns)}")
    if table["item_id"].duplicated().any():
        duplicated = table.loc[table["item_id"].duplicated(), "item_id"].iloc[0]
        raise ValueError(f"{source} contains duplicate item_id={duplicated}")

    source_ids = table["item_id"].to_numpy(dtype=np.int64)
    source_matrix = to_mat(table["item_emb_d128"], source=source)
    return align_embedding_matrix(
        source_ids,
        source_matrix,
        target_ids,
        source=source,
        require_full_coverage=require_full_coverage,
    )


def align_embedding_matrix(
    source_ids: np.ndarray,
    source_matrix: np.ndarray,
    target_ids: np.ndarray,
    *,
    source: str,
    require_full_coverage: bool,
) -> tuple[np.ndarray, float]:
    """Align a validated matrix without converting it back to Python lists."""
    positions = pd.Index(source_ids).get_indexer(target_ids)
    found = positions >= 0
    coverage = float(found.mean()) if len(found) else 1.0
    if require_full_coverage and not found.all():
        missing = target_ids[~found][:5].tolist()
        raise ValueError(
            f"{source} covers {coverage:.2%} of target items; missing examples: {missing}"
        )

    aligned = np.zeros((len(target_ids), 128), dtype=np.float32)
    aligned[found] = source_matrix[positions[found]]
    return aligned, coverage


def load_mm_ensemble(item_emb_path, target_ids):
    mm = pd.read_parquet(item_emb_path)
    emb_cols = [c for c in mm.columns if c.startswith("item_emb_d128")]
    if not emb_cols:
        raise ValueError(f"No embedding columns in {item_emb_path}")
    if "item_id" not in mm.columns:
        raise ValueError(f"{item_emb_path} missing item_id column")
    if mm["item_id"].duplicated().any():
        duplicated = mm.loc[mm["item_id"].duplicated(), "item_id"].iloc[0]
        raise ValueError(f"{item_emb_path} contains duplicate item_id={duplicated}")
    stack = np.stack(
        [to_mat(mm[c], source=f"{item_emb_path}:{c}") for c in emb_cols], axis=0
    ).mean(axis=0)
    stack = normalize(stack, norm="l2", axis=1)
    return align_embedding_matrix(
        mm["item_id"].to_numpy(dtype=np.int64),
        stack,
        target_ids,
        source=str(item_emb_path),
        require_full_coverage=False,
    )


def fuse_embeddings(
    item_info_path: str | Path,
    out_path: str | Path,
    *,
    behavior_emb: str | Path = "embeddings/behavior_item_emb_128.parquet",
    item_emb_path: str | Path = "data/item_emb.parquet",
    use_rq: bool = True,
    use_behavior: bool = True,
    use_mm_ensemble: bool = True,
    equal_weights: bool = False,
    official_weight: float = 0.40,
    rq_weight: float = 0.20,
    behavior_weight: float = 0.25,
    mm_weight: float = 0.15,
    rq_clusters: int = 5,
    rq_layers: int = 3,
) -> tuple[int, list[float]]:
    item_info = pd.read_parquet(item_info_path)
    required = {"item_id", "item_emb_d128"}
    missing_columns = required - set(item_info.columns)
    if missing_columns:
        raise ValueError(f"item_info missing columns: {sorted(missing_columns)}")
    if item_info["item_id"].duplicated().any():
        raise ValueError("item_info contains duplicate item_id values")
    target_ids = item_info["item_id"].to_numpy(dtype=np.int64)
    off = normalize(
        to_mat(item_info["item_emb_d128"], source=str(item_info_path)),
        norm="l2",
        axis=1,
    )

    parts, weights = [off], [official_weight]
    if use_rq:
        rq = ResidualQuantizer(rq_clusters, rq_layers)
        rq.fit(off)
        rq_emb = rq.reconstruct(off)
        parts.append(rq_emb)
        weights.append(rq_weight)
        print("[INFO] RQ reconstruction enabled")

    behavior_emb = Path(behavior_emb)
    if use_behavior:
        if not behavior_emb.exists():
            raise FileNotFoundError(
                f"行为嵌入不存在: {behavior_emb}，请先运行 build_behavior_emb.py"
            )
        behavior = pd.read_parquet(behavior_emb)
        beh, coverage = align_embeddings(
            behavior,
            target_ids,
            source=str(behavior_emb),
            require_full_coverage=True,
        )
        beh = normalize(beh, norm="l2", axis=1)
        parts.append(beh)
        weights.append(behavior_weight)
        print(f"[INFO] behavior SVD embedding enabled (coverage={coverage:.2%})")

    item_emb_path = Path(item_emb_path)
    if use_mm_ensemble:
        if not item_emb_path.exists():
            raise FileNotFoundError(f"多版本嵌入不存在: {item_emb_path}")
        mm, coverage = load_mm_ensemble(item_emb_path, target_ids)
        parts.append(mm)
        weights.append(mm_weight)
        print(f"[INFO] MM ensemble enabled (coverage={coverage:.2%})")

    if equal_weights:
        weights = [1.0] * len(parts)

    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError(f"Active fusion weights must be finite and non-negative: {weights}")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("At least one active fusion weight must be positive")
    fused = sum((w / w_sum) * p for w, p in zip(weights, parts))
    fused = normalize(fused, norm="l2", axis=1).astype(np.float32)
    assert fused.shape[1] == 128

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = item_info.copy()
    out_df["item_emb_d128"] = [r.tolist() for r in fused]
    out_df.to_parquet(out_path, index=False)
    norm_weights = [round(w / w_sum, 3) for w in weights]
    print("[OK] saved", out_path, fused.shape, "weights", norm_weights)
    return len(parts), norm_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_info", default="data/MicroLens_1M_x1/item_info.parquet")
    ap.add_argument("--behavior_emb", default="embeddings/behavior_item_emb_128.parquet")
    ap.add_argument("--item_emb_path", default="data/item_emb.parquet")
    ap.add_argument("--out", default="data/MicroLens_1M_x1/item_info_behavior_fusion.parquet")
    ap.add_argument("--official_weight", type=float, default=0.40)
    ap.add_argument("--rq_weight", type=float, default=0.20)
    ap.add_argument("--behavior_weight", type=float, default=0.25)
    ap.add_argument("--mm_weight", type=float, default=0.15)
    ap.add_argument("--use_rq", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_behavior", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_mm_ensemble", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--equal_weights", action="store_true", help="各激活组件等权融合（消融用）")
    ap.add_argument("--rq_clusters", type=int, default=5)
    ap.add_argument("--rq_layers", type=int, default=3)
    args = ap.parse_args()

    fuse_embeddings(
        args.item_info,
        args.out,
        behavior_emb=args.behavior_emb,
        item_emb_path=args.item_emb_path,
        use_rq=args.use_rq,
        use_behavior=args.use_behavior,
        use_mm_ensemble=args.use_mm_ensemble,
        equal_weights=args.equal_weights,
        official_weight=args.official_weight,
        rq_weight=args.rq_weight,
        behavior_weight=args.behavior_weight,
        mm_weight=args.mm_weight,
        rq_clusters=args.rq_clusters,
        rq_layers=args.rq_layers,
    )


if __name__ == "__main__":
    main()
