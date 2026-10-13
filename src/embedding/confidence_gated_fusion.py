#!/usr/bin/env python3
"""Confidence-Aware Gated Fusion (CAGF) for offline 128-D item embeddings.

CAGF learns per-item source weights without CTR labels. Its leave-one-source-out
objective predicts each available embedding from the remaining sources, while
confidence calibration, semantic anchoring, and collapse prevention regularize
the gate. The learned embedding remains a standalone offline artifact, keeping
Task 1 representation training separate from the downstream CTR model.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hybrid_emb import align_embedding_matrix, align_embeddings, to_mat
from confidence_features import (
    build_confidence_features,
    feature_names,
    multimodal_consistency,
    normalize_rows,
    robust_scale_support,
)
from residual_quantizer import ResidualQuantizer

SOURCE_NAMES = ["official", "rq", "behavior", "mm_ensemble"]
DEFAULT_PRIOR = np.asarray([0.40, 0.20, 0.25, 0.15], dtype=np.float32)


class ConfidenceGate(nn.Module):
    """Small residual gate initialized from source-specific confidence."""

    def __init__(self, input_dim: int, hidden_dim: int, num_sources: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_sources * 2)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, num_sources * 2), num_sources),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        intrinsic_confidence: torch.Tensor,
        available: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        residual_logits = self.network(features)
        confidence_logits = torch.log(intrinsic_confidence.clamp_min(1e-3))
        logits = (residual_logits + confidence_logits) / temperature
        logits = logits.masked_fill(~available, -1e4)
        return torch.softmax(logits, dim=1)


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positions(source_ids: np.ndarray, target_ids: np.ndarray, source: str) -> np.ndarray:
    if pd.Index(source_ids).has_duplicates:
        raise ValueError(f"{source} contains duplicate item_id values")
    return pd.Index(source_ids).get_indexer(target_ids)


def _align_scalar(
    source_ids: np.ndarray,
    values: np.ndarray,
    target_ids: np.ndarray,
    *,
    source: str,
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    positions = _positions(source_ids, target_ids, source)
    found = positions >= 0
    aligned = np.full(len(target_ids), fill, dtype=np.float32)
    aligned[found] = np.asarray(values, dtype=np.float32)[positions[found]]
    return aligned, found


def _count_behavior_support(seq_path: str | Path, num_items: int) -> np.ndarray:
    """Count non-padding item occurrences with bounded-memory Parquet batches."""
    import pyarrow.parquet as pq

    support = np.zeros(num_items, dtype=np.float64)
    parquet = pq.ParquetFile(seq_path)
    processed = 0
    for batch in parquet.iter_batches(batch_size=50_000, columns=["item_seq"]):
        sequence_array = batch.column(0)
        flat = sequence_array.values.to_numpy(zero_copy_only=False)
        valid = flat[(flat > 0) & (flat < num_items)].astype(np.int64, copy=False)
        support += np.bincount(valid, minlength=num_items)
        processed += len(sequence_array)
        if processed % 1_000_000 == 0:
            print(f"[CAGF] behavior support rows={processed:,}")
    return support.astype(np.float32)


def _load_multimodal_source(
    path: str | Path,
    target_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    table = pd.read_parquet(path)
    if "item_id" not in table.columns:
        raise ValueError(f"{path} missing item_id")
    source_ids = table["item_id"].to_numpy(dtype=np.int64)
    _positions(source_ids, target_ids, str(path))
    columns = [column for column in table.columns if column.startswith("item_emb_d128")]
    if not columns:
        raise ValueError(f"{path} contains no item_emb_d128* columns")
    variants = np.stack(
        [normalize_rows(to_mat(table[column], source=f"{path}:{column}")) for column in columns],
        axis=0,
    )
    mean_embedding = normalize_rows(variants.mean(axis=0))
    consistency = multimodal_consistency(variants)
    aligned, _ = align_embedding_matrix(
        source_ids,
        mean_embedding,
        target_ids,
        source=str(path),
        require_full_coverage=False,
    )
    aligned_consistency, found = _align_scalar(
        source_ids,
        consistency,
        target_ids,
        source=str(path),
    )
    available = found & (np.linalg.norm(aligned, axis=1) > 1e-8)
    return normalize_rows(aligned), available, aligned_consistency, columns


def load_cagf_sources(
    item_info_path: str | Path,
    behavior_emb_path: str | Path,
    item_emb_path: str | Path,
    *,
    behavior_seq_path: str | Path | None = "data/item_seq.parquet",
    rq_clusters: int = 5,
    rq_layers: int = 3,
) -> dict[str, object]:
    """Load and align four embedding sources plus their intrinsic confidence."""
    item_info = pd.read_parquet(item_info_path)
    required = {"item_id", "item_emb_d128"}
    missing = required - set(item_info.columns)
    if missing:
        raise ValueError(f"item_info missing columns: {sorted(missing)}")
    target_ids = item_info["item_id"].to_numpy(dtype=np.int64)
    if pd.Index(target_ids).has_duplicates:
        raise ValueError("item_info contains duplicate item_id values")

    official = normalize_rows(to_mat(item_info["item_emb_d128"], source=str(item_info_path)))
    quantizer = ResidualQuantizer(rq_clusters, rq_layers).fit(official)
    rq = quantizer.reconstruct(official)
    rq_confidence = np.clip((np.sum(official * rq, axis=1) + 1.0) * 0.5, 0.0, 1.0)

    behavior_table = pd.read_parquet(behavior_emb_path)
    behavior, _ = align_embeddings(
        behavior_table,
        target_ids,
        source=str(behavior_emb_path),
        require_full_coverage=False,
    )
    behavior_positions = _positions(
        behavior_table["item_id"].to_numpy(dtype=np.int64),
        target_ids,
        str(behavior_emb_path),
    )
    behavior_available = (behavior_positions >= 0) & (np.linalg.norm(behavior, axis=1) > 1e-8)
    support_source = "embedding_column" if "behavior_support" in behavior_table.columns else ""
    if support_source:
        raw_support, _ = _align_scalar(
            behavior_table["item_id"].to_numpy(dtype=np.int64),
            behavior_table["behavior_support"].to_numpy(dtype=np.float32),
            target_ids,
            source=str(behavior_emb_path),
        )
        behavior_confidence = robust_scale_support(raw_support)
    elif behavior_seq_path is not None and Path(behavior_seq_path).exists():
        raw_support = _count_behavior_support(behavior_seq_path, len(target_ids))
        behavior_confidence = robust_scale_support(raw_support)
        support_source = "sequence_frequency"
    else:
        behavior_confidence = np.full(len(target_ids), 0.5, dtype=np.float32)
    behavior_confidence *= behavior_available.astype(np.float32)

    mm, mm_available, mm_confidence, mm_columns = _load_multimodal_source(
        item_emb_path, target_ids
    )
    sources = np.stack([official, rq, normalize_rows(behavior), mm], axis=1).astype(np.float32)
    available = np.stack(
        [
            np.ones(len(target_ids), dtype=bool),
            np.ones(len(target_ids), dtype=bool),
            behavior_available,
            mm_available,
        ],
        axis=1,
    )
    intrinsic = np.stack(
        [
            np.ones(len(target_ids), dtype=np.float32),
            rq_confidence.astype(np.float32),
            behavior_confidence,
            mm_confidence.astype(np.float32) * mm_available.astype(np.float32),
        ],
        axis=1,
    )
    features, agreement = build_confidence_features(sources, available, intrinsic)
    if not support_source:
        intrinsic[:, 2] = agreement[:, 2] * behavior_available.astype(np.float32)
        features, agreement = build_confidence_features(sources, available, intrinsic)

    return {
        "item_info": item_info,
        "item_ids": target_ids,
        "sources": sources,
        "available": available,
        "intrinsic": intrinsic,
        "agreement": agreement,
        "features": features,
        "feature_names": feature_names(SOURCE_NAMES),
        "behavior_support_source": support_source or "cross_source_agreement_fallback",
        "mm_columns": mm_columns,
    }


def _fusion_loss(
    weights: torch.Tensor,
    sources: torch.Tensor,
    available: torch.Tensor,
    intrinsic: torch.Tensor,
    agreement: torch.Tensor,
    prior: torch.Tensor,
    *,
    anchor_weight: float,
    calibration_weight: float,
    entropy_weight: float,
    balance_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    available_float = available.float()
    fused = F.normalize(torch.einsum("bs,bsd->bd", weights, sources), dim=1)

    loo_numerator = sources.new_tensor(0.0)
    loo_denominator = sources.new_tensor(0.0)
    for target in range(sources.shape[1]):
        remaining = weights * available_float
        remaining = remaining.clone()
        remaining[:, target] = 0.0
        remaining_count = available.sum(dim=1) - available[:, target].long()
        eligible = available[:, target] & (remaining_count > 0)
        denominator = remaining.sum(dim=1, keepdim=True).clamp_min(1e-8)
        prediction = F.normalize(
            torch.einsum("bs,bsd->bd", remaining / denominator, sources), dim=1
        )
        cosine_loss = 1.0 - (prediction * sources[:, target, :]).sum(dim=1)
        reliability = intrinsic[:, target].clamp_min(0.05)
        sample_weight = reliability * eligible.float()
        loo_numerator = loo_numerator + (cosine_loss * sample_weight).sum()
        loo_denominator = loo_denominator + sample_weight.sum()
    loo = loo_numerator / loo_denominator.clamp_min(1.0)

    anchor = (1.0 - (fused * sources[:, 0, :]).sum(dim=1)).mean()
    pseudo_target = intrinsic * (0.25 + 0.75 * agreement) * available_float
    pseudo_target = pseudo_target / pseudo_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    calibration = -(pseudo_target * torch.log(weights.clamp_min(1e-8))).sum(dim=1).mean()

    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=1)
    entropy_floor = 0.65 * np.log(weights.shape[1])
    entropy_penalty = F.relu(weights.new_tensor(entropy_floor) - entropy).pow(2).mean()
    mean_weight = weights.mean(dim=0)
    balance = ((mean_weight - prior) ** 2).mean()

    total = (
        loo
        + anchor_weight * anchor
        + calibration_weight * calibration
        + entropy_weight * entropy_penalty
        + balance_weight * balance
    )
    return total, {
        "loss": total.detach(),
        "loo": loo.detach(),
        "anchor": anchor.detach(),
        "calibration": calibration.detach(),
        "entropy": entropy.mean().detach(),
        "balance": balance.detach(),
    }


def train_gate(
    features: np.ndarray,
    sources: np.ndarray,
    available: np.ndarray,
    intrinsic: np.ndarray,
    agreement: np.ndarray,
    *,
    seed: int = 20242025,
    hidden_dim: int = 64,
    dropout: float = 0.10,
    temperature: float = 0.80,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 4096,
    epochs: int = 80,
    patience: int = 12,
    val_fraction: float = 0.10,
    anchor_weight: float = 0.15,
    calibration_weight: float = 0.10,
    entropy_weight: float = 0.02,
    balance_weight: float = 0.03,
    prior: np.ndarray = DEFAULT_PRIOR,
    device: str = "auto",
) -> tuple[ConfidenceGate, np.ndarray, dict[str, object]]:
    _seed_everything(seed)
    torch_device = _resolve_device(device)
    n_items = len(features)
    if n_items < 10:
        raise ValueError("CAGF requires at least 10 items for a train/validation split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    val_size = max(1, int(round(n_items * val_fraction)))
    val_indices = order[:val_size]
    train_indices = order[val_size:]

    feature_mean = features[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = features[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    standardized = (features - feature_mean) / feature_std

    model = ConfidenceGate(features.shape[1], hidden_dim, sources.shape[1], dropout).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    prior_tensor = torch.as_tensor(prior / prior.sum(), dtype=torch.float32, device=torch_device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    def evaluate(indices: np.ndarray) -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                x = torch.as_tensor(standardized[batch], device=torch_device)
                src = torch.as_tensor(sources[batch], device=torch_device)
                avail = torch.as_tensor(available[batch], device=torch_device)
                conf = torch.as_tensor(intrinsic[batch], device=torch_device)
                agree = torch.as_tensor(agreement[batch], device=torch_device)
                weights = model(x, conf, avail, temperature)
                _, metrics = _fusion_loss(
                    weights,
                    src,
                    avail,
                    conf,
                    agree,
                    prior_tensor,
                    anchor_weight=anchor_weight,
                    calibration_weight=calibration_weight,
                    entropy_weight=entropy_weight,
                    balance_weight=balance_weight,
                )
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * len(batch)
                count += len(batch)
        return {name: value / max(count, 1) for name, value in totals.items()}

    for epoch in range(1, epochs + 1):
        model.train()
        train_order = rng.permutation(train_indices)
        for start in range(0, len(train_order), batch_size):
            batch = train_order[start:start + batch_size]
            x = torch.as_tensor(standardized[batch], device=torch_device)
            src = torch.as_tensor(sources[batch], device=torch_device)
            avail = torch.as_tensor(available[batch], device=torch_device)
            conf = torch.as_tensor(intrinsic[batch], device=torch_device)
            agree = torch.as_tensor(agreement[batch], device=torch_device)
            optimizer.zero_grad(set_to_none=True)
            weights = model(x, conf, avail, temperature)
            loss, _ = _fusion_loss(
                weights,
                src,
                avail,
                conf,
                agree,
                prior_tensor,
                anchor_weight=anchor_weight,
                calibration_weight=calibration_weight,
                entropy_weight=entropy_weight,
                balance_weight=balance_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        train_metrics = evaluate(train_indices)
        val_metrics = evaluate(val_indices)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_loo": val_metrics["loo"],
        })
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[CAGF] epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} loo={val_metrics['loo']:.6f}"
            )
        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[CAGF] early stop at epoch {epoch}; best epoch={best_epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_weights = []
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            batch = slice(start, min(start + batch_size, n_items))
            x = torch.as_tensor(standardized[batch], device=torch_device)
            conf = torch.as_tensor(intrinsic[batch], device=torch_device)
            avail = torch.as_tensor(available[batch], device=torch_device)
            weights = model(x, conf, avail, temperature)
            all_weights.append(weights.cpu().numpy())

    metadata = {
        "seed": seed,
        "device": str(torch_device),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "history": history,
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "val_fraction": val_fraction,
            "anchor_weight": anchor_weight,
            "calibration_weight": calibration_weight,
            "entropy_weight": entropy_weight,
            "balance_weight": balance_weight,
            "prior": (prior / prior.sum()).tolist(),
        },
    }
    return model, np.concatenate(all_weights, axis=0).astype(np.float32), metadata


def build_cagf_embedding(
    item_info_path: str | Path,
    out_path: str | Path,
    *,
    behavior_emb_path: str | Path = "embeddings/behavior_item_emb_128.parquet",
    behavior_seq_path: str | Path | None = "data/item_seq.parquet",
    item_emb_path: str | Path = "data/item_emb.parquet",
    diagnostics_path: str | Path = "results/cagf_item_weights.parquet",
    summary_path: str | Path = "results/cagf_summary.json",
    checkpoint_path: str | Path = "checkpoints/cagf_gate.pt",
    rq_clusters: int = 5,
    rq_layers: int = 3,
    **training_kwargs,
) -> dict[str, object]:
    bundle = load_cagf_sources(
        item_info_path,
        behavior_emb_path,
        item_emb_path,
        behavior_seq_path=behavior_seq_path,
        rq_clusters=rq_clusters,
        rq_layers=rq_layers,
    )
    model, weights, training_metadata = train_gate(
        bundle["features"],
        bundle["sources"],
        bundle["available"],
        bundle["intrinsic"],
        bundle["agreement"],
        **training_kwargs,
    )
    fused = normalize_rows(np.einsum("ns,nsd->nd", weights, bundle["sources"]))
    if not np.isfinite(fused).all() or fused.shape[1] != 128:
        raise RuntimeError(f"Invalid CAGF output shape/values: {fused.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = bundle["item_info"].copy()
    out_df["item_emb_d128"] = [row.tolist() for row in fused]
    out_df.to_parquet(out_path, index=False)

    entropy = -(weights * np.log(np.maximum(weights, 1e-8))).sum(axis=1)
    diagnostics = pd.DataFrame({"item_id": bundle["item_ids"], "gate_entropy": entropy})
    for index, source in enumerate(SOURCE_NAMES):
        diagnostics[f"available_{source}"] = bundle["available"][:, index].astype(np.int8)
        diagnostics[f"weight_{source}"] = weights[:, index]
        diagnostics[f"confidence_{source}"] = bundle["intrinsic"][:, index]
        diagnostics[f"agreement_{source}"] = bundle["agreement"][:, index]
    diagnostics_path = Path(diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(diagnostics_path, index=False)

    weight_summary = {}
    for index, source in enumerate(SOURCE_NAMES):
        values = weights[:, index]
        weight_summary[source] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }
    summary = {
        "method": "Confidence-Aware Gated Fusion (CAGF)",
        "objective": "self-supervised leave-one-source-out reconstruction",
        "num_items": len(fused),
        "sources": SOURCE_NAMES,
        "feature_names": bundle["feature_names"],
        "behavior_support_source": bundle["behavior_support_source"],
        "multimodal_columns": bundle["mm_columns"],
        "source_available_items": {
            source: int(bundle["available"][:, index].sum())
            for index, source in enumerate(SOURCE_NAMES)
        },
        "weight_summary": weight_summary,
        "mean_gate_entropy": float(entropy.mean()),
        "mean_itemwise_weight_std": float(weights.std(axis=0).mean()),
        "training": training_metadata,
        "output": str(out_path),
        "diagnostics": str(diagnostics_path),
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "source_names": SOURCE_NAMES,
            "feature_names": bundle["feature_names"],
            "training": training_metadata,
        },
        checkpoint_path,
    )
    print(f"[OK] CAGF embedding: {out_path} {fused.shape}")
    print(f"[OK] diagnostics: {diagnostics_path}")
    print(f"[OK] summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item_info", default="data/MicroLens_1M_x1/item_info.parquet")
    parser.add_argument("--behavior_emb", default="embeddings/behavior_item_emb_128.parquet")
    parser.add_argument("--behavior_seq", default="data/item_seq.parquet")
    parser.add_argument("--item_emb", default="data/item_emb.parquet")
    parser.add_argument("--out", default="data/MicroLens_1M_x1/item_info_cagf.parquet")
    parser.add_argument("--diagnostics", default="results/cagf_item_weights.parquet")
    parser.add_argument("--summary", default="results/cagf_summary.json")
    parser.add_argument("--checkpoint", default="checkpoints/cagf_gate.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20242025)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    build_cagf_embedding(
        args.item_info,
        args.out,
        behavior_emb_path=args.behavior_emb,
        behavior_seq_path=args.behavior_seq,
        item_emb_path=args.item_emb,
        diagnostics_path=args.diagnostics,
        summary_path=args.summary,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
