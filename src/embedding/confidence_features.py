"""Pure NumPy reliability features for confidence-aware embedding fusion."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, eps), dtype=np.float32)


def robust_scale_support(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Map non-negative item support to [0, 1] without letting hubs dominate."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Support must be a finite, non-negative vector")
    logged = np.log1p(values)
    positive = logged[logged > 0]
    if not len(positive):
        return np.zeros_like(logged)
    scale = float(np.percentile(positive, percentile))
    return np.clip(logged / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)


def multimodal_consistency(variants: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Mean pairwise cosine agreement among multimodal embedding variants."""
    variants = np.asarray(variants, dtype=np.float32)
    if variants.ndim != 3:
        raise ValueError(f"Expected (n_variants, n_items, dim), got {variants.shape}")
    if variants.shape[0] == 0:
        raise ValueError("At least one multimodal variant is required")
    normalized = np.stack([normalize_rows(part, eps) for part in variants], axis=0)
    if variants.shape[0] == 1:
        return np.ones(variants.shape[1], dtype=np.float32)
    scores = []
    for left, right in combinations(range(variants.shape[0]), 2):
        cosine = np.sum(normalized[left] * normalized[right], axis=1)
        scores.append(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0))
    return np.mean(np.stack(scores, axis=1), axis=1).astype(np.float32)


def feature_names(source_names: list[str]) -> list[str]:
    names = []
    for group in ("available", "intrinsic", "agreement", "disagreement", "official_similarity"):
        names.extend(f"{group}_{source}" for source in source_names)
    names.extend(
        f"pair_cosine_{source_names[left]}_{source_names[right]}"
        for left, right in combinations(range(len(source_names)), 2)
    )
    return names


def build_confidence_features(
    sources: np.ndarray,
    available: np.ndarray,
    intrinsic_confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-item features from source reliability and cross-source geometry.

    Args:
        sources: ``(n_items, n_sources, dim)`` embedding tensor.
        available: ``(n_items, n_sources)`` source-presence mask.
        intrinsic_confidence: ``(n_items, n_sources)`` confidence in ``[0, 1]``.

    Returns:
        A feature matrix and the per-source mean agreement matrix.
    """
    sources = np.asarray(sources, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    intrinsic = np.asarray(intrinsic_confidence, dtype=np.float32)
    if sources.ndim != 3:
        raise ValueError(f"Expected sources with 3 dimensions, got {sources.shape}")
    n_items, n_sources, _ = sources.shape
    if available.shape != (n_items, n_sources) or intrinsic.shape != (n_items, n_sources):
        raise ValueError("Availability and confidence shapes must match source axes")
    if not np.isfinite(sources).all() or not np.isfinite(intrinsic).all():
        raise ValueError("Sources and confidence values must be finite")
    if ((intrinsic < 0) | (intrinsic > 1)).any():
        raise ValueError("Intrinsic confidence must be in [0, 1]")

    normalized = np.stack(
        [normalize_rows(sources[:, index, :]) for index in range(n_sources)], axis=1
    )
    cosine = np.einsum("nsd,ntd->nst", normalized, normalized, optimize=True)
    pair_mask = available[:, :, None] & available[:, None, :]
    diagonal = np.eye(n_sources, dtype=bool)[None, :, :]
    other_mask = pair_mask & ~diagonal
    counts = other_mask.sum(axis=2)
    raw_agreement = np.divide(
        (cosine * other_mask).sum(axis=2),
        np.maximum(counts, 1),
        dtype=np.float32,
    )
    agreement = np.clip((raw_agreement + 1.0) * 0.5, 0.0, 1.0)
    agreement[counts == 0] = 0.0

    centered = cosine - raw_agreement[:, :, None]
    variance = np.divide(
        (centered * centered * other_mask).sum(axis=2),
        np.maximum(counts, 1),
        dtype=np.float32,
    )
    disagreement = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    disagreement[counts == 0] = 0.0

    official_similarity = np.clip((cosine[:, :, 0] + 1.0) * 0.5, 0.0, 1.0)
    official_similarity *= (available & available[:, :1]).astype(np.float32)

    pair_features = []
    for left, right in combinations(range(n_sources), 2):
        valid = pair_mask[:, left, right]
        mapped = np.clip((cosine[:, left, right] + 1.0) * 0.5, 0.0, 1.0)
        pair_features.append(np.where(valid, mapped, 0.0))
    pair_matrix = (
        np.stack(pair_features, axis=1).astype(np.float32)
        if pair_features
        else np.empty((n_items, 0), dtype=np.float32)
    )

    features = np.concatenate(
        [
            available.astype(np.float32),
            intrinsic,
            agreement.astype(np.float32),
            disagreement,
            official_similarity.astype(np.float32),
            pair_matrix,
        ],
        axis=1,
    )
    return features.astype(np.float32), agreement.astype(np.float32)
