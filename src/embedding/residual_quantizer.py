"""多层残差 K-Means 量化，用于物品嵌入离线去噪与重建（Task 1）。"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class ResidualQuantizer:
    def __init__(self, num_clusters=5, num_layers=3, random_state=20242025):
        if num_clusters < 1 or num_layers < 1:
            raise ValueError("num_clusters and num_layers must both be positive")
        self.num_clusters = num_clusters
        self.num_layers = num_layers
        self.random_state = random_state
        self.codebooks = []

    def fit(self, X: np.ndarray):
        X = self._validate_matrix(X)
        if len(X) < self.num_clusters:
            raise ValueError(
                f"num_clusters={self.num_clusters} exceeds sample count={len(X)}"
            )
        residual = X.copy()
        self.codebooks = []
        for _ in range(self.num_layers):
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                n_init="auto",
                random_state=self.random_state,
            )
            kmeans.fit(residual)
            self.codebooks.append(kmeans.cluster_centers_.astype(np.float32))
            residual = residual - kmeans.cluster_centers_[kmeans.labels_]
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("ResidualQuantizer must be fitted before reconstruct()")
        X = self._validate_matrix(X)
        if X.shape[1] != self.codebooks[0].shape[1]:
            raise ValueError(
                f"Embedding dimension {X.shape[1]} does not match fitted dimension "
                f"{self.codebooks[0].shape[1]}"
            )
        residual = X.copy()
        recon = np.zeros_like(residual)
        for centers in self.codebooks:
            dists = np.linalg.norm(residual[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            assigned = centers[labels]
            recon += assigned
            residual = residual - assigned
        return normalize(recon, norm="l2", axis=1).astype(np.float32)

    @staticmethod
    def _validate_matrix(X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"Expected a non-empty 2D matrix, got shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding matrix contains NaN or infinite values")
        return matrix
