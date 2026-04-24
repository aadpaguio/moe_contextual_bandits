from __future__ import annotations

import numpy as np


def _build_cluster_means(K, d, cluster_sep, cluster_std, rng):
    separation_scale = cluster_sep * cluster_std
    # Always use random unit vectors, regardless of d vs K relationship
    raw = rng.normal(size=(K, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / np.clip(norms, 1e-12, None)
    return separation_scale * unit


def generate_synthetic_data(
    n_samples: int,
    K: int,
    d: int,
    cluster_sep: float,
    cluster_std: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic contextual-bandit data with Gaussian clusters.

    Args:
        n_samples: Number of data points to draw.
        K: Number of clusters (and default number of classes/experts).
        d: Ambient context dimension.
        cluster_sep: Separation between cluster means in units of cluster_std.
            cluster_sep=3 means centers are 3σ apart along orthogonal directions.
        cluster_std: Shared isotropic within-cluster standard deviation.
        seed: Local RNG seed for deterministic generation.

    Returns:
        X: Shape (n_samples, d), context vectors.
        y: Shape (n_samples,), labels in [0, K). In the base setup y == cluster_id.
        cluster_id: Shape (n_samples,), sampled cluster indices in [0, K).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if K <= 1:
        raise ValueError("K must be at least 2.")
    if d <= 0:
        raise ValueError("d must be positive.")
    if cluster_std <= 0:
        raise ValueError("cluster_std must be positive.")
    if cluster_sep < 0:
        raise ValueError("cluster_sep must be non-negative.")

    rng = np.random.default_rng(seed)
    means = _build_cluster_means(
        K=K,
        d=d,
        cluster_sep=cluster_sep,
        cluster_std=cluster_std,
        rng=rng,
    )

    cluster_id = rng.integers(low=0, high=K, size=n_samples)
    noise = rng.normal(scale=cluster_std, size=(n_samples, d))
    X = means[cluster_id] + noise

    # Base setup keeps labels and clusters aligned; API remains flexible.
    y = cluster_id.copy()
    return X.astype(np.float64), y.astype(np.int64), cluster_id.astype(np.int64)
