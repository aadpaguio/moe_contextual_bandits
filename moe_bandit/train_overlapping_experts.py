from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moe_bandit.experts import Expert


@dataclass(frozen=True)
class OverlapTrainingStats:
    """Diagnostics for overlapping-expert training."""

    mixture_weights: np.ndarray
    sampled_cluster_proportions: np.ndarray
    own_cluster_accuracy: np.ndarray
    cross_cluster_accuracy: np.ndarray


def _get_default_device() -> torch.device:
    if os.environ.get("MOE_BANDIT_USE_MPS", "0") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cyclic_mixture_weights(K: int, overlap_strength: float = 0.5) -> np.ndarray:
    """
    Build a deterministic KxK expert-by-cluster mixture matrix.

    overlap_strength controls how much mass stays on each expert's anchor cluster.
    Remaining mass is distributed to neighboring clusters in a cyclic pattern.
    """
    if K < 2:
        raise ValueError("K must be at least 2.")
    if not (0.0 < overlap_strength < 1.0):
        raise ValueError("overlap_strength must be in (0, 1).")

    W = np.zeros((K, K), dtype=np.float64)
    for expert_idx in range(K):
        W[expert_idx, expert_idx] = overlap_strength
        remaining = 1.0 - overlap_strength
        if K == 2:
            W[expert_idx, (expert_idx + 1) % K] = remaining
            continue
        W[expert_idx, (expert_idx + 1) % K] = 0.70 * remaining
        W[expert_idx, (expert_idx + 2) % K] = 0.30 * remaining
    W /= np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    return W


def train_overlapping_experts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_id_train: np.ndarray,
    K: int,
    d: int,
    *,
    mixture_weights: np.ndarray | None = None,
    overlap_strength: float = 0.5,
    samples_per_expert: int | None = None,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
) -> tuple[list[Expert], OverlapTrainingStats]:
    """
    Train K frozen experts on overlapping mixtures of cluster-conditioned data.
    """
    if X_train.ndim != 2 or X_train.shape[1] != d:
        raise ValueError(f"X_train must have shape (n_samples, {d}).")
    if y_train.ndim != 1 or cluster_id_train.ndim != 1:
        raise ValueError("y_train and cluster_id_train must be 1D.")
    if not (len(X_train) == len(y_train) == len(cluster_id_train)):
        raise ValueError("X_train, y_train, and cluster_id_train lengths must match.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must satisfy 0.0 <= label_smoothing < 1.0.")

    if mixture_weights is None:
        W = build_cyclic_mixture_weights(K=K, overlap_strength=overlap_strength)
    else:
        W = np.asarray(mixture_weights, dtype=np.float64)
        if W.shape != (K, K):
            raise ValueError(f"mixture_weights must have shape ({K}, {K}).")
        if np.any(W < 0):
            raise ValueError("mixture_weights must be non-negative.")
        row_sum = W.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 0):
            raise ValueError("Each expert row in mixture_weights must have positive mass.")
        W = W / row_sum

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _get_default_device()

    cluster_indices = [np.flatnonzero(cluster_id_train == c) for c in range(K)]
    for c, idx in enumerate(cluster_indices):
        if idx.size == 0:
            raise ValueError(f"No training points for cluster {c}.")

    n_default = len(X_train) // K
    n_per_expert = int(n_default if samples_per_expert is None else samples_per_expert)
    if n_per_expert <= 0:
        raise ValueError("samples_per_expert must be positive when provided.")

    experts: list[Expert] = []
    sampled_cluster_props = np.zeros((K, K), dtype=np.float64)
    own_cluster_acc = np.zeros(K, dtype=np.float64)
    cross_cluster_acc = np.zeros(K, dtype=np.float64)

    x_all = torch.as_tensor(X_train, dtype=torch.float32, device=device)

    for expert_idx in range(K):
        rng = np.random.default_rng(seed + 10_000 + expert_idx)
        cluster_counts = rng.multinomial(n_per_expert, W[expert_idx])
        sampled_cluster_props[expert_idx] = cluster_counts / float(n_per_expert)

        selected_parts: list[np.ndarray] = []
        for c in range(K):
            count = int(cluster_counts[c])
            if count <= 0:
                continue
            base_idx = cluster_indices[c]
            chosen = rng.choice(base_idx, size=count, replace=count > base_idx.size)
            selected_parts.append(chosen)
        selected_idx = np.concatenate(selected_parts)
        selected_idx = selected_idx[rng.permutation(selected_idx.size)]

        x_t = torch.as_tensor(X_train[selected_idx], dtype=torch.float32)
        y_t = torch.as_tensor(y_train[selected_idx], dtype=torch.long)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 20_000 + expert_idx),
        )

        torch.manual_seed(seed + 30_000 + expert_idx)
        model = Expert(d=d, K=K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            model.eval()
            own_mask = cluster_id_train == expert_idx
            own_x = x_all[own_mask]
            own_y = torch.as_tensor(y_train[own_mask], dtype=torch.long, device=device)
            own_pred = model(own_x).argmax(dim=1)
            own_cluster_acc[expert_idx] = float((own_pred == own_y).float().mean().item())

            cross_mask = cluster_id_train != expert_idx
            cross_x = x_all[cross_mask]
            cross_y = torch.as_tensor(y_train[cross_mask], dtype=torch.long, device=device)
            cross_pred = model(cross_x).argmax(dim=1)
            cross_cluster_acc[expert_idx] = float((cross_pred == cross_y).float().mean().item())

        for p in model.parameters():
            p.requires_grad_(False)
        experts.append(model)

    stats = OverlapTrainingStats(
        mixture_weights=W,
        sampled_cluster_proportions=sampled_cluster_props,
        own_cluster_accuracy=own_cluster_acc,
        cross_cluster_accuracy=cross_cluster_acc,
    )
    return experts, stats
