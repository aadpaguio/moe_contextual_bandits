from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Expert(nn.Module):
    """Small MLP: d -> 64 -> 64 -> K, linear output (logits)."""

    def __init__(self, d: int, K: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class _TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 64
    seed: int = 0


def _set_training_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    return x_t, y_t


def _get_default_device() -> torch.device:
    # Keep CPU as the default for reproducibility/stability; opt in to MPS explicitly.
    if os.environ.get("MOE_BANDIT_USE_MPS", "0") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_experts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_id_train: np.ndarray,
    K: int,
    d: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
    contamination: float = 0.2,
) -> list[Expert]:
    """
    Train K experts, one per cluster. Returns frozen experts in eval mode.

    contamination: Fraction of each expert's training set drawn uniformly from
        other clusters (with true labels). 0.0 means each expert sees only its
        own cluster (degenerate — produces overconfident experts). Recommended: 0.2.
    """
    if X_train.ndim != 2 or X_train.shape[1] != d:
        raise ValueError(f"X_train must have shape (n_samples, {d}).")
    if y_train.ndim != 1 or cluster_id_train.ndim != 1:
        raise ValueError("y_train and cluster_id_train must be 1D.")
    if not (len(X_train) == len(y_train) == len(cluster_id_train)):
        raise ValueError("X_train, y_train, and cluster_id_train lengths must match.")
    if K <= 1:
        raise ValueError("K must be at least 2.")
    if not (0.0 <= contamination < 1.0):
        raise ValueError("contamination must satisfy 0.0 <= contamination < 1.0.")

    cfg = _TrainConfig(epochs=epochs, lr=lr, batch_size=batch_size, seed=seed)
    _set_training_seeds(cfg.seed)
    device = _get_default_device()

    experts: list[Expert] = []
    for expert_idx in range(K):
        mask = cluster_id_train == expert_idx
        if not np.any(mask):
            raise ValueError(f"No training points for cluster {expert_idx}.")

        own_indices = np.flatnonzero(mask)
        n_i = own_indices.size
        rng = np.random.default_rng(cfg.seed + 2000 + expert_idx)

        if contamination == 0.0:
            selected_indices = own_indices
        else:
            contam_count = int(round(contamination * n_i))
            own_count = n_i - contam_count

            selected_own = rng.choice(own_indices, size=own_count, replace=False)
            other_indices = np.flatnonzero(~mask)
            if other_indices.size == 0:
                raise ValueError("Contamination requested but no other-cluster points available.")
            selected_other = rng.choice(
                other_indices,
                size=contam_count,
                replace=contam_count > other_indices.size,
            )
            selected_indices = np.concatenate([selected_own, selected_other])
            selected_indices = selected_indices[rng.permutation(selected_indices.size)]

        x_cluster, y_cluster = _to_tensors(X_train[selected_indices], y_train[selected_indices])
        dataset = TensorDataset(x_cluster, y_cluster)
        generator = torch.Generator().manual_seed(cfg.seed + expert_idx)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=generator,
        )

        # Keep model initialization independent from DataLoader RNG progression.
        torch.manual_seed(cfg.seed + 1000 + expert_idx)
        model = Expert(d=d, K=K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(cfg.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        experts.append(model)

    return experts


def expert_reward_matrix(
    experts: list[Expert],
    X: np.ndarray,
    y: np.ndarray,
    clip_eps: float = 1e-3,
) -> np.ndarray:
    """
    Compute full reward matrix R[t, i] = log(clipped p_i(y_t | x_t)).
    """
    if len(experts) == 0:
        raise ValueError("experts must be non-empty.")
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be 1D with the same length as X.")
    if not (0.0 < clip_eps < 0.5):
        raise ValueError("clip_eps must be in (0, 0.5).")

    n_samples = X.shape[0]
    K = len(experts)
    device = next(experts[0].parameters()).device
    if any(next(expert.parameters()).device != device for expert in experts):
        raise ValueError("All experts must be on the same device.")
    x_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    rewards = np.zeros((n_samples, K), dtype=np.float64)

    with torch.no_grad():
        for i, expert in enumerate(experts):
            expert.eval()
            logits = expert(x_t)
            probs = torch.softmax(logits, dim=1)
            probs = torch.clamp(probs, min=clip_eps, max=1.0 - clip_eps)
            chosen = probs[torch.arange(n_samples, device=device), y_t]
            rewards[:, i] = torch.log(chosen).cpu().numpy().astype(np.float64)

    return rewards
