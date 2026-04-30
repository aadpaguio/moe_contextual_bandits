from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moe_bandit.experts import Expert
from moe_bandit.torch_device import default_torch_device


def _set_training_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class JointTrainingStats:
    """Epoch-wise diagnostics from `train_joint_moe`."""

    history_total_loss: list[float] = field(default_factory=list)
    history_ce_loss: list[float] = field(default_factory=list)
    history_load_loss: list[float] = field(default_factory=list)
    history_train_acc: list[float] = field(default_factory=list)
    history_val_acc: list[float] = field(default_factory=list)
    history_lr: list[float] = field(default_factory=list)
    """Per epoch: mean gate weight to each expert, shape (K,) — one array per epoch."""
    history_gate_means: list[np.ndarray] = field(default_factory=list)
    epochs_run: int = 0
    """Number of epochs actually trained (may be less than ``epochs`` if early stopping fired)."""
    early_stopped: bool = False
    best_val_acc: float = 0.0
    best_epoch_1based: int = 0
    """Checkpoint with best validation accuracy (restored into returned experts when ES enabled)."""
    final_pooled_train_acc: float = 0.0
    final_pooled_val_acc: float = 0.0
    collapse_warning_early: bool = False
    """True if any expert mean gate weight was below 0.05 by end of epoch 5 (1-based)."""
    collapse_warning_messages: list[str] = field(default_factory=list)
    final_gate_means: np.ndarray | None = None
    """Shape (K,). Mean gate weights after last training epoch (train split, full pass)."""


class MoERouter(nn.Module):
    """Linear `d -> K`, or small MLP `d -> hidden -> K` when `hidden` is set."""

    def __init__(self, d: int, K: int, hidden: int | None = None) -> None:
        super().__init__()
        if hidden is None or hidden <= 0:
            self.net = nn.Linear(d, K)
        else:
            self.net = nn.Sequential(
                nn.Linear(d, hidden),
                nn.ReLU(),
                nn.Linear(hidden, K),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointMoE(nn.Module):
    """K experts + router; forward returns mixed logits and gate weights."""

    def __init__(
        self,
        d: int,
        K_experts: int,
        n_classes: int,
        router_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.K = K_experts
        self.experts = nn.ModuleList(
            Expert(d=d, K=n_classes) for _ in range(K_experts)
        )
        self.router = MoERouter(d=d, K=K_experts, hidden=router_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_per_expert = torch.stack([e(x) for e in self.experts], dim=1)
        gate_logits = self.router(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        mixed = torch.einsum("bk,bkc->bc", gate_weights, logits_per_expert)
        return mixed, logits_per_expert, gate_weights


def _load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Switch-style soft routing load loss: K * sum_i f_i^2, f_i = batch mean of gate i."""
    f = gate_weights.mean(dim=0)
    K = f.shape[0]
    return K * (f * f).sum()


def _accuracy_from_mixed(mixed_logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = mixed_logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def _eval_mixed_accuracy(
    model: JointMoE,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    n = x.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        batch_x = x[start : start + batch_size].to(device)
        batch_y = y[start : start + batch_size].to(device)
        mixed, _, _ = model(batch_x)
        correct += int((mixed.argmax(dim=1) == batch_y).sum().item())
    return correct / max(n, 1)


@torch.no_grad()
def _mean_gate_weights_train_pass(
    model: JointMoE,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    K = model.K
    total_w = torch.zeros(K, device=device)
    n = x.shape[0]
    seen = 0
    for start in range(0, n, batch_size):
        batch_x = x[start : start + batch_size].to(device)
        _, _, gw = model(batch_x)
        bs = batch_x.shape[0]
        total_w += gw.sum(dim=0)
        seen += bs
    mean_w = (total_w / float(seen)).cpu().numpy()
    return mean_w.astype(np.float64)


def train_joint_moe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_id_train: np.ndarray,
    K: int,
    d: int,
    *,
    n_classes: int | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    lr_min: float = 1e-5,
    cosine_decay: bool = True,
    batch_size: int = 64,
    seed: int = 0,
    alpha_load: float = 0.001,
    val_frac: float = 0.1,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 0.0,
    router: Literal["linear", "mlp"] = "linear",
    router_mlp_hidden: int = 16,
    contamination: float = 0.2,
) -> tuple[list[Expert], JointTrainingStats]:
    """
    Jointly train K experts with a softmax router on the full pooled training set.

    Mixture output is a weighted sum of expert *logits* (standard MoE). Training-time
    router is discarded after training; returned experts match `Expert` used elsewhere.

    `cluster_id_train` and `contamination` are accepted for API parity with
    `train_experts` but are ignored: joint training always uses all rows of
    ``(X_train, y_train)`` pooled.

    Args:
        X_train: Shape (n, d).
        y_train: Shape (n,), class indices in [0, n_classes).
        cluster_id_train: Ignored for joint training (kept for call-site compatibility).
        K: Number of experts (= gate dimension).
        d: Context dimension.
        n_classes: Number of classes per expert output; defaults to ``K``.
        epochs: Maximum training epochs (default 200). Use early stopping to finish sooner.
        lr: Initial Adam learning rate (default ``1e-3``).
        lr_min: Floor LR for cosine decay (default ``1e-5``).
        cosine_decay: If True (default), cosine anneal LR from ``lr`` to ``lr_min`` over
            ``epochs`` steps (one ``scheduler.step()`` per epoch).
        alpha_load: Weight on load-balancing loss (default ``1e-3``; weaker than ``1e-2``
            so the router can specialize more while still penalizing collapse).
        val_frac: Fraction of data for pooled holdout metrics (stratified by shuffled index).
        early_stopping_patience: Stop if validation accuracy does not improve by at least
            ``early_stopping_min_delta`` for this many epochs. ``None`` disables early stopping.
        early_stopping_min_delta: Minimum val-acc improvement to reset the patience counter.
        router: ``"linear"`` (default) or ``"mlp"`` (``d -> 16 -> K``).
        router_mlp_hidden: Hidden size when ``router="mlp"``.
        contamination: Ignored (compatibility only).

    Returns:
        Frozen experts in eval mode, and ``JointTrainingStats`` with per-epoch logs.
    """
    del cluster_id_train, contamination

    if X_train.ndim != 2 or X_train.shape[1] != d:
        raise ValueError(f"X_train must have shape (n_samples, {d}).")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match.")
    if K <= 1:
        raise ValueError("K must be at least 2.")
    if not (0.0 < val_frac < 0.5):
        raise ValueError("val_frac must be in (0, 0.5).")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if lr <= 0 or lr_min <= 0:
        raise ValueError("lr and lr_min must be positive.")
    if cosine_decay and lr_min >= lr:
        raise ValueError("lr_min must be less than lr when cosine_decay is True.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive or None.")

    n_cls = int(K if n_classes is None else n_classes)
    if y_train.max() >= n_cls or y_train.min() < 0:
        raise ValueError(f"y_train must be in [0, {n_cls}).")

    _set_training_seeds(seed)
    device = default_torch_device()

    n_total = X_train.shape[0]
    rng_split = np.random.default_rng(seed + 42)
    perm = rng_split.permutation(n_total)
    n_val = max(1, int(round(val_frac * n_total)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.size == 0:
        raise ValueError("Train split is empty; increase data size or lower val_frac.")

    x_full = torch.as_tensor(X_train, dtype=torch.float32)
    y_full = torch.as_tensor(y_train, dtype=torch.long)
    x_tr = x_full[train_idx]
    y_tr = y_full[train_idx]
    x_va = x_full[val_idx]
    y_va = y_full[val_idx]

    hidden = None if router == "linear" else router_mlp_hidden
    torch.manual_seed(seed + 1000)
    model = JointMoE(d=d, K_experts=K, n_classes=n_cls, router_hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(x_tr, y_tr)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 3000),
    )

    stats = JointTrainingStats()
    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch_1based = 0
    epochs_without_improve = 0

    epoch = 0
    while epoch < epochs:
        model.train()
        sum_total = 0.0
        sum_ce = 0.0
        sum_load = 0.0
        n_batches = 0
        gate_sum = torch.zeros(K, dtype=torch.float64)
        n_gate_samples = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            mixed, _, gate_w = model(batch_x)
            ce = criterion(mixed, batch_y)
            load = _load_balancing_loss(gate_w)
            loss = ce + alpha_load * load
            loss.backward()
            optimizer.step()

            sum_total += float(loss.item())
            sum_ce += float(ce.item())
            sum_load += float(load.item())
            n_batches += 1

            with torch.no_grad():
                gw_cpu = gate_w.detach().mean(dim=0).cpu()
                gate_sum += gw_cpu.double() * batch_x.shape[0]
                n_gate_samples += batch_x.shape[0]

        mean_gate_epoch = (gate_sum / float(n_gate_samples)).numpy()
        stats.history_gate_means.append(mean_gate_epoch.copy())
        stats.history_total_loss.append(sum_total / max(n_batches, 1))
        stats.history_ce_loss.append(sum_ce / max(n_batches, 1))
        stats.history_load_loss.append(sum_load / max(n_batches, 1))

        tr_acc = _eval_mixed_accuracy(model, x_tr, y_tr, device, batch_size)
        va_acc = _eval_mixed_accuracy(model, x_va, y_va, device, batch_size)
        stats.history_train_acc.append(tr_acc)
        stats.history_val_acc.append(va_acc)
        stats.history_lr.append(float(optimizer.param_groups[0]["lr"]))

        improved = va_acc > best_val_acc + early_stopping_min_delta
        if improved:
            best_val_acc = va_acc
            best_epoch_1based = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        elif early_stopping_patience is not None:
            epochs_without_improve += 1

        if scheduler is not None:
            scheduler.step()

        epoch_1based = epoch + 1
        if epoch_1based == 5:
            if np.any(mean_gate_epoch < 0.05):
                stats.collapse_warning_early = True
                bad = np.flatnonzero(mean_gate_epoch < 0.05).tolist()
                msg = (
                    f"[joint MoE] COLLAPSE WARNING (epoch {epoch_1based}): "
                    f"experts with mean gate < 0.05: {bad}; "
                    f"gate_means={mean_gate_epoch.round(4).tolist()}"
                )
                stats.collapse_warning_messages.append(msg)
                print(msg)

        epoch += 1
        if (
            early_stopping_patience is not None
            and epochs_without_improve >= early_stopping_patience
            and epoch >= 1
        ):
            stats.early_stopped = True
            break

    stats.epochs_run = epoch
    stats.best_val_acc = float(best_val_acc)
    stats.best_epoch_1based = int(best_epoch_1based)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    stats.final_pooled_val_acc = float(_eval_mixed_accuracy(model, x_va, y_va, device, batch_size))
    stats.final_pooled_train_acc = float(_eval_mixed_accuracy(model, x_tr, y_tr, device, batch_size))
    stats.final_gate_means = _mean_gate_weights_train_pass(model, x_tr, device, batch_size)

    stop_note = f"early stop after {stats.epochs_run} epochs" if stats.early_stopped else "finished max epochs"
    print(
        "[joint MoE] training complete:",
        stop_note + ";",
        f"best val acc={stats.best_val_acc:.4f} (epoch {stats.best_epoch_1based}),",
        f"checkpoint train acc={stats.final_pooled_train_acc:.4f},",
        f"checkpoint val acc={stats.final_pooled_val_acc:.4f},",
        f"gate means={np.round(stats.final_gate_means, 4).tolist()}",
    )

    experts_out: list[Expert] = []
    for i in range(K):
        ex = model.experts[i]
        ex.eval()
        for p in ex.parameters():
            p.requires_grad_(False)
        experts_out.append(ex)

    return experts_out, stats
