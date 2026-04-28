from __future__ import annotations

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _default_device() -> torch.device:
    # Keep CPU as the default for reproducibility/stability; opt in to MPS explicitly.
    if os.environ.get("MOE_BANDIT_USE_MPS", "0") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _SoftmaxRouterNet(nn.Module):
    def __init__(self, d_in: int, K: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SoftmaxRouterPolicy:
    """
    Frozen supervised router baseline.
    """

    def __init__(self, model: nn.Module, d: int, K: int, device: torch.device) -> None:
        self.model = model
        self.d = int(d)
        self.K = int(K)
        self.device = device
        self.model.eval()

    def select(self, x_t: np.ndarray) -> int:
        x = np.asarray(x_t, dtype=np.float32)
        if x.shape != (self.d,):
            raise ValueError(f"x_t must have shape ({self.d},), got {x.shape}.")
        with torch.no_grad():
            logits = self.model(torch.as_tensor(x[None, :], device=self.device))
            arm = int(torch.argmax(logits, dim=1).item())
        if not (0 <= arm < self.K):
            raise ValueError(f"Predicted arm {arm} out of bounds [0, {self.K}).")
        return arm

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        del x_t, a_t, r_t


class OnlineSoftmaxPolicy:
    """
    Online policy-gradient softmax router trained from bandit feedback.
    """

    def __init__(
        self,
        d: int,
        K: int,
        lr: float = 1e-2,
        temperature: float = 1.0,
        baseline_momentum: float = 0.95,
        seed: int = 0,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive.")
        if K <= 1:
            raise ValueError("K must be at least 2.")
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not (0.0 <= baseline_momentum < 1.0):
            raise ValueError("baseline_momentum must be in [0, 1).")

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.d = int(d)
        self.K = int(K)
        self.temperature = float(temperature)
        self.baseline_momentum = float(baseline_momentum)
        self.device = _default_device()
        self.model = nn.Linear(self.d, self.K).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.baseline = 0.0
        self.t = 0
        self._last_log_prob: torch.Tensor | None = None

    def select(self, x_t: np.ndarray) -> int:
        x = np.asarray(x_t, dtype=np.float32)
        if x.shape != (self.d,):
            raise ValueError(f"x_t must have shape ({self.d},), got {x.shape}.")
        x_tensor = torch.as_tensor(x[None, :], dtype=torch.float32, device=self.device)
        logits = self.model(x_tensor).squeeze(0) / self.temperature
        probs = torch.softmax(logits, dim=0)
        probs_np = probs.detach().cpu().numpy()
        arm = int(self.rng.choice(self.K, p=probs_np))
        self._last_log_prob = torch.log(probs[arm].clamp_min(1e-12))
        return arm

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        del x_t
        if not (0 <= a_t < self.K):
            raise ValueError(f"a_t must be in [0, {self.K}).")
        if self._last_log_prob is None:
            raise RuntimeError("select must be called before update.")
        reward = float(r_t)
        if not np.isfinite(reward):
            raise ValueError("r_t must be finite.")

        advantage = reward - self.baseline
        loss = -advantage * self._last_log_prob
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.t += 1
        if self.t == 1:
            self.baseline = reward
        else:
            self.baseline = self.baseline_momentum * self.baseline + (
                1.0 - self.baseline_momentum
            ) * reward
        self._last_log_prob = None


def _train_router_on_labels(
    X_train: np.ndarray,
    y_router: np.ndarray,
    K: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> SoftmaxRouterPolicy:
    if X_train.ndim != 2:
        raise ValueError("X_train must have shape (T, d).")
    if y_router.ndim != 1:
        raise ValueError("y_router must be 1D.")
    if len(X_train) != len(y_router):
        raise ValueError("X_train and y_router must have the same number of rows.")
    if K <= 1:
        raise ValueError("K must be at least 2.")
    if np.any((y_router < 0) | (y_router >= K)):
        raise ValueError(f"y_router labels must be in [0, {K}).")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")

    _, d = X_train.shape
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _default_device()

    model = _SoftmaxRouterNet(d_in=d, K=K, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_router, dtype=torch.long)
    dataset = TensorDataset(x_t, y_t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return SoftmaxRouterPolicy(model=model, d=d, K=K, device=device)


def train_softmax_router(
    X_train: np.ndarray,
    R_train: np.ndarray,
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> SoftmaxRouterPolicy:
    """
    Train a supervised softmax router on oracle labels argmax(R_train, axis=1).
    Note: this baseline is typically evaluated in-sample in the notebook flow.
    """
    if R_train.ndim != 2:
        raise ValueError("R_train must have shape (T, K).")
    K = R_train.shape[1]
    y_router = np.argmax(R_train, axis=1).astype(np.int64)
    return _train_router_on_labels(
        X_train=X_train,
        y_router=y_router,
        K=K,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )


def train_cluster_label_router(
    X_train: np.ndarray,
    y_train: np.ndarray,
    K: int,
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> SoftmaxRouterPolicy:
    """
    Train a supervised router on true cluster/class labels, then route label i to expert i.
    """
    y_router = np.asarray(y_train, dtype=np.int64)
    return _train_router_on_labels(
        X_train=X_train,
        y_router=y_router,
        K=K,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
