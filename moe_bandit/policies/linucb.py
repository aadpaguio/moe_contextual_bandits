from __future__ import annotations

import numpy as np
from typing import Callable


class LinUCBPolicy:
    def __init__(
        self,
        K: int,
        d: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        forced_explore_per_arm: int = 20,
        seed: int = 0,
        feature_map: Callable[[np.ndarray], np.ndarray] | None = None,
        feature_dim: int | None = None,
        add_intercept: bool = True,
    ) -> None:
        if K < 2:
            raise ValueError("K must be at least 2.")
        if d < 1:
            raise ValueError("d must be at least 1.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if lambda_reg <= 0:
            raise ValueError("lambda_reg must be positive.")
        if forced_explore_per_arm < 0:
            raise ValueError("forced_explore_per_arm must be non-negative.")

        self.K = K
        self.d_raw = d
        self.feature_map = feature_map
        self.add_intercept = bool(add_intercept)
        self.d_feat = int(self.d_raw if feature_dim is None else feature_dim)
        if self.d_feat < 1:
            raise ValueError("feature_dim must be positive.")
        self.d_aug = self.d_feat + (1 if self.add_intercept else 0)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)

        eye = np.eye(self.d_aug, dtype=np.float64)
        self.V = np.tile((self.lambda_reg * eye)[None, :, :], (self.K, 1, 1))
        self.b = np.zeros((self.K, self.d_aug), dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.pull_counts = np.zeros(self.K, dtype=np.int64)
        self._forced_explore_steps = int(forced_explore_per_arm) * self.K

    def _transform(self, x_t: np.ndarray) -> np.ndarray:
        x = np.asarray(x_t, dtype=np.float64)
        if x.shape != (self.d_raw,):
            raise ValueError(f"x_t must have shape ({self.d_raw},), got {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x_t must contain only finite values.")
        if self.feature_map is not None:
            feat = np.asarray(self.feature_map(x), dtype=np.float64)
            if feat.shape != (self.d_feat,):
                raise ValueError(
                    f"feature_map(x_t) must have shape ({self.d_feat},), got {feat.shape}."
                )
            if not np.all(np.isfinite(feat)):
                raise ValueError("feature_map(x_t) must contain only finite values.")
            return feat
        return x

    def _augment(self, x_t: np.ndarray) -> np.ndarray:
        feat = self._transform(x_t)
        if not self.add_intercept:
            return feat
        x_aug = np.empty(self.d_aug, dtype=np.float64)
        x_aug[:-1] = feat
        x_aug[-1] = 1.0
        return x_aug

    def select(self, x_t: np.ndarray) -> int:
        x = self._augment(x_t)

        # Force minimal coverage so each arm gets initial data.
        total_pulls = int(self.pull_counts.sum())
        if total_pulls < self._forced_explore_steps:
            return total_pulls % self.K

        # Batched solve over K linear systems (NumPy 2.x needs explicit RHS axis).
        theta_hat = np.linalg.solve(self.V, self.b[..., None]).squeeze(-1)  # (K, d_aug)
        mean = theta_hat @ x  # (K,)

        x_stack = np.tile(x, (self.K, 1))  # (K, d_aug), writable
        y = np.linalg.solve(self.V, x_stack[..., None]).squeeze(-1)  # (K, d_aug)
        width_sq = np.einsum("kd,kd->k", x_stack, y)
        ucb = mean + self.alpha * np.sqrt(np.maximum(width_sq, 0.0))

        max_val = ucb.max()
        candidates = np.flatnonzero(ucb >= max_val - 1e-12)
        return int(self.rng.choice(candidates))

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        if not (0 <= a_t < self.K):
            raise ValueError(f"a_t must be in [0, {self.K}).")
        r = float(r_t)
        if not np.isfinite(r):
            raise ValueError("r_t must be finite.")
        x = self._augment(x_t)
        self.V[a_t] += np.outer(x, x)
        self.b[a_t] += r * x
        self.pull_counts[a_t] += 1