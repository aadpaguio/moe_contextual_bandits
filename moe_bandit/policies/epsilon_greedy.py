from __future__ import annotations

import numpy as np


class EpsilonGreedyPolicy:
    """
    Context-free epsilon-greedy baseline using per-arm empirical means.
    """

    def __init__(self, K: int, c: float = 50.0, seed: int = 0) -> None:
        if K <= 0:
            raise ValueError("K must be positive.")
        if c <= 0:
            raise ValueError("c must be positive.")
        self.K = K
        self.c = float(c)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(K, dtype=np.int64)
        self.means = np.zeros(K, dtype=np.float64)
        self.t = 0

    def _epsilon(self) -> float:
        return float(min(1.0, self.c / max(self.t, 1)))

    def select(self, x_t: np.ndarray) -> int:
        del x_t
        self.t += 1
        eps_t = self._epsilon()
        if self.rng.random() < eps_t:
            return int(self.rng.integers(0, self.K))
        return int(np.argmax(self.means))

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        del x_t
        if not (0 <= a_t < self.K):
            raise ValueError(f"a_t must be in [0, {self.K}).")
        self.counts[a_t] += 1
        n = self.counts[a_t]
        self.means[a_t] += (float(r_t) - self.means[a_t]) / n
