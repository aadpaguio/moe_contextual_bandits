from __future__ import annotations

import numpy as np


class UniformRandomPolicy:
    def __init__(self, K: int, seed: int = 0) -> None:
        if K <= 0:
            raise ValueError("K must be positive.")
        self.K = K
        self.rng = np.random.default_rng(seed)

    def select(self, x_t: np.ndarray) -> int:
        del x_t
        return int(self.rng.integers(0, self.K))

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        del x_t, a_t, r_t
