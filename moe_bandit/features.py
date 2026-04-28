from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawFeatureMap:
    """Identity map on raw context x."""

    d_in: int

    @property
    def d_out(self) -> int:
        return self.d_in

    def __call__(self, x_t: np.ndarray) -> np.ndarray:
        x = np.asarray(x_t, dtype=np.float64)
        if x.shape != (self.d_in,):
            raise ValueError(f"x_t must have shape ({self.d_in},), got {x.shape}.")
        return x


@dataclass(frozen=True)
class RBFFeatureMap:
    """
    RBF features anchored at provided centers:
      phi_j(x) = exp(-gamma ||x - c_j||^2), j=1..m
    """

    centers: np.ndarray
    gamma: float = 0.5

    def __post_init__(self) -> None:
        c = np.asarray(self.centers, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("centers must be a 2D array (m, d).")
        if c.shape[0] < 1:
            raise ValueError("centers must contain at least one center.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        object.__setattr__(self, "centers", c)

    @property
    def d_in(self) -> int:
        return int(self.centers.shape[1])

    @property
    def d_out(self) -> int:
        return int(self.centers.shape[0])

    def __call__(self, x_t: np.ndarray) -> np.ndarray:
        x = np.asarray(x_t, dtype=np.float64)
        if x.shape != (self.d_in,):
            raise ValueError(f"x_t must have shape ({self.d_in},), got {x.shape}.")
        sq = np.sum((self.centers - x[None, :]) ** 2, axis=1)
        return np.exp(-self.gamma * sq).astype(np.float64)
