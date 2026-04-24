from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearApproxErrorReport:
    """Per-arm linearization error of rewards under ridge regression on contexts."""

    epsilon_per_arm: np.ndarray
    """Shape (K,): max_t |r_{t,i} - theta_i^T x_t| for fitted theta_i."""
    mean_epsilon: float
    max_epsilon: float
    lambda_reg: float
    fit_intercept: bool


def linear_approx_max_error(
    R: np.ndarray,
    X: np.ndarray,
    *,
    lambda_reg: float = 1.0,
    fit_intercept: bool = True,
) -> LinearApproxErrorReport:
    """
    For each arm i, fit ridge regression of rewards R[:, i] on contexts x_t (optionally
    with intercept) and report epsilon_i = max_t |r_i(x_t) - <theta, x_t>|.

    Matches the offline diagnostic described in specs/moe_joint.md (Phase 1 gate).
    """
    if R.ndim != 2:
        raise ValueError("R must be 2D (T, K).")
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, d).")
    T, K = R.shape
    if X.shape[0] != T:
        raise ValueError("X and R must have the same number of rows.")
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive.")

    if fit_intercept:
        X_design = np.concatenate([X, np.ones((T, 1), dtype=np.float64)], axis=1)
    else:
        X_design = X.astype(np.float64, copy=False)

    d_aug = X_design.shape[1]
    reg = lambda_reg * np.eye(d_aug, dtype=np.float64)
    xtx = X_design.T @ X_design + reg
    epsilons = np.zeros(K, dtype=np.float64)

    for i in range(K):
        rhs = X_design.T @ R[:, i]
        theta = np.linalg.solve(xtx, rhs)
        pred = X_design @ theta
        epsilons[i] = float(np.max(np.abs(R[:, i] - pred)))

    return LinearApproxErrorReport(
        epsilon_per_arm=epsilons,
        mean_epsilon=float(np.mean(epsilons)),
        max_epsilon=float(np.max(epsilons)),
        lambda_reg=lambda_reg,
        fit_intercept=fit_intercept,
    )
