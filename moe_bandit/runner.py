from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Policy(Protocol):
    """Policies implement exactly these two methods."""

    def select(self, x_t: np.ndarray) -> int: ...

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None: ...


@dataclass
class RunResult:
    chosen_arm: np.ndarray
    oracle_arm: np.ndarray
    reward: np.ndarray
    oracle_reward: np.ndarray
    regret: np.ndarray

    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.regret)


def run_bandit(
    policy: Policy,
    X: np.ndarray,
    R: np.ndarray,
    seed: int = 0,
) -> RunResult:
    """
    Run `policy` on the (X, R) stream for T = len(X) steps. Policy-agnostic.
    """
    del seed  # Runner is deterministic given policy, X, and R.

    if X.ndim != 2:
        raise ValueError("X must have shape (T, d).")
    if R.ndim != 2:
        raise ValueError("R must have shape (T, K).")
    if len(X) != len(R):
        raise ValueError("X and R must have the same number of rows (T).")

    T, K = R.shape
    chosen = np.zeros(T, dtype=np.int64)
    reward = np.zeros(T, dtype=np.float64)

    for t in range(T):
        x_t = X[t]
        a_t = int(policy.select(x_t))
        if not (0 <= a_t < K):
            raise ValueError(f"Policy selected invalid arm {a_t}; expected in [0, {K}).")
        r_t = float(R[t, a_t])
        policy.update(x_t, a_t, r_t)
        chosen[t] = a_t
        reward[t] = r_t

    oracle_arm = np.argmax(R, axis=1).astype(np.int64)
    oracle_reward = np.max(R, axis=1).astype(np.float64)
    regret = oracle_reward - reward
    return RunResult(
        chosen_arm=chosen,
        oracle_arm=oracle_arm,
        reward=reward,
        oracle_reward=oracle_reward,
        regret=regret.astype(np.float64),
    )


def run_seeds(
    policy_factory: Callable[[int], Policy],
    X: np.ndarray,
    R: np.ndarray,
    n_seeds: int,
    base_seed: int = 0,
) -> list[RunResult]:
    """
    Run the same policy factory n_seeds times with different seeds.
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive.")

    results: list[RunResult] = []
    for i in range(n_seeds):
        seed = base_seed + i
        policy = policy_factory(seed)
        results.append(run_bandit(policy=policy, X=X, R=R, seed=seed))
    return results
