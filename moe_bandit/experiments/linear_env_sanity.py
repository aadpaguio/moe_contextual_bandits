from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from moe_bandit.features import RBFFeatureMap
from moe_bandit.policies import EpsilonGreedyPolicy, LinUCBPolicy, UniformRandomPolicy
from moe_bandit.runner import run_bandit


@dataclass(frozen=True)
class LinearEnvSettings:
    K: int = 4
    d: int = 10
    T: int = 10_000
    alpha: float = 1.0
    lambda_reg: float = 1.0
    forced_explore_per_arm: int = 20
    seed: int = 0


@dataclass
class LinearEnvRow:
    policy: str
    final_cum_regret: float
    avg_regret: float
    best_arm_acc: float


def _zscore(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=0, keepdims=True)
    s = np.clip(X.std(axis=0, keepdims=True), 1e-8, None)
    return ((X - m) / s).astype(np.float64)


def _make_linear_env(settings: LinearEnvSettings) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(settings.seed)
    X = rng.normal(size=(settings.T, settings.d))
    theta = rng.normal(size=(settings.K, settings.d))
    noise = 0.05 * rng.normal(size=(settings.T, settings.K))
    R = X @ theta.T + noise
    R = _zscore(R)
    R = (R - R.min()) / max(float(R.max() - R.min()), 1e-12)
    return X.astype(np.float64), R.astype(np.float64)


def run_linear_env_sanity(
    output_dir: str | Path,
    settings: LinearEnvSettings = LinearEnvSettings(),
) -> list[LinearEnvRow]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, R = _make_linear_env(settings)
    X_std = _zscore(X)
    centers = np.stack(
        [X_std[i * (len(X_std) // settings.K)] for i in range(settings.K)], axis=0
    )
    rbf_map = RBFFeatureMap(centers=centers, gamma=0.5)

    policies = {
        "uniform": UniformRandomPolicy(K=settings.K, seed=settings.seed),
        "epsilon_greedy": EpsilonGreedyPolicy(K=settings.K, c=50.0, seed=settings.seed),
        "linucb_raw": LinUCBPolicy(
            K=settings.K,
            d=settings.d,
            alpha=settings.alpha,
            lambda_reg=settings.lambda_reg,
            forced_explore_per_arm=settings.forced_explore_per_arm,
            seed=settings.seed,
        ),
        "linucb_rbf": LinUCBPolicy(
            K=settings.K,
            d=settings.d,
            alpha=settings.alpha,
            lambda_reg=settings.lambda_reg,
            forced_explore_per_arm=settings.forced_explore_per_arm,
            seed=settings.seed + 17,
            feature_map=rbf_map,
            feature_dim=rbf_map.d_out,
            add_intercept=True,
        ),
    }

    rows: list[LinearEnvRow] = []
    for name, policy in policies.items():
        result = run_bandit(policy=policy, X=X_std, R=R, seed=settings.seed)
        rows.append(
            LinearEnvRow(
                policy=name,
                final_cum_regret=float(result.cumulative_regret()[-1]),
                avg_regret=float(result.regret.mean()),
                best_arm_acc=float(np.mean(result.chosen_arm == result.oracle_arm)),
            )
        )

    with (out / "linear_env_sanity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return rows
