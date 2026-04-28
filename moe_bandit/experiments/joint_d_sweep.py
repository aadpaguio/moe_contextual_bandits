from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import expert_reward_matrix
from moe_bandit.policies import (
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    UniformRandomPolicy,
    train_softmax_router,
)
from moe_bandit.runner import RunResult, run_bandit
from moe_bandit.train_joint_moe import train_joint_moe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointDSweepSettings:
    K: int = 4
    T: int = 10_000
    cluster_sep: float = 1.5
    cluster_std: float = 1.0
    n_train_per_cluster: int = 2_000
    d_values: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    seed_bundles: tuple[tuple[int, int, int, int], ...] = (
        (101, 201, 301, 401),
        (102, 202, 302, 402),
        (103, 203, 303, 403),
        (104, 204, 304, 404),
        (105, 205, 305, 405),
    )
    clip_eps: float = 1e-3
    linucb_alpha: float = 1.0
    linucb_alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    linucb_lambda: float = 1.0
    forced_explore_per_arm: int = 20
    softmax_hidden_dim: int = 64
    softmax_epochs: int = 300
    softmax_batch_size: int = 64
    softmax_lr: float = 1e-3
    epsilon_greedy_c: float = 50.0
    joint_moe_max_epochs: int = 200
    joint_moe_alpha_load: float = 0.001
    joint_moe_lr_min: float = 1e-5
    joint_moe_cosine_decay: bool = True
    joint_moe_early_stopping_patience: int | None = 20
    joint_moe_batch_size: int = 64
    seed_joint_expert_offset: int = 10_000


@dataclass(frozen=True)
class DSweepConfig:
    d: int
    seed_idx: int
    seed_train_data: int
    seed_train_experts: int
    seed_bandit_stream: int
    seed_policy: int

    @property
    def cfg_name(self) -> str:
        return f"d_{self.d}_seed_{self.seed_idx}"


@dataclass
class DSweepResultRow:
    d: int
    seed_idx: int
    policy: str
    rmse_epsilon: float
    final_cum_regret: float
    avg_regret: float
    chosen_arm_mean_reward: float
    best_arm_acc: float
    oracle_mean_reward: float
    linucb_alpha: float | None = None


class OraclePolicy:
    def __init__(self, oracle_arm: np.ndarray) -> None:
        self.oracle_arm = np.asarray(oracle_arm, dtype=np.int64)
        self.t = 0

    def select(self, x_t: np.ndarray) -> int:
        del x_t
        arm = int(self.oracle_arm[self.t])
        self.t += 1
        return arm

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        del x_t, a_t, r_t


def _rescale_rewards_01(R_raw: np.ndarray) -> np.ndarray:
    r_min = float(np.min(R_raw))
    r_max = float(np.max(R_raw))
    if r_max - r_min <= 1e-12:
        return np.zeros_like(R_raw, dtype=np.float64)
    return ((R_raw - r_min) / (r_max - r_min)).astype(np.float64)


def _zscore_from_reference(reference: np.ndarray, X: np.ndarray) -> np.ndarray:
    mean = reference.mean(axis=0, keepdims=True)
    std = reference.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-8, None)
    return ((X - mean) / std).astype(np.float64)


def _ridge_rmse_epsilon(
    R: np.ndarray, X: np.ndarray, *, lambda_reg: float = 1.0, fit_intercept: bool = True
) -> float:
    if fit_intercept:
        X_design = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    else:
        X_design = X.astype(np.float64, copy=False)
    reg = lambda_reg * np.eye(X_design.shape[1], dtype=np.float64)
    xtx = X_design.T @ X_design + reg

    arm_rmses = np.zeros(R.shape[1], dtype=np.float64)
    for arm in range(R.shape[1]):
        theta = np.linalg.solve(xtx, X_design.T @ R[:, arm])
        residual = R[:, arm] - (X_design @ theta)
        arm_rmses[arm] = float(np.sqrt(np.mean(np.square(residual))))
    return float(np.mean(arm_rmses))


def _row_from_result(
    cfg: DSweepConfig,
    policy: str,
    result: RunResult,
    *,
    rmse_epsilon: float,
    oracle_mean_reward: float,
    linucb_alpha: float | None = None,
) -> DSweepResultRow:
    cum_regret = result.cumulative_regret()
    best_arm_acc = float(np.mean(result.chosen_arm == result.oracle_arm))
    return DSweepResultRow(
        d=cfg.d,
        seed_idx=cfg.seed_idx,
        policy=policy,
        rmse_epsilon=rmse_epsilon,
        final_cum_regret=float(cum_regret[-1]),
        avg_regret=float(result.regret.mean()),
        chosen_arm_mean_reward=float(result.reward.mean()),
        best_arm_acc=best_arm_acc,
        oracle_mean_reward=oracle_mean_reward,
        linucb_alpha=linucb_alpha,
    )


def _write_rows_csv(path: Path, rows: list[DSweepResultRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_rows_jsonl(path: Path, rows: list[DSweepResultRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def _write_artifacts_json(path: Path, artifacts: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(artifacts, f)


def _aggregate_metric(rows: list[DSweepResultRow], *, policy: str, metric: str, d: int) -> float:
    values = [getattr(row, metric) for row in rows if row.policy == policy and row.d == d]
    return float(np.mean(values))


def _write_summary_csv(path: Path, rows: list[DSweepResultRow], d_values: tuple[int, ...]) -> None:
    fieldnames = [
        "d",
        "rmse_epsilon",
        "uniform_regret",
        "linucb_regret",
        "epsilon_greedy_regret",
        "softmax_regret",
        "oracle_mean_reward",
        "linucb_minus_epsilon_greedy",
        "linucb_minus_softmax",
        "epsilon_greedy_minus_linucb",
        "softmax_minus_linucb",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in d_values:
            rmse_epsilon = float(np.mean([row.rmse_epsilon for row in rows if row.d == d]))
            uniform_regret = _aggregate_metric(rows, policy="uniform", metric="final_cum_regret", d=d)
            linucb_regret = _aggregate_metric(rows, policy="linucb_raw", metric="final_cum_regret", d=d)
            eps_regret = _aggregate_metric(
                rows, policy="epsilon_greedy", metric="final_cum_regret", d=d
            )
            softmax_regret = _aggregate_metric(
                rows, policy="softmax_router", metric="final_cum_regret", d=d
            )
            oracle_reward = _aggregate_metric(
                rows, policy="oracle", metric="chosen_arm_mean_reward", d=d
            )
            writer.writerow(
                {
                    "d": d,
                    "rmse_epsilon": rmse_epsilon,
                    "uniform_regret": uniform_regret,
                    "linucb_regret": linucb_regret,
                    "epsilon_greedy_regret": eps_regret,
                    "softmax_regret": softmax_regret,
                    "oracle_mean_reward": oracle_reward,
                    "linucb_minus_epsilon_greedy": linucb_regret - eps_regret,
                    "linucb_minus_softmax": linucb_regret - softmax_regret,
                    "epsilon_greedy_minus_linucb": eps_regret - linucb_regret,
                    "softmax_minus_linucb": softmax_regret - linucb_regret,
                }
            )


def _plot_regret_curves_by_d(
    artifacts: dict[str, Any], output_path: Path, d_values: tuple[int, ...], n_cols: int = 3
) -> None:
    n_rows = int(np.ceil(len(d_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    policy_order = ["uniform", "epsilon_greedy", "linucb_raw", "softmax_router", "oracle"]

    for idx, d in enumerate(d_values):
        ax = axes_arr[idx]
        for policy in policy_order:
            curves: list[list[float]] = []
            for run in artifacts["runs"].values():
                if run["d"] == d:
                    curves.append(run["policies"][policy]["cumulative_regret_curve"])
            if not curves:
                continue
            mean_curve = np.asarray(curves, dtype=np.float64).mean(axis=0)
            ax.plot(mean_curve, label=policy)
        ax.set_title(f"d={d}")
        ax.set_xlabel("t")
        if idx % n_cols == 0:
            ax.set_ylabel("cumulative regret")
        if ax.lines:
            ax.legend(fontsize=8)

    for idx in range(len(d_values), len(axes_arr)):
        fig.delaxes(axes_arr[idx])

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_summary_vs_d(output_dir: Path, rows: list[DSweepResultRow], d_values: tuple[int, ...]) -> None:
    d_arr = np.asarray(d_values, dtype=np.int64)
    rmse_arr = np.asarray([float(np.mean([r.rmse_epsilon for r in rows if r.d == d])) for d in d_values])
    linucb = np.asarray(
        [_aggregate_metric(rows, policy="linucb_raw", metric="final_cum_regret", d=d) for d in d_values]
    )
    uniform = np.asarray(
        [_aggregate_metric(rows, policy="uniform", metric="final_cum_regret", d=d) for d in d_values]
    )
    eps = np.asarray(
        [
            _aggregate_metric(rows, policy="epsilon_greedy", metric="final_cum_regret", d=d)
            for d in d_values
        ]
    )
    softmax = np.asarray(
        [_aggregate_metric(rows, policy="softmax_router", metric="final_cum_regret", d=d) for d in d_values]
    )
    oracle_reward = np.asarray(
        [_aggregate_metric(rows, policy="oracle", metric="chosen_arm_mean_reward", d=d) for d in d_values]
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(d_arr, rmse_arr, marker="o")
    ax.set_title("Offline ridge RMSE epsilon vs d")
    ax.set_xlabel("d")
    ax.set_ylabel("RMSE epsilon")
    ax.set_xscale("log", base=2)

    ax = axes[0, 1]
    ax.plot(d_arr, uniform, marker="o", label="uniform")
    ax.plot(d_arr, linucb, marker="o", label="LinUCB")
    ax.plot(d_arr, eps, marker="o", label="epsilon-greedy")
    ax.plot(d_arr, softmax, marker="o", label="softmax")
    ax.set_title("Mean final cumulative regret vs d")
    ax.set_xlabel("d")
    ax.set_ylabel("final cumulative regret")
    ax.set_xscale("log", base=2)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(d_arr, linucb - eps, marker="o", label="LinUCB - epsilon-greedy")
    ax.plot(d_arr, linucb - softmax, marker="o", label="LinUCB - softmax")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_title("Regret gaps vs d")
    ax.set_xlabel("d")
    ax.set_ylabel("gap in final cumulative regret")
    ax.set_xscale("log", base=2)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(d_arr, oracle_reward, marker="o", color="tab:green")
    ax.set_title("Oracle mean reward vs d")
    ax.set_xlabel("d")
    ax.set_ylabel("mean reward")
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    fig.savefig(output_dir / "d_sweep_summary_metrics.png", dpi=140)
    plt.close(fig)


def run_joint_d_sweep(
    output_dir: str | Path, settings: JointDSweepSettings = JointDSweepSettings()
) -> tuple[list[DSweepResultRow], dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[DSweepResultRow] = []
    artifacts: dict[str, Any] = {
        "settings": asdict(settings),
        "sweep": {"d_values": list(settings.d_values), "seed_bundles": [list(s) for s in settings.seed_bundles]},
        "runs": {},
    }
    n_train = settings.K * settings.n_train_per_cluster
    total_runs = len(settings.d_values) * len(settings.seed_bundles)

    run_counter = 0
    for d in settings.d_values:
        for seed_idx, bundle in enumerate(settings.seed_bundles):
            seed_train_data, seed_train_experts, seed_bandit_stream, seed_policy = bundle
            cfg = DSweepConfig(
                d=d,
                seed_idx=seed_idx,
                seed_train_data=seed_train_data,
                seed_train_experts=seed_train_experts,
                seed_bandit_stream=seed_bandit_stream,
                seed_policy=seed_policy,
            )
            run_counter += 1
            logger.info(
                "[%d/%d] d=%d seed_idx=%d | train_data=%d train_exp=%d bandit=%d policy=%d",
                run_counter,
                total_runs,
                d,
                seed_idx,
                seed_train_data,
                seed_train_experts,
                seed_bandit_stream,
                seed_policy,
            )

            X_train, y_train, cluster_train = generate_synthetic_data(
                n_samples=n_train,
                K=settings.K,
                d=d,
                cluster_sep=settings.cluster_sep,
                cluster_std=settings.cluster_std,
                seed=seed_train_data,
            )
            X_bandit, y_bandit, _ = generate_synthetic_data(
                n_samples=settings.T,
                K=settings.K,
                d=d,
                cluster_sep=settings.cluster_sep,
                cluster_std=settings.cluster_std,
                seed=seed_bandit_stream,
            )
            X_train_std = _zscore_from_reference(X_train, X_train)
            X_bandit_std = _zscore_from_reference(X_train, X_bandit)

            experts, _ = train_joint_moe(
                X_train=X_train,
                y_train=y_train,
                cluster_id_train=cluster_train,
                K=settings.K,
                d=d,
                epochs=settings.joint_moe_max_epochs,
                lr=1e-3,
                lr_min=settings.joint_moe_lr_min,
                cosine_decay=settings.joint_moe_cosine_decay,
                batch_size=settings.joint_moe_batch_size,
                seed=seed_train_experts + settings.seed_joint_expert_offset,
                alpha_load=settings.joint_moe_alpha_load,
                early_stopping_patience=settings.joint_moe_early_stopping_patience,
                router="linear",
            )
            R_raw = expert_reward_matrix(experts, X_bandit, y_bandit, clip_eps=settings.clip_eps)
            R = _rescale_rewards_01(R_raw)
            rmse_epsilon = _ridge_rmse_epsilon(R_raw, X_bandit, lambda_reg=settings.linucb_lambda)
            oracle_arm = np.argmax(R, axis=1)

            R_router_train_raw = expert_reward_matrix(
                experts, X_train, y_train, clip_eps=settings.clip_eps
            )
            R_router_train = _rescale_rewards_01(R_router_train_raw)
            softmax_router = train_softmax_router(
                X_train=X_train_std,
                R_train=R_router_train,
                hidden_dim=settings.softmax_hidden_dim,
                epochs=settings.softmax_epochs,
                batch_size=settings.softmax_batch_size,
                lr=settings.softmax_lr,
                seed=seed_policy,
            )

            linucb_policies: dict[str, tuple[LinUCBPolicy, float]] = {}
            for alpha in settings.linucb_alphas:
                alpha_f = float(alpha)
                label = f"linucb_raw_alpha_{alpha_f:g}"
                linucb_policies[label] = (
                    LinUCBPolicy(
                        K=settings.K,
                        d=d,
                        alpha=alpha_f,
                        lambda_reg=settings.linucb_lambda,
                        forced_explore_per_arm=settings.forced_explore_per_arm,
                        seed=seed_policy,
                    ),
                    alpha_f,
                )
                # Keep legacy key for downstream summaries/plots.
                if np.isclose(alpha_f, settings.linucb_alpha):
                    linucb_policies["linucb_raw"] = (
                        LinUCBPolicy(
                            K=settings.K,
                            d=d,
                            alpha=alpha_f,
                            lambda_reg=settings.linucb_lambda,
                            forced_explore_per_arm=settings.forced_explore_per_arm,
                            seed=seed_policy,
                        ),
                        alpha_f,
                    )

            policies: dict[str, Any] = {
                "uniform": UniformRandomPolicy(K=settings.K, seed=seed_policy),
                "epsilon_greedy": EpsilonGreedyPolicy(
                    K=settings.K, c=settings.epsilon_greedy_c, seed=seed_policy
                ),
                "softmax_router": softmax_router,
                "oracle": OraclePolicy(oracle_arm=oracle_arm),
            }
            for policy_name, (policy_obj, _) in linucb_policies.items():
                policies[policy_name] = policy_obj

            run_store: dict[str, Any] = {
                "d": d,
                "seed_idx": seed_idx,
                "seed_bundle": {
                    "train_data": seed_train_data,
                    "train_experts": seed_train_experts,
                    "bandit_stream": seed_bandit_stream,
                    "policy": seed_policy,
                },
                "rmse_epsilon": rmse_epsilon,
                "cluster_sep": settings.cluster_sep,
                "policies": {},
            }

            oracle_mean_reward = float(np.max(R, axis=1).mean())
            for policy_name, policy_obj in policies.items():
                result = run_bandit(policy=policy_obj, X=X_bandit_std, R=R, seed=seed_policy)
                alpha_for_row = (
                    linucb_policies[policy_name][1] if policy_name in linucb_policies else None
                )
                rows.append(
                    _row_from_result(
                        cfg,
                        policy_name,
                        result,
                        rmse_epsilon=rmse_epsilon,
                        oracle_mean_reward=oracle_mean_reward,
                        linucb_alpha=alpha_for_row,
                    )
                )
                run_store["policies"][policy_name] = {
                    "cumulative_regret_curve": result.cumulative_regret().astype(float).tolist(),
                }
            artifacts["runs"][cfg.cfg_name] = run_store

    _write_rows_csv(out / "results_rows.csv", rows)
    _write_rows_jsonl(out / "results_rows.jsonl", rows)
    _write_summary_csv(out / "d_sweep_summary.csv", rows, settings.d_values)
    _write_artifacts_json(out / "artifacts.json", artifacts)
    _plot_regret_curves_by_d(
        artifacts=artifacts, output_path=out / "regret_curves_by_d.png", d_values=settings.d_values
    )
    _plot_summary_vs_d(out, rows, settings.d_values)

    logger.info("Joint d-sweep complete. Saved outputs to %s", out.resolve())
    return rows, artifacts
