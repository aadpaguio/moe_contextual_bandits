from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import expert_reward_matrix, train_experts
from moe_bandit.policies import (
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    OnlineSoftmaxPolicy,
    UniformRandomPolicy,
    train_cluster_label_router,
    train_softmax_router,
)
from moe_bandit.runner import RunResult, run_bandit
from moe_bandit.train_joint_moe import JointTrainingStats, train_joint_moe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportPacketMotivationSettings:
    d: int = 4
    K: int = 4
    cluster_sep: float = 1.5
    cluster_std: float = 1.0
    contamination: float = 0.05
    n_train: int = 8_000
    T: int = 10_000
    seeds: tuple[int, ...] = (0, 1, 2)
    epsilon_greedy_c: float = 50.0
    linucb_alpha: float = 1.0


@dataclass(frozen=True)
class ReportPacketMainSettings:
    K: int = 4
    cluster_sep: float = 1.5
    cluster_std: float = 1.0
    n_train: int = 8_000
    T: int = 10_000
    d_values: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    alpha_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    epsilon_greedy_c: float = 50.0
    select_alpha_by: Literal["mean_regret", "median_regret"] = "mean_regret"


@dataclass(frozen=True)
class ReportPacketSettings:
    motivation: ReportPacketMotivationSettings = field(default_factory=ReportPacketMotivationSettings)
    main: ReportPacketMainSettings = field(default_factory=ReportPacketMainSettings)
    clip_eps: float = 1e-3
    linucb_lambda: float = 1.0
    forced_explore_per_arm: int = 20
    softmax_hidden_dim: int = 64
    softmax_epochs: int = 300
    softmax_batch_size: int = 64
    softmax_lr: float = 1e-3
    online_softmax_lr: float = 1e-2
    online_softmax_temperature: float = 1.0
    online_softmax_baseline_momentum: float = 0.95
    joint_moe_max_epochs: int = 200
    joint_moe_alpha_load: float = 0.001
    joint_moe_lr_min: float = 1e-5
    joint_moe_cosine_decay: bool = True
    joint_moe_early_stopping_patience: int | None = 20
    joint_moe_batch_size: int = 64
    seed_train_experts_offset: int = 1_000
    seed_bandit_stream_offset: int = 2_000
    seed_policy_offset: int = 3_000
    seed_joint_expert_offset: int = 10_000


@dataclass
class ReportResultRow:
    block: str
    expert_regime: str
    d: int
    seed_idx: int
    seed: int
    policy: str
    final_cum_regret: float
    avg_regret: float
    chosen_arm_mean_reward: float
    best_arm_acc: float
    oracle_mean_reward: float
    rmse_epsilon: float
    max_epsilon: float
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


def run_report_packet(
    output_dir: str | Path,
    settings: ReportPacketSettings = ReportPacketSettings(),
    *,
    timestamped: bool = True,
) -> Path:
    root = Path(output_dir)
    if timestamped:
        root = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    _prepare_dirs(root)
    _write_manifest(root / "manifest.json", settings)

    rows: list[ReportResultRow] = []
    artifacts: dict[str, Any] = {"settings": asdict(settings), "runs": {}}
    approx_rows: list[dict[str, Any]] = []
    joint_stats_rows: list[dict[str, Any]] = []

    _run_motivation_block(root, settings, rows, artifacts, approx_rows, joint_stats_rows)
    _run_main_block(root, settings, rows, artifacts, approx_rows, joint_stats_rows)

    _write_rows_csv(root / "results_rows.csv", rows)
    _write_rows_jsonl(root / "results_rows.jsonl", [asdict(row) for row in rows])
    _write_rows_jsonl(root / "approx_error_by_regime.jsonl", approx_rows)
    _write_rows_jsonl(root / "joint_training_stats.jsonl", joint_stats_rows)
    _write_json(root / "artifacts.json", artifacts)
    _make_plots(root, rows, artifacts)

    logger.info("Report packet written to %s", root.resolve())
    return root


def _run_motivation_block(
    root: Path,
    settings: ReportPacketSettings,
    rows: list[ReportResultRow],
    artifacts: dict[str, Any],
    approx_rows: list[dict[str, Any]],
    joint_stats_rows: list[dict[str, Any]],
) -> None:
    cfg = settings.motivation
    for seed_idx, seed in enumerate(cfg.seeds):
        X_train, y_train, cluster_train = generate_synthetic_data(
            n_samples=cfg.n_train,
            K=cfg.K,
            d=cfg.d,
            cluster_sep=cfg.cluster_sep,
            cluster_std=cfg.cluster_std,
            seed=seed,
        )
        X_bandit, y_bandit, cluster_bandit = generate_synthetic_data(
            n_samples=cfg.T,
            K=cfg.K,
            d=cfg.d,
            cluster_sep=cfg.cluster_sep,
            cluster_std=cfg.cluster_std,
            seed=seed + settings.seed_bandit_stream_offset,
        )
        X_train_std = _zscore_from_reference(X_train, X_train)
        X_bandit_std = _zscore_from_reference(X_train, X_bandit)

        for regime in ("independent", "joint"):
            if regime == "independent":
                experts = train_experts(
                    X_train=X_train,
                    y_train=y_train,
                    cluster_id_train=cluster_train,
                    K=cfg.K,
                    d=cfg.d,
                    epochs=30,
                    lr=1e-3,
                    batch_size=64,
                    seed=seed + settings.seed_train_experts_offset,
                    contamination=cfg.contamination,
                )
            else:
                experts, stats = train_joint_moe(
                    X_train=X_train,
                    y_train=y_train,
                    cluster_id_train=cluster_train,
                    K=cfg.K,
                    d=cfg.d,
                    epochs=settings.joint_moe_max_epochs,
                    lr=1e-3,
                    lr_min=settings.joint_moe_lr_min,
                    cosine_decay=settings.joint_moe_cosine_decay,
                    batch_size=settings.joint_moe_batch_size,
                    seed=seed + settings.seed_joint_expert_offset,
                    alpha_load=settings.joint_moe_alpha_load,
                    early_stopping_patience=settings.joint_moe_early_stopping_patience,
                    router="linear",
                )
                joint_stats_rows.append(
                    _joint_stats_to_row(
                        block="motivation", regime=regime, d=cfg.d, seed_idx=seed_idx, seed=seed, stats=stats
                    )
                )

            _evaluate_and_store(
                root=root,
                settings=settings,
                block="motivation",
                regime=regime,
                d=cfg.d,
                seed_idx=seed_idx,
                seed=seed,
                K=cfg.K,
                X_train=X_train,
                y_train=y_train,
                cluster_train=cluster_train,
                X_train_std=X_train_std,
                X_bandit=X_bandit,
                y_bandit=y_bandit,
                cluster_bandit=cluster_bandit,
                X_bandit_std=X_bandit_std,
                experts=experts,
                epsilon_greedy_c=cfg.epsilon_greedy_c,
                linucb_alphas=(cfg.linucb_alpha,),
                rows=rows,
                artifacts=artifacts,
                approx_rows=approx_rows,
            )


def _run_main_block(
    root: Path,
    settings: ReportPacketSettings,
    rows: list[ReportResultRow],
    artifacts: dict[str, Any],
    approx_rows: list[dict[str, Any]],
    joint_stats_rows: list[dict[str, Any]],
) -> None:
    cfg = settings.main
    for d in cfg.d_values:
        for seed_idx, seed in enumerate(cfg.seeds):
            X_train, y_train, cluster_train = generate_synthetic_data(
                n_samples=cfg.n_train,
                K=cfg.K,
                d=d,
                cluster_sep=cfg.cluster_sep,
                cluster_std=cfg.cluster_std,
                seed=seed,
            )
            X_bandit, y_bandit, cluster_bandit = generate_synthetic_data(
                n_samples=cfg.T,
                K=cfg.K,
                d=d,
                cluster_sep=cfg.cluster_sep,
                cluster_std=cfg.cluster_std,
                seed=seed + settings.seed_bandit_stream_offset,
            )
            X_train_std = _zscore_from_reference(X_train, X_train)
            X_bandit_std = _zscore_from_reference(X_train, X_bandit)

            experts, stats = train_joint_moe(
                X_train=X_train,
                y_train=y_train,
                cluster_id_train=cluster_train,
                K=cfg.K,
                d=d,
                epochs=settings.joint_moe_max_epochs,
                lr=1e-3,
                lr_min=settings.joint_moe_lr_min,
                cosine_decay=settings.joint_moe_cosine_decay,
                batch_size=settings.joint_moe_batch_size,
                seed=seed + settings.seed_joint_expert_offset,
                alpha_load=settings.joint_moe_alpha_load,
                early_stopping_patience=settings.joint_moe_early_stopping_patience,
                router="linear",
            )
            joint_stats_rows.append(
                _joint_stats_to_row(block="main", regime="joint", d=d, seed_idx=seed_idx, seed=seed, stats=stats)
            )

            _evaluate_and_store(
                root=root,
                settings=settings,
                block="main",
                regime="joint",
                d=d,
                seed_idx=seed_idx,
                seed=seed,
                K=cfg.K,
                X_train=X_train,
                y_train=y_train,
                cluster_train=cluster_train,
                X_train_std=X_train_std,
                X_bandit=X_bandit,
                y_bandit=y_bandit,
                cluster_bandit=cluster_bandit,
                X_bandit_std=X_bandit_std,
                experts=experts,
                epsilon_greedy_c=cfg.epsilon_greedy_c,
                linucb_alphas=cfg.alpha_values,
                rows=rows,
                artifacts=artifacts,
                approx_rows=approx_rows,
            )


def _evaluate_and_store(
    *,
    root: Path,
    settings: ReportPacketSettings,
    block: str,
    regime: str,
    d: int,
    seed_idx: int,
    seed: int,
    K: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_train: np.ndarray,
    X_train_std: np.ndarray,
    X_bandit: np.ndarray,
    y_bandit: np.ndarray,
    cluster_bandit: np.ndarray,
    X_bandit_std: np.ndarray,
    experts: Any,
    epsilon_greedy_c: float,
    linucb_alphas: tuple[float, ...],
    rows: list[ReportResultRow],
    artifacts: dict[str, Any],
    approx_rows: list[dict[str, Any]],
) -> None:
    R_raw = expert_reward_matrix(experts, X_bandit, y_bandit, clip_eps=settings.clip_eps)
    R = _rescale_rewards_01(R_raw)
    R_router_train_raw = expert_reward_matrix(experts, X_train, y_train, clip_eps=settings.clip_eps)
    R_router_train = _rescale_rewards_01(R_router_train_raw)
    oracle_arm = np.argmax(R, axis=1)
    oracle_mean_reward = float(np.max(R, axis=1).mean())
    eps_report = _linear_diagnostics(R_raw, X_bandit, lambda_reg=settings.linucb_lambda)

    run_key = f"{block}__{regime}__d_{d}__seed_{seed_idx}"
    approx_rows.append(
        {
            "block": block,
            "expert_regime": regime,
            "d": d,
            "seed_idx": seed_idx,
            "seed": seed,
            **eps_report,
        }
    )

    softmax_router = train_softmax_router(
        X_train=X_train_std,
        R_train=R_router_train,
        hidden_dim=settings.softmax_hidden_dim,
        epochs=settings.softmax_epochs,
        batch_size=settings.softmax_batch_size,
        lr=settings.softmax_lr,
        seed=seed + settings.seed_policy_offset,
    )
    softmax_train_acc = _softmax_accuracy(softmax_router, X_train_std, R_router_train)
    softmax_eval_acc = _softmax_accuracy(softmax_router, X_bandit_std, R)
    cluster_label_router = train_cluster_label_router(
        X_train=X_train_std,
        y_train=y_train,
        K=K,
        hidden_dim=settings.softmax_hidden_dim,
        epochs=settings.softmax_epochs,
        batch_size=settings.softmax_batch_size,
        lr=settings.softmax_lr,
        seed=seed + settings.seed_policy_offset + 17,
    )
    cluster_label_train_acc = _label_accuracy(cluster_label_router, X_train_std, y_train)
    cluster_label_eval_acc = _label_accuracy(cluster_label_router, X_bandit_std, y_bandit)

    policies: dict[str, tuple[Any, float | None]] = {
        "uniform": (UniformRandomPolicy(K=K, seed=seed + settings.seed_policy_offset), None),
        "epsilon_greedy": (
            EpsilonGreedyPolicy(K=K, c=epsilon_greedy_c, seed=seed + settings.seed_policy_offset),
            None,
        ),
        "online_softmax_best_arm": (
            OnlineSoftmaxPolicy(
                d=d,
                K=K,
                lr=settings.online_softmax_lr,
                temperature=settings.online_softmax_temperature,
                baseline_momentum=settings.online_softmax_baseline_momentum,
                seed=seed + settings.seed_policy_offset,
            ),
            None,
        ),
        "softmax_best_arm": (softmax_router, None),
        "cluster_label_router": (cluster_label_router, None),
        "oracle": (OraclePolicy(oracle_arm=oracle_arm), None),
    }
    for alpha in linucb_alphas:
        alpha_f = float(alpha)
        policies[f"linucb_raw_alpha_{alpha_f:g}"] = (
            LinUCBPolicy(
                K=K,
                d=d,
                alpha=alpha_f,
                lambda_reg=settings.linucb_lambda,
                forced_explore_per_arm=settings.forced_explore_per_arm,
                seed=seed + settings.seed_policy_offset,
            ),
            alpha_f,
        )
    if 1.0 in [float(a) for a in linucb_alphas]:
        policies["linucb_raw"] = (
            LinUCBPolicy(
                K=K,
                d=d,
                alpha=1.0,
                lambda_reg=settings.linucb_lambda,
                forced_explore_per_arm=settings.forced_explore_per_arm,
                seed=seed + settings.seed_policy_offset,
            ),
            1.0,
        )

    artifacts["runs"][run_key] = {
        "block": block,
        "expert_regime": regime,
        "d": d,
        "seed_idx": seed_idx,
        "seed": seed,
        "rmse_epsilon": eps_report["rmse_epsilon"],
        "max_epsilon": eps_report["max_epsilon"],
        "policies": {},
        "reward_heatmap": _reward_heatmap(R_raw, cluster_bandit, K).tolist(),
        "softmax_best_arm": {
            "train_best_arm_acc": softmax_train_acc,
            "eval_best_arm_acc": softmax_eval_acc,
            "train_context_source": "X_train",
            "eval_context_source": "X_bandit",
        },
        "cluster_label_router": {
            "train_label_acc": cluster_label_train_acc,
            "eval_label_acc": cluster_label_eval_acc,
            "target": "y_train / cluster label",
        },
    }

    npz_payload: dict[str, np.ndarray] = {
        "X_train": X_train,
        "y_train": y_train,
        "cluster_train": cluster_train,
        "X_bandit": X_bandit,
        "y_bandit": y_bandit,
        "cluster_bandit": cluster_bandit,
        "R_raw": R_raw,
        "R_scaled": R,
        "R_router_train_raw": R_router_train_raw,
        "R_router_train_scaled": R_router_train,
        "oracle_arm": oracle_arm,
    }

    for policy_name, (policy, alpha) in policies.items():
        result = run_bandit(policy=policy, X=X_bandit_std, R=R, seed=seed + settings.seed_policy_offset)
        _append_result_row(
            rows=rows,
            block=block,
            regime=regime,
            d=d,
            seed_idx=seed_idx,
            seed=seed,
            policy_name=policy_name,
            result=result,
            oracle_mean_reward=oracle_mean_reward,
            rmse_epsilon=float(eps_report["rmse_epsilon"]),
            max_epsilon=float(eps_report["max_epsilon"]),
            alpha=alpha,
        )
        cum_regret = result.cumulative_regret()
        artifacts["runs"][run_key]["policies"][policy_name] = {
            "final_cum_regret": float(cum_regret[-1]),
            "cumulative_regret_curve": cum_regret.astype(float).tolist(),
        }
        safe_policy = policy_name.replace(".", "p")
        npz_payload[f"{safe_policy}__chosen_arm"] = result.chosen_arm
        npz_payload[f"{safe_policy}__reward"] = result.reward
        npz_payload[f"{safe_policy}__oracle_reward"] = result.oracle_reward
        npz_payload[f"{safe_policy}__regret"] = result.regret
        npz_payload[f"{safe_policy}__cumulative_regret"] = cum_regret

    raw_dir = root / "raw" / f"{block}_{regime}_d={d}_seed={seed_idx}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / "seed_data.npz", **npz_payload)


def _append_result_row(
    *,
    rows: list[ReportResultRow],
    block: str,
    regime: str,
    d: int,
    seed_idx: int,
    seed: int,
    policy_name: str,
    result: RunResult,
    oracle_mean_reward: float,
    rmse_epsilon: float,
    max_epsilon: float,
    alpha: float | None,
) -> None:
    cum_regret = result.cumulative_regret()
    rows.append(
        ReportResultRow(
            block=block,
            expert_regime=regime,
            d=d,
            seed_idx=seed_idx,
            seed=seed,
            policy=policy_name,
            final_cum_regret=float(cum_regret[-1]),
            avg_regret=float(result.regret.mean()),
            chosen_arm_mean_reward=float(result.reward.mean()),
            best_arm_acc=float(np.mean(result.chosen_arm == result.oracle_arm)),
            oracle_mean_reward=oracle_mean_reward,
            rmse_epsilon=rmse_epsilon,
            max_epsilon=max_epsilon,
            linucb_alpha=alpha,
        )
    )


def _make_plots(root: Path, rows: list[ReportResultRow], artifacts: dict[str, Any]) -> None:
    _plot_motivation_heatmaps(root, artifacts)
    _plot_epsilon_comparison(root, rows)
    _plot_regret_vs_d(root, rows)
    _plot_regret_curves_per_d(root, artifacts)
    _plot_diagnostic_regret_curves_per_d(root, artifacts)
    _plot_alpha_sweep(root, rows)
    _plot_linucb_minus_softmax(root, rows)


def _plot_motivation_heatmaps(root: Path, artifacts: dict[str, Any]) -> None:
    runs = [
        run
        for run in artifacts["runs"].values()
        if run["block"] == "motivation" and run["seed_idx"] == 0
    ]
    if not runs:
        return
    values = [np.asarray(run["reward_heatmap"], dtype=np.float64) for run in runs]
    vmin = min(float(v.min()) for v in values)
    vmax = max(float(v.max()) for v in values)
    for run, heatmap in zip(runs, values, strict=False):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(heatmap, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xlabel("arm")
        ax.set_ylabel("true cluster")
        ax.set_title(f"{run['expert_regime']} reward heatmap, RMSE={run['rmse_epsilon']:.3f}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(root / "plots" / "motivation" / f"reward_heatmap_{run['expert_regime']}.png", dpi=150)
        plt.close(fig)


def _plot_epsilon_comparison(root: Path, rows: list[ReportResultRow]) -> None:
    regimes = ["independent", "joint"]
    means = []
    stds = []
    for regime in regimes:
        vals = [
            row.rmse_epsilon
            for row in rows
            if row.block == "motivation" and row.expert_regime == regime and row.policy == "oracle"
        ]
        if not vals:
            return
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(regimes, means, yerr=stds, capsize=4)
    if means[0] > 0:
        reduction = 100.0 * (means[0] - means[1]) / means[0]
        ax.set_title(f"RMSE epsilon by regime ({reduction:.1f}% reduction)")
    else:
        ax.set_title("RMSE epsilon by regime")
    ax.set_ylabel("RMSE epsilon")
    fig.tight_layout()
    fig.savefig(root / "plots" / "motivation" / "epsilon_rmse_comparison.png", dpi=150)
    plt.close(fig)


def _plot_regret_vs_d(root: Path, rows: list[ReportResultRow]) -> None:
    policies = [
        "uniform",
        "epsilon_greedy",
        "online_softmax_best_arm",
        "linucb_raw",
        "softmax_best_arm",
        "cluster_label_router",
        "oracle",
    ]
    d_values = sorted({row.d for row in rows if row.block == "main"})
    if not d_values:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in policies:
        means = []
        stds = []
        for d in d_values:
            vals = [
                row.final_cum_regret
                for row in rows
                if row.block == "main" and row.policy == policy and row.d == d
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if vals else np.nan)
        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)
        ax.plot(d_values, means_arr, marker="o", label=policy)
        ax.fill_between(d_values, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d")
    ax.set_ylabel("final cumulative regret")
    ax.set_title("Final regret vs context dimension")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "plots" / "main" / "regret_vs_d.png", dpi=150)
    plt.close(fig)


def _plot_regret_curves_per_d(root: Path, artifacts: dict[str, Any]) -> None:
    d_values = sorted({run["d"] for run in artifacts["runs"].values() if run["block"] == "main"})
    if not d_values:
        return
    n_cols = min(3, len(d_values))
    n_rows = int(np.ceil(len(d_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.4 * n_rows), squeeze=False)
    policies = ["uniform", "epsilon_greedy", "online_softmax_best_arm", "linucb_raw", "softmax_best_arm"]
    styles = _policy_plot_styles()
    handles_by_label: dict[str, Any] = {}
    for ax, d in zip(axes.ravel(), d_values, strict=False):
        for policy in policies:
            curves = [
                run["policies"][policy]["cumulative_regret_curve"]
                for run in artifacts["runs"].values()
                if run["block"] == "main" and run["d"] == d and policy in run["policies"]
            ]
            if curves:
                style = styles[policy]
                (line,) = ax.plot(
                    np.asarray(curves, dtype=np.float64).mean(axis=0),
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                )
                handles_by_label[style["label"]] = line
        ax.set_title(f"d={d}")
        ax.set_xlabel("t")
        ax.set_ylabel("cumulative regret")
        ax.grid(True, alpha=0.25, linewidth=0.7)
    for ax in axes.ravel()[len(d_values) :]:
        fig.delaxes(ax)
    fig.suptitle("Mean Cumulative Regret Trajectories", y=0.995)
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=min(len(handles_by_label), 5),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(root / "plots" / "main" / "regret_curves_per_d.png", dpi=150)
    plt.close(fig)


def _plot_diagnostic_regret_curves_per_d(root: Path, artifacts: dict[str, Any]) -> None:
    d_values = sorted({run["d"] for run in artifacts["runs"].values() if run["block"] == "main"})
    if not d_values:
        return
    n_cols = min(3, len(d_values))
    n_rows = int(np.ceil(len(d_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.4 * n_rows), squeeze=False)
    policies = ["linucb_raw", "softmax_best_arm", "cluster_label_router"]
    styles = _policy_plot_styles()
    handles_by_label: dict[str, Any] = {}
    for ax, d in zip(axes.ravel(), d_values, strict=False):
        for policy in policies:
            curves = [
                run["policies"][policy]["cumulative_regret_curve"]
                for run in artifacts["runs"].values()
                if run["block"] == "main" and run["d"] == d and policy in run["policies"]
            ]
            if curves:
                style = styles[policy]
                (line,) = ax.plot(
                    np.asarray(curves, dtype=np.float64).mean(axis=0),
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                )
                handles_by_label[style["label"]] = line
        ax.set_title(f"d={d}")
        ax.set_xlabel("t")
        ax.set_ylabel("cumulative regret")
        ax.grid(True, alpha=0.25, linewidth=0.7)
    for ax in axes.ravel()[len(d_values) :]:
        fig.delaxes(ax)
    fig.suptitle("Routing Diagnostic Trajectories", y=0.995)
    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="lower center",
        ncol=min(len(handles_by_label), 3),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(root / "plots" / "supplementary" / "regret_curves_router_diagnostics_per_d.png", dpi=150)
    plt.close(fig)


def _policy_plot_styles() -> dict[str, dict[str, Any]]:
    return {
        "uniform": {
            "label": "Uniform",
            "color": "0.55",
            "linestyle": "--",
            "linewidth": 1.6,
            "alpha": 0.9,
        },
        "epsilon_greedy": {
            "label": "epsilon-greedy",
            "color": "tab:orange",
            "linestyle": ":",
            "linewidth": 1.8,
            "alpha": 0.95,
        },
        "online_softmax_best_arm": {
            "label": "Online softmax",
            "color": "tab:green",
            "linestyle": "-",
            "linewidth": 2.0,
            "alpha": 0.95,
        },
        "linucb_raw": {
            "label": "LinUCB",
            "color": "tab:red",
            "linestyle": "-",
            "linewidth": 2.2,
            "alpha": 0.95,
        },
        "softmax_best_arm": {
            "label": "Supervised softmax",
            "color": "tab:purple",
            "linestyle": "-.",
            "linewidth": 2.0,
            "alpha": 0.95,
        },
        "cluster_label_router": {
            "label": "Cluster-label router",
            "color": "tab:brown",
            "linestyle": "--",
            "linewidth": 2.0,
            "alpha": 0.95,
        },
        "oracle": {
            "label": "Oracle",
            "color": "tab:pink",
            "linestyle": "-",
            "linewidth": 1.5,
            "alpha": 0.8,
        },
    }


def _plot_alpha_sweep(root: Path, rows: list[ReportResultRow]) -> None:
    d_values = sorted({row.d for row in rows if row.block == "main"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for d in d_values:
        alpha_rows = [
            row
            for row in rows
            if row.block == "main" and row.d == d and row.policy.startswith("linucb_raw_alpha_")
        ]
        if not alpha_rows:
            continue
        alphas = sorted({float(row.linucb_alpha) for row in alpha_rows if row.linucb_alpha is not None})
        means = [
            float(np.mean([row.final_cum_regret for row in alpha_rows if row.linucb_alpha == alpha]))
            for alpha in alphas
        ]
        ax.plot(alphas, means, marker="o", label=f"d={d}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LinUCB alpha")
    ax.set_ylabel("final cumulative regret")
    ax.set_title("LinUCB alpha sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "plots" / "main" / "alpha_sweep.png", dpi=150)
    plt.close(fig)


def _plot_linucb_minus_softmax(root: Path, rows: list[ReportResultRow]) -> None:
    d_values = sorted({row.d for row in rows if row.block == "main"})
    if not d_values:
        return
    means = []
    stds = []
    for d in d_values:
        by_seed: list[float] = []
        seeds = sorted({row.seed_idx for row in rows if row.block == "main" and row.d == d})
        for seed_idx in seeds:
            lin = [
                row.final_cum_regret
                for row in rows
                if row.block == "main" and row.d == d and row.seed_idx == seed_idx and row.policy == "linucb_raw"
            ]
            sm = [
                row.final_cum_regret
                for row in rows
                if row.block == "main"
                and row.d == d
                and row.seed_idx == seed_idx
                and row.policy == "softmax_best_arm"
            ]
            if lin and sm:
                by_seed.append(float(lin[0] - sm[0]))
        means.append(float(np.mean(by_seed)) if by_seed else np.nan)
        stds.append(float(np.std(by_seed)) if by_seed else np.nan)
    means_arr = np.asarray(means)
    stds_arr = np.asarray(stds)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(d_values, means_arr, marker="o")
    ax.fill_between(d_values, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d")
    ax.set_ylabel("LinUCB regret - softmax regret")
    ax.set_title("LinUCB vs supervised softmax gap")
    fig.tight_layout()
    fig.savefig(root / "plots" / "main" / "linucb_minus_softmax_vs_d.png", dpi=150)
    plt.close(fig)


def _joint_stats_to_row(
    *, block: str, regime: str, d: int, seed_idx: int, seed: int, stats: JointTrainingStats
) -> dict[str, Any]:
    return {
        "block": block,
        "expert_regime": regime,
        "d": d,
        "seed_idx": seed_idx,
        "seed": seed,
        "history_total_loss": stats.history_total_loss,
        "history_ce_loss": stats.history_ce_loss,
        "history_load_loss": stats.history_load_loss,
        "history_train_acc": stats.history_train_acc,
        "history_val_acc": stats.history_val_acc,
        "history_lr": stats.history_lr,
        "history_gate_means": [arr.astype(float).tolist() for arr in stats.history_gate_means],
        "epochs_run": stats.epochs_run,
        "early_stopped": stats.early_stopped,
        "best_val_acc": stats.best_val_acc,
        "best_epoch_1based": stats.best_epoch_1based,
        "final_pooled_train_acc": stats.final_pooled_train_acc,
        "final_pooled_val_acc": stats.final_pooled_val_acc,
        "collapse_warning_early": stats.collapse_warning_early,
        "collapse_warning_messages": stats.collapse_warning_messages,
        "final_gate_means": None if stats.final_gate_means is None else stats.final_gate_means.astype(float).tolist(),
    }


def _linear_diagnostics(R: np.ndarray, X: np.ndarray, *, lambda_reg: float) -> dict[str, Any]:
    X_design = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    reg = lambda_reg * np.eye(X_design.shape[1], dtype=np.float64)
    xtx = X_design.T @ X_design + reg
    rmse_per_arm = np.zeros(R.shape[1], dtype=np.float64)
    max_resid_per_arm = np.zeros(R.shape[1], dtype=np.float64)
    theta_per_arm = np.zeros((R.shape[1], X_design.shape[1]), dtype=np.float64)
    for arm in range(R.shape[1]):
        theta = np.linalg.solve(xtx, X_design.T @ R[:, arm])
        residual = R[:, arm] - X_design @ theta
        rmse_per_arm[arm] = float(np.sqrt(np.mean(np.square(residual))))
        max_resid_per_arm[arm] = float(np.max(np.abs(residual)))
        theta_per_arm[arm] = theta
    return {
        "rmse_epsilon": float(np.mean(rmse_per_arm)),
        "max_epsilon": float(np.max(max_resid_per_arm)),
        "rmse_per_arm": rmse_per_arm.astype(float).tolist(),
        "max_resid_per_arm": max_resid_per_arm.astype(float).tolist(),
        "theta_per_arm": theta_per_arm.astype(float).tolist(),
        "ridge_lambda": lambda_reg,
        "fit_intercept": True,
    }


def _reward_heatmap(R: np.ndarray, cluster_id: np.ndarray, K: int) -> np.ndarray:
    heatmap = np.zeros((K, R.shape[1]), dtype=np.float64)
    for cluster in range(K):
        mask = cluster_id == cluster
        heatmap[cluster] = R[mask].mean(axis=0) if np.any(mask) else np.nan
    return heatmap


def _softmax_accuracy(policy: Any, X: np.ndarray, R: np.ndarray) -> float:
    labels = np.argmax(R, axis=1)
    preds = np.asarray([policy.select(x) for x in X], dtype=np.int64)
    return float(np.mean(preds == labels))


def _label_accuracy(policy: Any, X: np.ndarray, y: np.ndarray) -> float:
    preds = np.asarray([policy.select(x) for x in X], dtype=np.int64)
    return float(np.mean(preds == y))


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


def _prepare_dirs(root: Path) -> None:
    for rel in [
        "plots/motivation",
        "plots/main",
        "plots/supplementary",
        "raw",
        "logs",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, settings: ReportPacketSettings) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "git_commit": _git_commit(),
        "settings": asdict(settings),
    }
    _write_json(path, manifest)


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def _write_rows_csv(path: Path, rows: list[ReportResultRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the full report artifact packet.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/report_packet"),
        help="Directory where the timestamped report packet will be written.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write directly into --output-dir instead of creating a timestamped child directory.",
    )
    args = parser.parse_args()
    packet_dir = run_report_packet(args.output_dir, timestamped=not args.no_timestamp)
    print(f"Report packet written to: {packet_dir.resolve()}")


if __name__ == "__main__":
    main()
