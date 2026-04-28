from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import expert_reward_matrix
from moe_bandit.features import RBFFeatureMap
from moe_bandit.linear_approx_error import linear_approx_max_error
from moe_bandit.policies import (
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    UniformRandomPolicy,
    train_softmax_router,
)
from moe_bandit.runner import RunResult, run_bandit
from moe_bandit.train_overlapping_experts import train_overlapping_experts

logger = logging.getLogger(__name__)

RewardTarget = Literal["log_prob", "prob", "margin"]


@dataclass(frozen=True)
class RegularizationConfig:
    name: str
    weight_decay: float
    label_smoothing: float
    temperature: float


@dataclass(frozen=True)
class OverlapLinearitySettings:
    K: int = 4
    d: int = 10
    T: int = 10_000
    cluster_std: float = 1.0
    n_train_per_cluster: int = 2_000
    clip_eps: float = 1e-3
    linucb_alpha: float = 1.0
    linucb_lambda: float = 1.0
    forced_explore_per_arm: int = 20
    softmax_hidden_dim: int = 64
    softmax_epochs: int = 300
    softmax_batch_size: int = 64
    softmax_lr: float = 1e-3
    epsilon_greedy_c: float = 50.0
    rbf_gamma: float = 0.5
    include_learned_feature_policy: bool = True
    overlap_strengths: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)
    cluster_seps: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    seed_bundles: tuple[tuple[int, int, int, int], ...] = (
        (101, 201, 301, 401),
        (102, 202, 302, 402),
        (103, 203, 303, 403),
    )
    regularization_configs: tuple[RegularizationConfig, ...] = (
        RegularizationConfig("baseline", 0.0, 0.0, 1.0),
        RegularizationConfig("regularized", 1e-4, 0.05, 1.0),
        RegularizationConfig("calibrated", 1e-4, 0.05, 2.0),
    )
    reward_targets: tuple[RewardTarget, ...] = ("log_prob",)


@dataclass
class OverlapLinearityRow:
    overlap_strength: float
    cluster_sep: float
    seed_idx: int
    regularization_config: str
    reward_target: str
    policy: str
    final_cum_regret: float
    avg_regret: float
    chosen_arm_mean_reward: float
    best_arm_acc: float
    mean_epsilon: float
    max_epsilon: float
    mean_rmse_epsilon: float
    max_rmse_epsilon: float
    mean_p95_abs_epsilon: float
    max_p95_abs_epsilon: float
    oracle_gap_mean: float
    cumulative_oracle_gap: float
    normalized_regret: float
    reward_range: float
    relative_epsilon: float


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


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _rescale_rewards_01(R_raw: np.ndarray) -> np.ndarray:
    r_min = float(np.min(R_raw))
    r_max = float(np.max(R_raw))
    if r_max - r_min <= 1e-12:
        return np.zeros_like(R_raw, dtype=np.float64)
    return ((R_raw - r_min) / (r_max - r_min)).astype(np.float64)


def _zscore_with_stream_stats(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-8, None)
    return ((X - mean) / std).astype(np.float64)


def _confusion_matrix(chosen_arm: np.ndarray, oracle_arm: np.ndarray, K: int) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    for truth, pred in zip(oracle_arm, chosen_arm, strict=False):
        cm[int(truth), int(pred)] += 1
    return cm


def _oracle_by_cluster(oracle_arm: np.ndarray, y_true: np.ndarray, K: int) -> np.ndarray:
    table = np.zeros((K, K), dtype=np.int64)
    for cluster, arm in zip(y_true, oracle_arm, strict=False):
        table[int(cluster), int(arm)] += 1
    return table


def summarize_reward_gap(R: np.ndarray) -> dict[str, float]:
    sorted_rewards = np.sort(R, axis=1)
    gap = sorted_rewards[:, -1] - sorted_rewards[:, -2]
    return {
        "mean": float(np.mean(gap)),
        "median": float(np.median(gap)),
        "p10": float(np.quantile(gap, 0.10)),
        "p90": float(np.quantile(gap, 0.90)),
    }


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits_shifted - np.log(np.exp(logits_shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def np_to_torch(X: np.ndarray, device: Any) -> Any:
    import torch

    return torch.as_tensor(X, dtype=torch.float32, device=device)


def _row_from_result(
    result: RunResult,
    *,
    overlap_strength: float,
    cluster_sep: float,
    seed_idx: int,
    regularization_config: str,
    reward_target: str,
    mean_epsilon: float,
    max_epsilon: float,
    mean_rmse_epsilon: float,
    max_rmse_epsilon: float,
    mean_p95_abs_epsilon: float,
    max_p95_abs_epsilon: float,
    oracle_gap_mean: float,
    cumulative_oracle_gap: float,
    reward_range: float,
    policy: str,
) -> OverlapLinearityRow:
    cum_regret = result.cumulative_regret()
    best_arm_acc = float(np.mean(result.chosen_arm == result.oracle_arm))
    return OverlapLinearityRow(
        overlap_strength=overlap_strength,
        cluster_sep=cluster_sep,
        seed_idx=seed_idx,
        regularization_config=regularization_config,
        reward_target=reward_target,
        policy=policy,
        final_cum_regret=float(cum_regret[-1]),
        avg_regret=float(result.regret.mean()),
        chosen_arm_mean_reward=float(result.reward.mean()),
        best_arm_acc=best_arm_acc,
        mean_epsilon=mean_epsilon,
        max_epsilon=max_epsilon,
        mean_rmse_epsilon=mean_rmse_epsilon,
        max_rmse_epsilon=max_rmse_epsilon,
        mean_p95_abs_epsilon=mean_p95_abs_epsilon,
        max_p95_abs_epsilon=max_p95_abs_epsilon,
        oracle_gap_mean=oracle_gap_mean,
        cumulative_oracle_gap=cumulative_oracle_gap,
        normalized_regret=float(cum_regret[-1] / max(cumulative_oracle_gap, 1e-12)),
        reward_range=reward_range,
        relative_epsilon=float(max_epsilon / max(reward_range, 1e-12)),
    )


def _write_rows_csv(path: Path, rows: list[OverlapLinearityRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_rows_jsonl(path: Path, rows: list[OverlapLinearityRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def _write_artifacts_json(path: Path, artifacts: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(artifacts, f)


def _aggregate_metric(
    rows: list[OverlapLinearityRow],
    *,
    policy: str,
    metric: str,
    overlap_strengths: tuple[float, ...],
    cluster_seps: tuple[float, ...],
    regularization_config: str,
    reward_target: str,
) -> np.ndarray:
    arr = np.full((len(cluster_seps), len(overlap_strengths)), np.nan, dtype=np.float64)
    for y_idx, sep in enumerate(cluster_seps):
        for x_idx, ov in enumerate(overlap_strengths):
            vals = [
                getattr(row, metric)
                for row in rows
                if row.policy == policy
                and row.regularization_config == regularization_config
                and row.reward_target == reward_target
                and abs(row.cluster_sep - sep) < 1e-12
                and abs(row.overlap_strength - ov) < 1e-12
            ]
            arr[y_idx, x_idx] = float(np.mean(vals)) if vals else np.nan
    return arr


def _plot_heatmap(
    data: np.ndarray,
    title: str,
    overlap_strengths: tuple[float, ...],
    cluster_seps: tuple[float, ...],
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(overlap_strengths)))
    ax.set_xticklabels([f"{v:.2f}" for v in overlap_strengths])
    ax.set_yticks(range(len(cluster_seps)))
    ax.set_yticklabels([f"{v:.1f}" for v in cluster_seps])
    ax.set_xlabel("overlap_strength")
    ax.set_ylabel("cluster_sep")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_regret_vs_epsilon(rows: list[OverlapLinearityRow], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [
        row.max_epsilon
        for row in rows
        if row.policy == "linucb_raw"
    ]
    ys = [
        row.final_cum_regret
        for row in rows
        if row.policy == "linucb_raw"
    ]
    ax.scatter(xs, ys, alpha=0.65)
    ax.set_xlabel("max_epsilon")
    ax.set_ylabel("linucb final cumulative regret")
    ax.set_title("LinUCB Regret vs Approximation Error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_regret_curves_representative(
    artifacts: dict[str, Any],
    output_path: Path,
    *,
    regularization_config: str,
    reward_target: str,
    representative_cells: tuple[tuple[float, float], ...] = ((0.25, 1.0), (0.50, 2.0), (0.90, 3.0)),
) -> None:
    fig, axes = plt.subplots(1, len(representative_cells), figsize=(15, 4), sharey=True)
    if len(representative_cells) == 1:
        axes = [axes]
    policy_order = [
        "linucb_raw",
        "linucb_rbf",
        "linucb_router_feat",
        "epsilon_greedy",
        "softmax_router",
        "oracle",
    ]

    for idx, (ov, sep) in enumerate(representative_cells):
        ax = axes[idx]
        for policy in policy_order:
            curves: list[list[float]] = []
            for run in artifacts.get("runs", {}).values():
                if run.get("regularization_config") != regularization_config:
                    continue
                if run.get("reward_target") != reward_target:
                    continue
                if abs(float(run.get("overlap_strength", -1.0)) - ov) > 1e-12:
                    continue
                if abs(float(run.get("cluster_sep", -1.0)) - sep) > 1e-12:
                    continue
                pol = run.get("policies", {}).get(policy, {})
                curve = pol.get("cumulative_regret_curve")
                if curve:
                    curves.append(curve)
            if not curves:
                continue
            mean_curve = np.asarray(curves, dtype=np.float64).mean(axis=0)
            ax.plot(mean_curve, label=policy)
        ax.set_title(f"ov={ov:.2f}, sep={sep:.1f}")
        ax.set_xlabel("t")
        if idx == 0:
            ax.set_ylabel("cumulative regret")
        if ax.lines:
            ax.legend()

    fig.suptitle(f"Regret Curves ({regularization_config}, {reward_target})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def regenerate_overlap_linearity_plots(
    output_dir: str | Path,
    settings: OverlapLinearitySettings = OverlapLinearitySettings(),
) -> None:
    """
    Rebuild overlap-linearity plots from existing results_rows.jsonl/artifacts.json.
    """
    out = Path(output_dir)
    rows_path = out / "results_rows.jsonl"
    artifacts_path = out / "artifacts.json"
    if not rows_path.exists() or not artifacts_path.exists():
        raise FileNotFoundError(
            "Missing results_rows.jsonl or artifacts.json for plot regeneration."
        )
    rows: list[OverlapLinearityRow] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        obj.setdefault("mean_rmse_epsilon", float("nan"))
        obj.setdefault("max_rmse_epsilon", float("nan"))
        obj.setdefault("mean_p95_abs_epsilon", float("nan"))
        obj.setdefault("max_p95_abs_epsilon", float("nan"))
        obj.setdefault("cumulative_oracle_gap", float("nan"))
        obj.setdefault("normalized_regret", float("nan"))
        obj.setdefault("reward_range", float("nan"))
        obj.setdefault("relative_epsilon", float("nan"))
        rows.append(OverlapLinearityRow(**obj))
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))

    for regularization in settings.regularization_configs:
        for reward_target in settings.reward_targets:
            linucb_regret = _aggregate_metric(
                rows,
                policy="linucb_raw",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            softmax_regret = _aggregate_metric(
                rows,
                policy="softmax_router",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            max_eps = _aggregate_metric(
                rows,
                policy="oracle",
                metric="max_epsilon",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            linucb_rbf_regret = _aggregate_metric(
                rows,
                policy="linucb_rbf",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            linucb_norm_regret = _aggregate_metric(
                rows,
                policy="linucb_raw",
                metric="normalized_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            linucb_rbf_regret = _aggregate_metric(
                rows,
                policy="linucb_rbf",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            linucb_norm_regret = _aggregate_metric(
                rows,
                policy="linucb_raw",
                metric="normalized_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            sfx = f"_{regularization.name}_{reward_target}"
            _plot_heatmap(
                linucb_regret,
                f"LinUCB final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_regret{sfx}.png",
            )
            _plot_heatmap(
                softmax_regret,
                f"Softmax final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_softmax_regret{sfx}.png",
            )
            _plot_heatmap(
                linucb_rbf_regret,
                f"LinUCB RBF final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_rbf_regret{sfx}.png",
            )
            _plot_heatmap(
                linucb_norm_regret,
                f"LinUCB normalized regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_normalized_regret{sfx}.png",
            )
            _plot_heatmap(
                linucb_rbf_regret,
                f"LinUCB RBF final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_rbf_regret{sfx}.png",
            )
            _plot_heatmap(
                linucb_norm_regret,
                f"LinUCB normalized regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_normalized_regret{sfx}.png",
            )
            _plot_heatmap(
                max_eps,
                f"Max epsilon ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_max_epsilon{sfx}.png",
            )
            _plot_regret_curves_representative(
                artifacts,
                out / f"regret_curves_representative{sfx}.png",
                regularization_config=regularization.name,
                reward_target=reward_target,
            )

    _plot_regret_vs_epsilon(rows, out / "scatter_linucb_regret_vs_epsilon.png")

def _compute_linearity_summary(path: Path, rows: list[OverlapLinearityRow]) -> None:
    keys = sorted(
        {
            (
                row.overlap_strength,
                row.cluster_sep,
                row.regularization_config,
                row.reward_target,
            )
            for row in rows
        }
    )
    fieldnames = [
        "overlap_strength",
        "cluster_sep",
        "regularization_config",
        "reward_target",
        "mean_epsilon",
        "max_epsilon",
        "mean_rmse_epsilon",
        "max_rmse_epsilon",
        "mean_p95_abs_epsilon",
        "max_p95_abs_epsilon",
        "relative_epsilon",
        "linucb_regret",
        "linucb_rbf_regret",
        "linucb_router_feat_regret",
        "linucb_normalized_regret",
        "softmax_regret",
        "oracle_gap_mean",
        "cumulative_oracle_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ov, sep, reg, reward_t in keys:
            group = [
                row
                for row in rows
                if abs(row.overlap_strength - ov) < 1e-12
                and abs(row.cluster_sep - sep) < 1e-12
                and row.regularization_config == reg
                and row.reward_target == reward_t
            ]
            linucb_vals = [r.final_cum_regret for r in group if r.policy == "linucb_raw"]
            linucb_rbf_vals = [r.final_cum_regret for r in group if r.policy == "linucb_rbf"]
            linucb_router_feat_vals = [
                r.final_cum_regret for r in group if r.policy == "linucb_router_feat"
            ]
            linucb_norm_vals = [r.normalized_regret for r in group if r.policy == "linucb_raw"]
            softmax_vals = [r.final_cum_regret for r in group if r.policy == "softmax_router"]
            eps_vals = [r.mean_epsilon for r in group if r.policy == "oracle"]
            eps_max_vals = [r.max_epsilon for r in group if r.policy == "oracle"]
            rmse_vals = [r.mean_rmse_epsilon for r in group if r.policy == "oracle"]
            rmse_max_vals = [r.max_rmse_epsilon for r in group if r.policy == "oracle"]
            p95_vals = [r.mean_p95_abs_epsilon for r in group if r.policy == "oracle"]
            p95_max_vals = [r.max_p95_abs_epsilon for r in group if r.policy == "oracle"]
            rel_eps_vals = [r.relative_epsilon for r in group if r.policy == "oracle"]
            gap_vals = [r.oracle_gap_mean for r in group if r.policy == "oracle"]
            cum_gap_vals = [r.cumulative_oracle_gap for r in group if r.policy == "oracle"]
            writer.writerow(
                {
                    "overlap_strength": ov,
                    "cluster_sep": sep,
                    "regularization_config": reg,
                    "reward_target": reward_t,
                    "mean_epsilon": float(np.mean(eps_vals)),
                    "max_epsilon": float(np.mean(eps_max_vals)),
                    "mean_rmse_epsilon": float(np.mean(rmse_vals)),
                    "max_rmse_epsilon": float(np.mean(rmse_max_vals)),
                    "mean_p95_abs_epsilon": float(np.mean(p95_vals)),
                    "max_p95_abs_epsilon": float(np.mean(p95_max_vals)),
                    "relative_epsilon": float(np.mean(rel_eps_vals)),
                    "linucb_regret": float(np.mean(linucb_vals)),
                    "linucb_rbf_regret": float(np.mean(linucb_rbf_vals))
                    if linucb_rbf_vals
                    else float("nan"),
                    "linucb_router_feat_regret": float(np.mean(linucb_router_feat_vals))
                    if linucb_router_feat_vals
                    else float("nan"),
                    "linucb_normalized_regret": float(np.mean(linucb_norm_vals)),
                    "softmax_regret": float(np.mean(softmax_vals)),
                    "oracle_gap_mean": float(np.mean(gap_vals)),
                    "cumulative_oracle_gap": float(np.mean(cum_gap_vals)),
                }
            )


def _epsilon_regret_corr(rows: list[OverlapLinearityRow]) -> float:
    pairs = [
        (row.max_epsilon, row.final_cum_regret)
        for row in rows
        if row.policy == "linucb_raw"
    ]
    if len(pairs) < 2:
        return float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def run_overlap_linearity_experiment(
    output_dir: str | Path,
    settings: OverlapLinearitySettings = OverlapLinearitySettings(),
    *,
    log_level: int = logging.WARNING,
) -> tuple[list[OverlapLinearityRow], dict[str, Any]]:
    configure_logging(log_level)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[OverlapLinearityRow] = []
    artifacts: dict[str, Any] = {
        "settings": asdict(settings),
        "runs": {},
    }

    n_train = settings.K * settings.n_train_per_cluster
    total_cells = (
        len(settings.overlap_strengths)
        * len(settings.cluster_seps)
        * len(settings.seed_bundles)
        * len(settings.regularization_configs)
        * len(settings.reward_targets)
    )
    cell_idx = 0
    for regularization in settings.regularization_configs:
        for reward_target in settings.reward_targets:
            for ov in settings.overlap_strengths:
                for sep in settings.cluster_seps:
                    for seed_idx, seeds in enumerate(settings.seed_bundles):
                        cell_idx += 1
                        seed_train_data, seed_train_experts, seed_bandit_stream, seed_policy = seeds
                        logger.info(
                            "[%d/%d] overlap=%.2f sep=%.1f seed=%d reg=%s reward=%s",
                            cell_idx,
                            total_cells,
                            ov,
                            sep,
                            seed_idx,
                            regularization.name,
                            reward_target,
                        )
                        X_train, y_train, cluster_train = generate_synthetic_data(
                            n_samples=n_train,
                            K=settings.K,
                            d=settings.d,
                            cluster_sep=sep,
                            cluster_std=settings.cluster_std,
                            seed=seed_train_data,
                        )
                        X_bandit, y_bandit, _ = generate_synthetic_data(
                            n_samples=settings.T,
                            K=settings.K,
                            d=settings.d,
                            cluster_sep=sep,
                            cluster_std=settings.cluster_std,
                            seed=seed_bandit_stream,
                        )
                        X_bandit_std = _zscore_with_stream_stats(X_bandit)

                        experts, overlap_stats = train_overlapping_experts(
                            X_train=X_train,
                            y_train=y_train,
                            cluster_id_train=cluster_train,
                            K=settings.K,
                            d=settings.d,
                            overlap_strength=ov,
                            epochs=30,
                            lr=1e-3,
                            batch_size=64,
                            seed=seed_train_experts,
                            weight_decay=regularization.weight_decay,
                            label_smoothing=regularization.label_smoothing,
                        )
                        R_raw = expert_reward_matrix(
                            experts=experts,
                            X=X_bandit,
                            y=y_bandit,
                            clip_eps=settings.clip_eps,
                            reward_type=reward_target,
                            temperature=regularization.temperature,
                        )
                        R = _rescale_rewards_01(R_raw)

                        approx_rep = linear_approx_max_error(
                            R_raw,
                            X_bandit,
                            lambda_reg=settings.linucb_lambda,
                            fit_intercept=True,
                        )
                        gap_summary = summarize_reward_gap(R)
                        reward_range = float(np.max(R_raw) - np.min(R_raw))
                        cumulative_oracle_gap = float(
                            np.sum(np.max(R, axis=1) - np.min(R, axis=1))
                        )

                        # Build RBF centers from observed training clusters.
                        centers = np.zeros((settings.K, settings.d), dtype=np.float64)
                        for c in range(settings.K):
                            mask = cluster_train == c
                            if not np.any(mask):
                                raise ValueError(f"No points for cluster {c} while building centers.")
                            centers[c] = X_train[mask].mean(axis=0)
                        rbf_map = RBFFeatureMap(centers=centers, gamma=settings.rbf_gamma)

                        softmax_router = train_softmax_router(
                            X_train=X_bandit_std,
                            R_train=R,
                            hidden_dim=settings.softmax_hidden_dim,
                            epochs=settings.softmax_epochs,
                            batch_size=settings.softmax_batch_size,
                            lr=settings.softmax_lr,
                            seed=seed_policy,
                        )
                        with np.errstate(invalid="ignore"):
                            logits = softmax_router.model(
                                np_to_torch(X_bandit_std, device=softmax_router.device)
                            ).detach()
                            train_labels = np.argmax(R, axis=1)
                            train_acc = float(
                                (logits.argmax(dim=1).cpu().numpy() == train_labels).mean()
                            )
                            train_loss = cross_entropy_from_logits(
                                logits.cpu().numpy(), train_labels
                            )

                        def router_prob_feature_map(x_t: np.ndarray) -> np.ndarray:
                            import torch

                            with torch.no_grad():
                                logits_t = softmax_router.model(
                                    np_to_torch(x_t[None, :], device=softmax_router.device)
                                ).detach()
                            probs_t = np.exp(
                                logits_t.cpu().numpy() - logits_t.cpu().numpy().max(axis=1, keepdims=True)
                            )
                            probs_t /= np.clip(probs_t.sum(axis=1, keepdims=True), 1e-12, None)
                            return probs_t[0].astype(np.float64)

                        oracle_arm = np.argmax(R, axis=1)
                        policies = {
                            "uniform": UniformRandomPolicy(K=settings.K, seed=seed_policy),
                            "epsilon_greedy": EpsilonGreedyPolicy(
                                K=settings.K, c=settings.epsilon_greedy_c, seed=seed_policy
                            ),
                            "linucb_raw": LinUCBPolicy(
                                K=settings.K,
                                d=settings.d,
                                alpha=settings.linucb_alpha,
                                lambda_reg=settings.linucb_lambda,
                                forced_explore_per_arm=settings.forced_explore_per_arm,
                                seed=seed_policy,
                            ),
                            "linucb_rbf": LinUCBPolicy(
                                K=settings.K,
                                d=settings.d,
                                alpha=settings.linucb_alpha,
                                lambda_reg=settings.linucb_lambda,
                                forced_explore_per_arm=settings.forced_explore_per_arm,
                                seed=seed_policy + 17,
                                feature_map=rbf_map,
                                feature_dim=rbf_map.d_out,
                                add_intercept=True,
                            ),
                            "softmax_router": softmax_router,
                            "oracle": OraclePolicy(oracle_arm=oracle_arm),
                        }
                        if settings.include_learned_feature_policy:
                            policies["linucb_router_feat"] = LinUCBPolicy(
                                K=settings.K,
                                d=settings.d,
                                alpha=settings.linucb_alpha,
                                lambda_reg=settings.linucb_lambda,
                                forced_explore_per_arm=settings.forced_explore_per_arm,
                                seed=seed_policy + 29,
                                feature_map=router_prob_feature_map,
                                feature_dim=settings.K,
                                add_intercept=True,
                            )

                        run_key = (
                            f"ov_{ov:.2f}_sep_{sep:.1f}_seed_{seed_idx}"
                            f"__{regularization.name}__{reward_target}"
                        )
                        run_store: dict[str, Any] = {
                            "overlap_strength": ov,
                            "cluster_sep": sep,
                            "seed_idx": seed_idx,
                            "regularization_config": regularization.name,
                            "reward_target": reward_target,
                            "seed_bundle": {
                                "train_data": seed_train_data,
                                "train_experts": seed_train_experts,
                                "bandit_stream": seed_bandit_stream,
                                "policy": seed_policy,
                            },
                            "expert_training_diagnostics": {
                                "mixture_weights": overlap_stats.mixture_weights.astype(float).tolist(),
                                "sampled_cluster_proportions": overlap_stats.sampled_cluster_proportions.astype(
                                    float
                                ).tolist(),
                                "own_cluster_accuracy": overlap_stats.own_cluster_accuracy.astype(float).tolist(),
                                "cross_cluster_accuracy": overlap_stats.cross_cluster_accuracy.astype(float).tolist(),
                            },
                            "softmax_train": {"accuracy": train_acc, "loss": train_loss},
                            "reward_gap_summary": gap_summary,
                            "oracle_by_cluster": _oracle_by_cluster(
                                oracle_arm=oracle_arm, y_true=y_bandit, K=settings.K
                            ).tolist(),
                            "approx_error": {
                                "mean_epsilon": approx_rep.mean_epsilon,
                                "max_epsilon": approx_rep.max_epsilon,
                                "mean_rmse": approx_rep.mean_rmse,
                                "max_rmse": approx_rep.max_rmse,
                                "mean_p95_abs": approx_rep.mean_p95_abs,
                                "max_p95_abs": approx_rep.max_p95_abs,
                                "epsilon_per_arm": approx_rep.epsilon_per_arm.astype(float).tolist(),
                                "rmse_per_arm": approx_rep.rmse_per_arm.astype(float).tolist(),
                                "p95_abs_per_arm": approx_rep.p95_abs_per_arm.astype(float).tolist(),
                                "relative_epsilon": float(
                                    approx_rep.max_epsilon / max(reward_range, 1e-12)
                                ),
                            },
                            "policies": {},
                        }

                        for policy_name, policy_obj in policies.items():
                            result = run_bandit(
                                policy=policy_obj, X=X_bandit_std, R=R, seed=seed_policy
                            )
                            row = _row_from_result(
                                result,
                                overlap_strength=ov,
                                cluster_sep=sep,
                                seed_idx=seed_idx,
                                regularization_config=regularization.name,
                                reward_target=reward_target,
                                mean_epsilon=approx_rep.mean_epsilon,
                                max_epsilon=approx_rep.max_epsilon,
                                mean_rmse_epsilon=approx_rep.mean_rmse,
                                max_rmse_epsilon=approx_rep.max_rmse,
                                mean_p95_abs_epsilon=approx_rep.mean_p95_abs,
                                max_p95_abs_epsilon=approx_rep.max_p95_abs,
                                oracle_gap_mean=gap_summary["mean"],
                                cumulative_oracle_gap=cumulative_oracle_gap,
                                reward_range=reward_range,
                                policy=policy_name,
                            )
                            rows.append(row)
                            run_store["policies"][policy_name] = {
                                "confusion_matrix": _confusion_matrix(
                                    chosen_arm=result.chosen_arm,
                                    oracle_arm=result.oracle_arm,
                                    K=settings.K,
                                ).tolist(),
                                "cumulative_regret_curve": result.cumulative_regret()
                                .astype(float)
                                .tolist(),
                            }
                        artifacts["runs"][run_key] = run_store

    _write_rows_csv(out / "results_rows.csv", rows)
    _write_rows_jsonl(out / "results_rows.jsonl", rows)
    _write_artifacts_json(out / "artifacts.json", artifacts)
    _compute_linearity_summary(out / "linearity_summary.csv", rows)

    for regularization in settings.regularization_configs:
        for reward_target in settings.reward_targets:
            linucb_regret = _aggregate_metric(
                rows,
                policy="linucb_raw",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            softmax_regret = _aggregate_metric(
                rows,
                policy="softmax_router",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            max_eps = _aggregate_metric(
                rows,
                policy="oracle",
                metric="max_epsilon",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            linucb_rbf_regret = _aggregate_metric(
                rows,
                policy="linucb_rbf",
                metric="final_cum_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            linucb_norm_regret = _aggregate_metric(
                rows,
                policy="linucb_raw",
                metric="normalized_regret",
                overlap_strengths=settings.overlap_strengths,
                cluster_seps=settings.cluster_seps,
                regularization_config=regularization.name,
                reward_target=reward_target,
            )
            sfx = f"_{regularization.name}_{reward_target}"
            _plot_heatmap(
                linucb_regret,
                f"LinUCB final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_regret{sfx}.png",
            )
            _plot_heatmap(
                softmax_regret,
                f"Softmax final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_softmax_regret{sfx}.png",
            )
            _plot_heatmap(
                linucb_rbf_regret,
                f"LinUCB RBF final regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_rbf_regret{sfx}.png",
            )
            _plot_heatmap(
                linucb_norm_regret,
                f"LinUCB normalized regret ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_linucb_normalized_regret{sfx}.png",
            )
            _plot_heatmap(
                max_eps,
                f"Max epsilon ({regularization.name}, {reward_target})",
                settings.overlap_strengths,
                settings.cluster_seps,
                out / f"heatmap_max_epsilon{sfx}.png",
            )
            _plot_regret_curves_representative(
                artifacts,
                out / f"regret_curves_representative{sfx}.png",
                regularization_config=regularization.name,
                reward_target=reward_target,
            )

    _plot_regret_vs_epsilon(rows, out / "scatter_linucb_regret_vs_epsilon.png")
    artifacts["diagnostics"] = {"epsilon_regret_correlation": _epsilon_regret_corr(rows)}
    _write_artifacts_json(out / "artifacts.json", artifacts)

    return rows, artifacts
