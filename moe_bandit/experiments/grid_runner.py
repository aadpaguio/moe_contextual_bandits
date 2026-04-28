from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import Expert, expert_reward_matrix, train_experts
from moe_bandit.linear_approx_error import linear_approx_max_error
from moe_bandit.train_joint_moe import train_joint_moe
from moe_bandit.policies import (
    EpsilonGreedyPolicy,
    LinUCBPolicy,
    UniformRandomPolicy,
    train_softmax_router,
)
from moe_bandit.runner import RunResult, run_bandit

logger = logging.getLogger(__name__)

ExpertTrainingRegime = Literal["independent", "joint"]


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once (safe if called multiple times).
    """
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


@dataclass(frozen=True)
class FixedSettings:
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
    # Joint MoE expert training (used when regime is "joint")
    seed_joint_expert_offset: int = 10_000
    """Joint regime uses ``seed_train_experts + offset`` so init differs from independent."""
    joint_moe_max_epochs: int = 200
    joint_moe_alpha_load: float = 0.001
    joint_moe_lr_min: float = 1e-5
    joint_moe_cosine_decay: bool = True
    joint_moe_early_stopping_patience: int | None = 20
    joint_moe_batch_size: int = 64


@dataclass(frozen=True)
class GridConfig:
    contamination: float
    cluster_sep: float
    seed_idx: int
    seed_train_data: int
    seed_train_experts: int
    seed_bandit_stream: int
    seed_policy: int

    @property
    def cfg_name(self) -> str:
        return (
            f"contam_{self.contamination:.2f}_sep_{self.cluster_sep:.1f}_seed_{self.seed_idx}"
        )


@dataclass
class ResultRow:
    contamination: float
    cluster_sep: float
    seed_idx: int
    expert_regime: str
    policy: str
    final_cum_regret: float
    avg_regret: float
    chosen_arm_mean_reward: float
    best_arm_acc: float


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


def _build_grid(
    contaminations: list[float],
    cluster_seps: list[float],
    seed_bundles: list[tuple[int, int, int, int]],
) -> list[GridConfig]:
    configs: list[GridConfig] = []
    for contamination in contaminations:
        for cluster_sep in cluster_seps:
            for seed_idx, bundle in enumerate(seed_bundles):
                seed_train_data, seed_train_experts, seed_bandit_stream, seed_policy = bundle
                configs.append(
                    GridConfig(
                        contamination=contamination,
                        cluster_sep=cluster_sep,
                        seed_idx=seed_idx,
                        seed_train_data=seed_train_data,
                        seed_train_experts=seed_train_experts,
                        seed_bandit_stream=seed_bandit_stream,
                        seed_policy=seed_policy,
                    )
                )
    return configs


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


def _confusion_matrix(chosen_arm: np.ndarray, oracle_arm: np.ndarray, K: int) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    for truth, pred in zip(oracle_arm, chosen_arm, strict=False):
        cm[int(truth), int(pred)] += 1
    return cm


def _row_from_result(
    config: GridConfig,
    expert_regime: ExpertTrainingRegime,
    policy: str,
    result: RunResult,
) -> ResultRow:
    cum_regret = result.cumulative_regret()
    best_arm_acc = float(np.mean(result.chosen_arm == result.oracle_arm))
    return ResultRow(
        contamination=config.contamination,
        cluster_sep=config.cluster_sep,
        seed_idx=config.seed_idx,
        expert_regime=expert_regime,
        policy=policy,
        final_cum_regret=float(cum_regret[-1]),
        avg_regret=float(result.regret.mean()),
        chosen_arm_mean_reward=float(result.reward.mean()),
        best_arm_acc=best_arm_acc,
    )


def _train_experts_for_regime(
    regime: ExpertTrainingRegime,
    *,
    settings: FixedSettings,
    cfg: GridConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_train: np.ndarray,
) -> list[Expert]:
    if regime == "independent":
        return train_experts(
            X_train=X_train,
            y_train=y_train,
            cluster_id_train=cluster_train,
            K=settings.K,
            d=settings.d,
            epochs=30,
            lr=1e-3,
            batch_size=64,
            seed=cfg.seed_train_experts,
            contamination=cfg.contamination,
        )
    seed_joint = cfg.seed_train_experts + settings.seed_joint_expert_offset
    experts, _stats = train_joint_moe(
        X_train=X_train,
        y_train=y_train,
        cluster_id_train=cluster_train,
        K=settings.K,
        d=settings.d,
        epochs=settings.joint_moe_max_epochs,
        lr=1e-3,
        lr_min=settings.joint_moe_lr_min,
        cosine_decay=settings.joint_moe_cosine_decay,
        batch_size=settings.joint_moe_batch_size,
        seed=seed_joint,
        alpha_load=settings.joint_moe_alpha_load,
        early_stopping_patience=settings.joint_moe_early_stopping_patience,
        router="linear",
    )
    return experts


def run_main_grid(
    output_dir: str | Path,
    settings: FixedSettings = FixedSettings(),
    contaminations: list[float] | None = None,
    cluster_seps: list[float] | None = None,
    seed_bundles: list[tuple[int, int, int, int]] | None = None,
    expert_training_regimes: tuple[ExpertTrainingRegime, ...] = ("independent",),
    write_approx_error_jsonl: bool = False,
    log_level: int = logging.WARNING,
) -> tuple[list[ResultRow], dict[str, Any]]:
    contaminations = contaminations or [0.05, 0.10, 0.20, 0.30, 0.50]
    cluster_seps = cluster_seps or [1.0, 1.5, 2.0, 3.0]
    seed_bundles = seed_bundles or [
        (101, 201, 301, 401),
        (102, 202, 302, 402),
        (103, 203, 303, 403),
    ]

    configure_logging(log_level)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Starting grid: output_dir=%s", out.resolve())

    rows: list[ResultRow] = []
    artifacts: dict[str, Any] = {
        "settings": asdict(settings),
        "grid": {
            "contaminations": contaminations,
            "cluster_seps": cluster_seps,
            "seed_bundles": seed_bundles,
            "expert_training_regimes": list(expert_training_regimes),
        },
        "runs": {},
    }
    approx_lines: list[dict[str, Any]] = []

    grid = _build_grid(contaminations, cluster_seps, seed_bundles)
    n_train = settings.K * settings.n_train_per_cluster
    n_cells = len(grid)
    logger.info(
        "Grid size: %d cells (contaminations=%d × seps=%d × seeds=%d), T=%d, n_train=%d",
        n_cells,
        len(contaminations),
        len(cluster_seps),
        len(seed_bundles),
        settings.T,
        n_train,
    )

    for cell_idx, cfg in enumerate(grid, start=1):
        t_cell0 = time.perf_counter()
        logger.info(
            "[%d/%d] %s | contam=%.2f sep=%.1f | seeds train_data=%d train_experts=%d bandit=%d policy=%d",
            cell_idx,
            n_cells,
            cfg.cfg_name,
            cfg.contamination,
            cfg.cluster_sep,
            cfg.seed_train_data,
            cfg.seed_train_experts,
            cfg.seed_bandit_stream,
            cfg.seed_policy,
        )

        t0 = time.perf_counter()
        X_train, y_train, cluster_train = generate_synthetic_data(
            n_samples=n_train,
            K=settings.K,
            d=settings.d,
            cluster_sep=cfg.cluster_sep,
            cluster_std=settings.cluster_std,
            seed=cfg.seed_train_data,
        )
        logger.debug("Generated train data in %.3fs shape=%s", time.perf_counter() - t0, X_train.shape)

        t0 = time.perf_counter()
        X_bandit, y_bandit, _ = generate_synthetic_data(
            n_samples=settings.T,
            K=settings.K,
            d=settings.d,
            cluster_sep=cfg.cluster_sep,
            cluster_std=settings.cluster_std,
            seed=cfg.seed_bandit_stream,
        )
        logger.debug("Generated bandit stream in %.3fs shape=%s", time.perf_counter() - t0, X_bandit.shape)
        X_train_std = _zscore_from_reference(X_train, X_train)
        X_bandit_std = _zscore_from_reference(X_train, X_bandit)

        for regime in expert_training_regimes:
            t0 = time.perf_counter()
            experts = _train_experts_for_regime(
                regime,
                settings=settings,
                cfg=cfg,
                X_train=X_train,
                y_train=y_train,
                cluster_train=cluster_train,
            )
            logger.info("Trained experts (%s) in %.3fs", regime, time.perf_counter() - t0)

            t0 = time.perf_counter()
            R_raw = expert_reward_matrix(experts, X_bandit, y_bandit, clip_eps=settings.clip_eps)
            logger.info(
                "Built R_raw (%s) in %.3fs shape=%s", regime, time.perf_counter() - t0, R_raw.shape
            )
            R = _rescale_rewards_01(R_raw)
            if float(np.max(R_raw) - np.min(R_raw)) <= 1e-12:
                logger.warning("R_raw is (near-)constant; rescaled rewards are all zeros.")
            if write_approx_error_jsonl:
                approx_rep = linear_approx_max_error(
                    R_raw, X_bandit, lambda_reg=settings.linucb_lambda, fit_intercept=True
                )
                approx_lines.append(
                    {
                        "contamination": cfg.contamination,
                        "cluster_sep": cfg.cluster_sep,
                        "seed_idx": cfg.seed_idx,
                        "expert_regime": regime,
                        "seed_train_data": cfg.seed_train_data,
                        "seed_train_experts": cfg.seed_train_experts,
                        "mean_epsilon": approx_rep.mean_epsilon,
                        "max_epsilon": approx_rep.max_epsilon,
                        "epsilon_per_arm": approx_rep.epsilon_per_arm.astype(float).tolist(),
                        "ridge_lambda": approx_rep.lambda_reg,
                        "fit_intercept": approx_rep.fit_intercept,
                    }
                )

            t0 = time.perf_counter()
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
                seed=cfg.seed_policy,
            )
            logger.info(
                "Trained softmax router (%s) in %.3fs (epochs=%d, hidden=%d, batch=%d, lr=%g)",
                regime,
                time.perf_counter() - t0,
                settings.softmax_epochs,
                settings.softmax_hidden_dim,
                settings.softmax_batch_size,
                settings.softmax_lr,
            )
            with np.errstate(invalid="ignore"):
                logits = softmax_router.model(
                    np_to_torch(X_train_std, device=softmax_router.device)
                ).detach()
                train_labels = np.argmax(R_router_train, axis=1)
                train_acc = float((logits.argmax(dim=1).cpu().numpy() == train_labels).mean())
                train_loss = cross_entropy_from_logits(logits.cpu().numpy(), train_labels)
            logger.info(
                "Softmax in-sample metrics (%s): acc=%.4f loss=%.6f", regime, train_acc, train_loss
            )

            oracle_arm = np.argmax(R, axis=1)
            policies = {
                "uniform": UniformRandomPolicy(K=settings.K, seed=cfg.seed_policy),
                "epsilon_greedy": EpsilonGreedyPolicy(
                    K=settings.K,
                    c=settings.epsilon_greedy_c,
                    seed=cfg.seed_policy,
                ),
                "linucb_raw": LinUCBPolicy(
                    K=settings.K,
                    d=settings.d,
                    alpha=settings.linucb_alpha,
                    lambda_reg=settings.linucb_lambda,
                    forced_explore_per_arm=settings.forced_explore_per_arm,
                    seed=cfg.seed_policy,
                ),
                "softmax_router": softmax_router,
                "oracle": OraclePolicy(oracle_arm=oracle_arm),
            }

            run_key = f"{cfg.cfg_name}__{regime}"
            run_store: dict[str, Any] = {
                "expert_regime": regime,
                "contamination": cfg.contamination,
                "cluster_sep": cfg.cluster_sep,
                "seed_idx": cfg.seed_idx,
                "seed_bundle": {
                    "train_data": cfg.seed_train_data,
                    "train_experts": cfg.seed_train_experts,
                    "bandit_stream": cfg.seed_bandit_stream,
                    "policy": cfg.seed_policy,
                },
                "softmax_train": {"accuracy": train_acc, "loss": train_loss},
                "reward_gap_summary": summarize_reward_gap(R),
                "policies": {},
            }

            for policy_name, policy_obj in policies.items():
                t0 = time.perf_counter()
                result = run_bandit(policy=policy_obj, X=X_bandit_std, R=R, seed=cfg.seed_policy)
                row = _row_from_result(
                    config=cfg, expert_regime=regime, policy=policy_name, result=result
                )
                rows.append(row)
                logger.info(
                    "Policy %-14s (%s) done in %.3fs | final_cum_regret=%.4f avg_regret=%.6f best_arm_acc=%.4f mean_r=%.4f",
                    policy_name,
                    regime,
                    time.perf_counter() - t0,
                    row.final_cum_regret,
                    row.avg_regret,
                    row.best_arm_acc,
                    row.chosen_arm_mean_reward,
                )
                run_store["policies"][policy_name] = {
                    "confusion_matrix": _confusion_matrix(
                        chosen_arm=result.chosen_arm, oracle_arm=result.oracle_arm, K=settings.K
                    ).tolist(),
                    "cumulative_regret_curve": result.cumulative_regret().astype(float).tolist(),
                }

            artifacts["runs"][run_key] = run_store
        logger.info("Finished cell %s in %.3fs", cfg.cfg_name, time.perf_counter() - t_cell0)

    logger.info("Writing tables and artifacts...")
    _write_rows_csv(out / "results_rows.csv", rows)
    _write_rows_jsonl(out / "results_rows.jsonl", rows)
    _write_artifacts_json(out / "artifacts.json", artifacts)
    if write_approx_error_jsonl and approx_lines:
        approx_path = out / "approx_error_by_regime.jsonl"
        with approx_path.open("w", encoding="utf-8") as af:
            for obj in approx_lines:
                af.write(json.dumps(obj) + "\n")
        logger.info("Wrote linear approx errors: %s", approx_path.resolve())
    logger.info("Building plots...")
    _make_main_plots(
        output_dir=out,
        rows=rows,
        artifacts=artifacts,
        contaminations=contaminations,
        cluster_seps=cluster_seps,
    )
    logger.info("Done.")
    return rows, artifacts


def np_to_torch(X: np.ndarray, device: Any) -> Any:
    import torch

    return torch.as_tensor(X, dtype=torch.float32, device=device)


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits_shifted - np.log(np.exp(logits_shifted).sum(axis=1, keepdims=True))
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def summarize_reward_gap(R: np.ndarray) -> dict[str, float]:
    sorted_rewards = np.sort(R, axis=1)
    gap = sorted_rewards[:, -1] - sorted_rewards[:, -2]
    return {
        "mean": float(np.mean(gap)),
        "median": float(np.median(gap)),
        "p10": float(np.quantile(gap, 0.10)),
        "p90": float(np.quantile(gap, 0.90)),
    }


def _write_rows_csv(path: Path, rows: list[ResultRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_rows_jsonl(path: Path, rows: list[ResultRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def _write_artifacts_json(path: Path, artifacts: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(artifacts, f)


def _aggregate_metric(
    rows: list[ResultRow],
    policy: str,
    metric: str,
    contaminations: list[float],
    cluster_seps: list[float],
    expert_regime: str | None = None,
) -> np.ndarray:
    arr = np.full((len(cluster_seps), len(contaminations)), np.nan, dtype=np.float64)
    for y_idx, sep in enumerate(cluster_seps):
        for x_idx, contam in enumerate(contaminations):
            vals = [
                getattr(row, metric)
                for row in rows
                if row.policy == policy
                and (expert_regime is None or row.expert_regime == expert_regime)
                and abs(row.cluster_sep - sep) < 1e-12
                and abs(row.contamination - contam) < 1e-12
            ]
            arr[y_idx, x_idx] = float(np.mean(vals))
    return arr


def _plot_heatmap(
    data: np.ndarray,
    title: str,
    contaminations: list[float],
    cluster_seps: list[float],
    output_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(contaminations)))
    ax.set_xticklabels([f"{c:.2f}" for c in contaminations])
    ax.set_yticks(range(len(cluster_seps)))
    ax.set_yticklabels([f"{s:.1f}" for s in cluster_seps])
    ax.set_xlabel("contamination")
    ax.set_ylabel("cluster_sep")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_regret_curves(
    artifacts: dict[str, Any],
    output_path: Path,
    regimes: list[tuple[float, float]],
    expert_regime: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(regimes), figsize=(15, 4), sharey=True)
    if len(regimes) == 1:
        axes = [axes]

    for idx, (contam, sep) in enumerate(regimes):
        ax = axes[idx]
        for policy in ["linucb_raw", "softmax_router"]:
            curves = []
            for run in artifacts["runs"].values():
                if expert_regime is not None and run.get("expert_regime") != expert_regime:
                    continue
                if (
                    abs(run["contamination"] - contam) < 1e-12
                    and abs(run["cluster_sep"] - sep) < 1e-12
                ):
                    curves.append(run["policies"][policy]["cumulative_regret_curve"])
            if not curves:
                continue
            mean_curve = np.asarray(curves, dtype=np.float64).mean(axis=0)
            ax.plot(mean_curve, label=policy)
        title_extra = f" [{expert_regime}]" if expert_regime else ""
        ax.set_title(f"contam={contam:.2f}, sep={sep:.1f}{title_extra}")
        ax.set_xlabel("t")
        if idx == 0:
            ax.set_ylabel("cumulative regret")
        if ax.lines:
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _make_main_plots(
    output_dir: Path,
    rows: list[ResultRow],
    artifacts: dict[str, Any],
    contaminations: list[float],
    cluster_seps: list[float],
) -> None:
    regimes_present = sorted({r.expert_regime for r in rows})
    multi = len(regimes_present) > 1

    for regime in regimes_present:
        sfx = f"_{regime}" if multi else ""
        linucb_regret = _aggregate_metric(
            rows,
            "linucb_raw",
            "final_cum_regret",
            contaminations,
            cluster_seps,
            expert_regime=regime,
        )
        softmax_regret = _aggregate_metric(
            rows,
            "softmax_router",
            "final_cum_regret",
            contaminations,
            cluster_seps,
            expert_regime=regime,
        )
        diff = softmax_regret - linucb_regret
        linucb_acc = _aggregate_metric(
            rows, "linucb_raw", "best_arm_acc", contaminations, cluster_seps, expert_regime=regime
        )
        softmax_acc = _aggregate_metric(
            rows,
            "softmax_router",
            "best_arm_acc",
            contaminations,
            cluster_seps,
            expert_regime=regime,
        )

        title_suffix = f" ({regime})" if multi else ""
        _plot_heatmap(
            data=linucb_regret,
            title=f"LinUCB mean final cumulative regret{title_suffix}",
            contaminations=contaminations,
            cluster_seps=cluster_seps,
            output_path=output_dir / f"heatmap_linucb_regret{sfx}.png",
        )
        _plot_heatmap(
            data=softmax_regret,
            title=f"Softmax mean final cumulative regret{title_suffix}",
            contaminations=contaminations,
            cluster_seps=cluster_seps,
            output_path=output_dir / f"heatmap_softmax_regret{sfx}.png",
        )
        _plot_heatmap(
            data=diff,
            title=f"Regret difference (softmax - linucb){title_suffix}",
            contaminations=contaminations,
            cluster_seps=cluster_seps,
            output_path=output_dir / f"heatmap_regret_diff{sfx}.png",
            cmap="coolwarm",
        )
        _plot_heatmap(
            data=linucb_acc,
            title=f"LinUCB best-arm accuracy{title_suffix}",
            contaminations=contaminations,
            cluster_seps=cluster_seps,
            output_path=output_dir / f"heatmap_linucb_best_arm_acc{sfx}.png",
        )
        _plot_heatmap(
            data=softmax_acc,
            title=f"Softmax best-arm accuracy{title_suffix}",
            contaminations=contaminations,
            cluster_seps=cluster_seps,
            output_path=output_dir / f"heatmap_softmax_best_arm_acc{sfx}.png",
        )

        representative_regimes = [(0.05, 1.0), (0.20, 2.0), (0.50, 3.0)]
        curve_name = (
            "regret_curves_representative.png"
            if not multi
            else f"regret_curves_representative_{regime}.png"
        )
        _plot_regret_curves(
            artifacts=artifacts,
            output_path=output_dir / curve_name,
            regimes=representative_regimes,
            expert_regime=regime,
        )
