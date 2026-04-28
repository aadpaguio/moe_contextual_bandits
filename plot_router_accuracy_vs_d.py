from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PACKET_DIR = Path("outputs/report_packet/20260426_192145")


def _read_main_runs(artifacts_path: Path) -> list[dict]:
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Could not find artifacts file: {artifacts_path}")

    with artifacts_path.open(encoding="utf-8") as f:
        artifacts = json.load(f)

    return [
        run
        for run in artifacts.get("runs", {}).values()
        if run.get("block") == "main" and run.get("expert_regime") == "joint"
    ]


def _aggregate_by_d(
    runs: list[dict], section: str, metric: str
) -> tuple[list[int], list[float], list[float]]:
    by_d: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        if section not in run or metric not in run[section]:
            continue
        by_d[int(run["d"])].append(float(run[section][metric]))

    d_values = sorted(by_d)
    means = [float(np.mean(by_d[d])) for d in d_values]
    stds = [float(np.std(by_d[d])) for d in d_values]
    return d_values, means, stds


def _plot_metric(
    ax: plt.Axes,
    runs: list[dict],
    section: str,
    metric: str,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    d_values, means, stds = _aggregate_by_d(runs, section, metric)
    if not d_values:
        return

    d_arr = np.asarray(d_values, dtype=np.int64)
    means_arr = np.asarray(means, dtype=np.float64)
    stds_arr = np.asarray(stds, dtype=np.float64)
    ax.plot(d_arr, means_arr, marker="o", label=label, color=color, linestyle=linestyle)
    ax.fill_between(d_arr, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.12)


def plot_router_accuracy_vs_d(packet_dir: Path) -> Path:
    runs = _read_main_runs(packet_dir / "artifacts.json")
    if not runs:
        raise ValueError(f"No main/joint runs found in {packet_dir / 'artifacts.json'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_metric(
        ax,
        runs,
        section="softmax_best_arm",
        metric="train_best_arm_acc",
        label="Supervised softmax train best-arm acc",
        color="tab:purple",
        linestyle="-",
    )
    _plot_metric(
        ax,
        runs,
        section="softmax_best_arm",
        metric="eval_best_arm_acc",
        label="Supervised softmax eval best-arm acc",
        color="tab:purple",
        linestyle="--",
    )
    _plot_metric(
        ax,
        runs,
        section="cluster_label_router",
        metric="train_label_acc",
        label="Cluster-label train label acc",
        color="tab:brown",
        linestyle="-",
    )
    _plot_metric(
        ax,
        runs,
        section="cluster_label_router",
        metric="eval_label_acc",
        label="Cluster-label eval label acc",
        color="tab:brown",
        linestyle="--",
    )

    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("d")
    ax.set_ylabel("accuracy")
    ax.set_title("Router Train/Eval Accuracy vs Context Dimension")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = packet_dir / "plots" / "supplementary" / "router_accuracy_vs_d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot router train/eval accuracy diagnostics for one report packet."
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=DEFAULT_PACKET_DIR,
        help=f"Report packet directory. Defaults to {DEFAULT_PACKET_DIR}.",
    )
    args = parser.parse_args()
    plot_router_accuracy_vs_d(args.packet_dir)


if __name__ == "__main__":
    main()
