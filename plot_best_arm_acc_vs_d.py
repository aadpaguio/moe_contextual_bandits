from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PACKET_DIR = Path("outputs/report_packet/20260426_192145")

POLICY_ORDER = [
    "uniform",
    "epsilon_greedy",
    "online_softmax_best_arm",
    "linucb_raw",
    "softmax_best_arm",
    "cluster_label_router",
    "oracle",
]

POLICY_LABELS = {
    "uniform": "Uniform",
    "epsilon_greedy": "epsilon-greedy",
    "online_softmax_best_arm": "Online softmax",
    "linucb_raw": "LinUCB",
    "softmax_best_arm": "Supervised softmax",
    "cluster_label_router": "Cluster-label router",
    "oracle": "Oracle",
}


def _read_main_rows(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find results file: {results_path}")

    with results_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return [
        row
        for row in rows
        if row["block"] == "main"
        and row["expert_regime"] == "joint"
        and row["policy"] in set(POLICY_ORDER)
    ]


def plot_best_arm_acc_vs_d(packet_dir: Path) -> Path:
    rows = _read_main_rows(packet_dir / "results_rows.csv")
    if not rows:
        raise ValueError(f"No main/joint rows found in {packet_dir / 'results_rows.csv'}")

    d_values = sorted({int(row["d"]) for row in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in POLICY_ORDER:
        means: list[float] = []
        stds: list[float] = []
        for d in d_values:
            vals = [
                float(row["best_arm_acc"])
                for row in rows
                if row["policy"] == policy and int(row["d"]) == d
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if vals else np.nan)

        means_arr = np.asarray(means, dtype=np.float64)
        stds_arr = np.asarray(stds, dtype=np.float64)
        ax.plot(d_values, means_arr, marker="o", label=POLICY_LABELS[policy])
        ax.fill_between(d_values, means_arr - stds_arr, means_arr + stds_arr, alpha=0.12)

    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("d")
    ax.set_ylabel("best-arm accuracy")
    ax.set_title("Best-Arm Accuracy vs Context Dimension")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = packet_dir / "plots" / "main" / "best_arm_acc_vs_d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot best-arm accuracy vs d for one report packet."
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=DEFAULT_PACKET_DIR,
        help=f"Report packet directory. Defaults to {DEFAULT_PACKET_DIR}.",
    )
    args = parser.parse_args()
    plot_best_arm_acc_vs_d(args.packet_dir)


if __name__ == "__main__":
    main()
