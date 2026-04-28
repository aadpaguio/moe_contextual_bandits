from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PACKET_DIR = Path("outputs/report_packet/20260426_192145")


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
        and row["policy"] in {"linucb_raw", "online_softmax_best_arm"}
    ]


def plot_linucb_minus_online_softmax(packet_dir: Path) -> Path:
    rows = _read_main_rows(packet_dir / "results_rows.csv")
    regrets: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)

    for row in rows:
        d = int(row["d"])
        seed_idx = int(row["seed_idx"])
        regrets[(d, seed_idx)][row["policy"]] = float(row["final_cum_regret"])

    d_values = sorted({d for d, _ in regrets})
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []

    for d in d_values:
        gaps = []
        for (row_d, _seed_idx), by_policy in regrets.items():
            if row_d != d:
                continue
            if "linucb_raw" in by_policy and "online_softmax_best_arm" in by_policy:
                gaps.append(by_policy["linucb_raw"] - by_policy["online_softmax_best_arm"])

        if not gaps:
            means.append(np.nan)
            stds.append(np.nan)
            counts.append(0)
            continue

        gaps_arr = np.asarray(gaps, dtype=np.float64)
        means.append(float(np.mean(gaps_arr)))
        stds.append(float(np.std(gaps_arr)))
        counts.append(len(gaps))

    d_arr = np.asarray(d_values, dtype=np.int64)
    means_arr = np.asarray(means, dtype=np.float64)
    stds_arr = np.asarray(stds, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(d_arr, means_arr, marker="o", color="tab:blue", label="mean gap")
    ax.fill_between(
        d_arr,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color="tab:blue",
        alpha=0.15,
        label="std across seeds",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d")
    ax.set_ylabel("LinUCB regret - online softmax regret")
    ax.set_title("LinUCB vs online softmax gap")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = packet_dir / "plots" / "main" / "linucb_minus_online_softmax_vs_d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {output_path}")
    for d, mean, std, n in zip(d_values, means, stds, counts, strict=True):
        print(f"d={d}: mean_gap={mean:.3f}, std={std:.3f}, n={n}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot LinUCB minus online softmax regret for one report packet."
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=DEFAULT_PACKET_DIR,
        help=f"Report packet directory. Defaults to {DEFAULT_PACKET_DIR}.",
    )
    args = parser.parse_args()
    plot_linucb_minus_online_softmax(args.packet_dir)


if __name__ == "__main__":
    main()
