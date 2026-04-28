from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_PACKET_DIR = Path("outputs/report_packet/20260426_192145")


def generate_tables(packet_dir: Path) -> tuple[Path, Path]:
    rows = list(csv.DictReader((packet_dir / "results_rows.csv").open()))
    rows = [r for r in rows if r["block"] == "main" and r["expert_regime"] == "joint"]

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], int(row["d"]))].append(float(row["best_arm_acc"]))

    summary: list[tuple[str, int, float, float, int]] = []
    for (policy, d), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(vals, dtype=float)
        summary.append((policy, d, float(arr.mean()), float(arr.std()), int(arr.size)))

    out_dir = packet_dir / "diagnostics" / "best_arm_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "best_arm_acc_table.tex"
    full_lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Best-arm accuracy by policy and dimension (mean $\pm$ std across seeds).}",
        r"\label{tab:best-arm-acc-summary}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Policy & $d$ & Mean acc. & Std acc. \\",
        r"\midrule",
    ]
    for policy, d, mean_acc, std_acc, _n in summary:
        full_lines.append(f"{policy.replace('_', r'\_')} & {d} & {mean_acc:.4f} & {std_acc:.4f} \\\\")
    full_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    full_path.write_text("\n".join(full_lines) + "\n", encoding="utf-8")

    keep = [
        "cluster_label_router",
        "linucb_raw",
        "online_softmax_best_arm",
        "softmax_best_arm",
        "uniform",
    ]
    display_name = {
        "cluster_label_router": "Cluster-label router",
        "linucb_raw": "LinUCB",
        "online_softmax_best_arm": "Online softmax",
        "softmax_best_arm": "Offline softmax",
        "uniform": "Uniform",
    }

    by_policy = defaultdict(list)
    for policy, _d, mean_acc, _std_acc, _n in summary:
        if policy in keep:
            by_policy[policy].append(mean_acc)

    condensed_path = out_dir / "best_arm_acc_table_condensed.tex"
    condensed_lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Condensed best-arm accuracy across dimensions $d\in\{2,4,8,16,32,64\}$.}",
        r"\label{tab:best-arm-acc-condensed}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Mean acc. over $d$ & Range over $d$ \\",
        r"\midrule",
    ]
    for policy in keep:
        arr = np.asarray(by_policy[policy], dtype=float)
        condensed_lines.append(
            f"{display_name[policy]} & {arr.mean():.4f} & [{arr.min():.4f}, {arr.max():.4f}] \\\\"
        )
    condensed_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    condensed_path.write_text("\n".join(condensed_lines) + "\n", encoding="utf-8")

    return full_path, condensed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX best-arm accuracy tables.")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=DEFAULT_PACKET_DIR,
        help=f"Report packet directory. Defaults to {DEFAULT_PACKET_DIR}.",
    )
    args = parser.parse_args()
    full_path, condensed_path = generate_tables(args.packet_dir)
    print(full_path)
    print(condensed_path)


if __name__ == "__main__":
    main()
