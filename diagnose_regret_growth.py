from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PACKET_DIR = Path("outputs/report_packet/20260426_192145")
RUN_DIR_RE = re.compile(r"^main_joint_d=(?P<d>\d+)_seed=(?P<seed>\d+)$")


def estimate_regret_exponent(cum_regret: np.ndarray, t_min: int) -> tuple[float, float]:
    """
    Estimate beta in R(T) ~= c * T^beta using log-log regression.
    Returns (beta, r2).
    """
    t = np.arange(1, len(cum_regret) + 1, dtype=np.float64)
    mask = (t >= float(t_min)) & (cum_regret > 0.0) & np.isfinite(cum_regret)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("nan")

    x = np.log(t[mask])
    y = np.log(cum_regret[mask])
    beta, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + beta * x

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return float(beta), float(r2)


def _iter_main_runs(raw_dir: Path) -> list[tuple[int, int, Path]]:
    runs: list[tuple[int, int, Path]] = []
    if not raw_dir.exists():
        raise FileNotFoundError(f"Could not find raw run directory: {raw_dir}")

    for run_dir in sorted(raw_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_RE.match(run_dir.name)
        if not match:
            continue
        d = int(match.group("d"))
        seed_idx = int(match.group("seed"))
        npz_path = run_dir / "seed_data.npz"
        if npz_path.exists():
            runs.append((d, seed_idx, npz_path))
    return runs


def _fmt_float(value: float, places: int = 4) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{places}f}"


def run_diagnostic(packet_dir: Path, t_min: int, include_oracle: bool) -> tuple[Path, Path]:
    raw_dir = packet_dir / "raw"
    runs = _iter_main_runs(raw_dir)
    if not runs:
        raise RuntimeError(f"No main joint runs found in: {raw_dir}")

    beta_records: list[dict[str, float | int | str]] = []
    avg_regret_by_policy_d: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for d, seed_idx, npz_path in runs:
        arrays = np.load(npz_path)
        for key in arrays.files:
            if not key.endswith("__cumulative_regret"):
                continue
            policy = key[: -len("__cumulative_regret")]
            if not include_oracle and policy == "oracle":
                continue

            cum_regret = np.asarray(arrays[key], dtype=np.float64)
            if cum_regret.ndim != 1 or len(cum_regret) == 0:
                continue

            beta, r2 = estimate_regret_exponent(cum_regret=cum_regret, t_min=t_min)
            beta_records.append(
                {
                    "policy": policy,
                    "d": d,
                    "seed_idx": seed_idx,
                    "beta": beta,
                    "r2": r2,
                }
            )

            t = np.arange(1, len(cum_regret) + 1, dtype=np.float64)
            avg_regret_by_policy_d[(policy, d)].append(cum_regret / t)

    if not beta_records:
        raise RuntimeError("No cumulative regret arrays found in selected runs.")

    out_dir = packet_dir / "diagnostics" / "regret_growth"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = out_dir / "beta_per_seed.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "d", "seed_idx", "beta", "r2"])
        writer.writeheader()
        writer.writerows(beta_records)

    grouped_beta: dict[tuple[str, int], list[float]] = defaultdict(list)
    for rec in beta_records:
        grouped_beta[(str(rec["policy"]), int(rec["d"]))].append(float(rec["beta"]))

    summary_rows: list[dict[str, str | int | float]] = []
    for (policy, d), beta_values in sorted(grouped_beta.items(), key=lambda x: (x[0][0], x[0][1])):
        beta_arr = np.asarray(beta_values, dtype=np.float64)
        finite = beta_arr[np.isfinite(beta_arr)]
        if finite.size == 0:
            mean_beta = float("nan")
            std_beta = float("nan")
        else:
            mean_beta = float(np.mean(finite))
            std_beta = float(np.std(finite))
        summary_rows.append(
            {
                "policy": policy,
                "d": d,
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "n_seeds": int(finite.size),
            }
        )

    summary_csv = out_dir / "beta_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "d", "mean_beta", "std_beta", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (policy, d), curves in sorted(avg_regret_by_policy_d.items(), key=lambda x: (x[0][0], x[0][1])):
        lengths = {len(c) for c in curves}
        if len(lengths) != 1:
            min_len = min(lengths)
            stack = np.vstack([c[:min_len] for c in curves])
            t = np.arange(1, min_len + 1, dtype=np.int64)
        else:
            stack = np.vstack(curves)
            t = np.arange(1, len(curves[0]) + 1, dtype=np.int64)

        mean_curve = np.mean(stack, axis=0)
        std_curve = np.std(stack, axis=0)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(t, mean_curve, color="tab:blue", lw=1.8, label="mean R(T)/T")
        ax.fill_between(
            t,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="tab:blue",
            alpha=0.15,
            label="std across seeds",
        )
        ax.set_xlabel("T")
        ax.set_ylabel("Average regret so far, R(T)/T")
        ax.set_title(f"{policy} | d={d} | average regret trajectory")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.legend(fontsize=8)
        fig.tight_layout()

        out_png = plot_dir / f"avg_regret_{policy}_d={d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    print(f"Wrote per-seed beta table: {per_seed_csv}")
    print(f"Wrote summary beta table:  {summary_csv}")
    print("")
    print("policy | d | mean_beta | std_beta | n_seeds")
    for row in summary_rows:
        print(
            f"{row['policy']} | {row['d']} | {_fmt_float(float(row['mean_beta']))} | "
            f"{_fmt_float(float(row['std_beta']))} | {row['n_seeds']}"
        )
    print("")
    print(f"Wrote average-regret plots to: {plot_dir}")

    return per_seed_csv, summary_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate regret-growth exponent beta from cumulative-regret curves and "
            "plot average regret R(T)/T."
        )
    )
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=DEFAULT_PACKET_DIR,
        help=f"Report packet directory. Defaults to {DEFAULT_PACKET_DIR}.",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        default=100,
        help="Minimum T used in the log-log fit. Default: 100.",
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Include oracle policy in diagnostics (off by default).",
    )
    args = parser.parse_args()

    run_diagnostic(packet_dir=args.packet_dir, t_min=args.t_min, include_oracle=args.include_oracle)


if __name__ == "__main__":
    main()
