from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYS = (
    "overlap_strength",
    "cluster_sep",
    "regularization_config",
)


def _read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        out[key] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare log_prob and prob linearity summaries."
    )
    parser.add_argument(
        "--log-prob-summary",
        type=Path,
        default=Path("outputs/overlap_linearity/linearity_summary.csv"),
    )
    parser.add_argument(
        "--prob-summary",
        type=Path,
        default=Path("outputs/overlap_linearity_prob/linearity_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reward_target_comparison.csv"),
    )
    args = parser.parse_args()

    log_rows = _read_rows(args.log_prob_summary)
    prob_rows = _read_rows(args.prob_summary)

    fieldnames = [
        *KEYS,
        "linucb_regret_log_prob",
        "linucb_regret_prob",
        "linucb_regret_delta_prob_minus_log_prob",
        "linucb_normalized_regret_log_prob",
        "linucb_normalized_regret_prob",
        "linucb_normalized_regret_delta_prob_minus_log_prob",
        "relative_epsilon_log_prob",
        "relative_epsilon_prob",
        "relative_epsilon_delta_prob_minus_log_prob",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(set(log_rows) & set(prob_rows)):
            log_row = log_rows[key]
            prob_row = prob_rows[key]
            writer.writerow(
                {
                    "overlap_strength": key[0],
                    "cluster_sep": key[1],
                    "regularization_config": key[2],
                    "linucb_regret_log_prob": log_row["linucb_regret"],
                    "linucb_regret_prob": prob_row["linucb_regret"],
                    "linucb_regret_delta_prob_minus_log_prob": float(prob_row["linucb_regret"])
                    - float(log_row["linucb_regret"]),
                    "linucb_normalized_regret_log_prob": log_row.get(
                        "linucb_normalized_regret", "nan"
                    ),
                    "linucb_normalized_regret_prob": prob_row.get(
                        "linucb_normalized_regret", "nan"
                    ),
                    "linucb_normalized_regret_delta_prob_minus_log_prob": float(
                        prob_row.get("linucb_normalized_regret", "nan")
                    )
                    - float(log_row.get("linucb_normalized_regret", "nan")),
                    "relative_epsilon_log_prob": log_row.get("relative_epsilon", "nan"),
                    "relative_epsilon_prob": prob_row.get("relative_epsilon", "nan"),
                    "relative_epsilon_delta_prob_minus_log_prob": float(
                        prob_row.get("relative_epsilon", "nan")
                    )
                    - float(log_row.get("relative_epsilon", "nan")),
                }
            )
    print(f"Wrote reward-target comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
