from __future__ import annotations

import argparse
from pathlib import Path

from moe_bandit.experiments import LinearEnvSettings, run_linear_env_sanity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run linearizable environment sanity check.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/linear_env_sanity"),
    )
    parser.add_argument("--T", type=int, default=10_000)
    args = parser.parse_args()

    settings = LinearEnvSettings(T=args.T)
    rows = run_linear_env_sanity(output_dir=args.output_dir, settings=settings)
    print(f"Saved linear-env sanity outputs to: {args.output_dir.resolve()}")
    for row in rows:
        print(
            f"{row.policy:14s} final_cum_regret={row.final_cum_regret:.4f} "
            f"avg_regret={row.avg_regret:.6f} best_arm_acc={row.best_arm_acc:.4f}"
        )


if __name__ == "__main__":
    main()
