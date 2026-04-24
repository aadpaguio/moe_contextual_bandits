from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moe_bandit.experiments import run_joint_d_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run joint-only d sweep for contextual bandits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/joint_d_sweep_phase2"),
        help="Directory where result tables, artifacts, and plots are written.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity: default WARNING, -v INFO, -vv DEBUG.",
    )
    args = parser.parse_args()
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run_joint_d_sweep(output_dir=args.output_dir)
    print(f"Saved joint d-sweep outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
