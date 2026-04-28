from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moe_bandit.experiments import run_main_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run overlap-regime contextual-bandit experiment grid."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/overlap_grid"),
        help="Directory where overlap experiment outputs are written.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity: default WARNING, -v INFO, -vv DEBUG.",
    )
    parser.add_argument(
        "--approx-error",
        action="store_true",
        help="Write ridge epsilon diagnostic to approx_error_by_regime.jsonl.",
    )
    args = parser.parse_args()

    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    run_main_grid(
        output_dir=args.output_dir,
        expert_training_regimes=("overlap",),
        write_approx_error_jsonl=args.approx_error,
        log_level=log_level,
    )
    print(f"Saved overlap grid outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
