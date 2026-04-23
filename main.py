from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moe_bandit.experiments import run_main_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contextual-bandit experiment grid.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/main_grid"),
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
    output_dir: Path = args.output_dir
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    run_main_grid(output_dir=output_dir, log_level=log_level)
    print(f"Saved main grid outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
