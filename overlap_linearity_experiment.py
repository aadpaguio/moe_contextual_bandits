from __future__ import annotations

import argparse
import logging
from pathlib import Path

from moe_bandit.experiments import OverlapLinearitySettings, run_overlap_linearity_experiment
from moe_bandit.experiments.overlap_linearity import regenerate_overlap_linearity_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run overlap-linearity experiment (x-axis = overlap_strength)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/overlap_linearity"),
        help="Directory where overlap-linearity outputs are written.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity: default WARNING, -v INFO, -vv DEBUG.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny smoke configuration for quick validation.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots only from existing artifacts/results in output-dir.",
    )
    args = parser.parse_args()

    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    settings = OverlapLinearitySettings()
    if args.smoke:
        settings = OverlapLinearitySettings(
            T=1_000,
            n_train_per_cluster=200,
            overlap_strengths=(0.25, 0.75),
            cluster_seps=(1.5,),
            seed_bundles=((101, 201, 301, 401),),
            regularization_configs=(
                OverlapLinearitySettings().regularization_configs[0],
                OverlapLinearitySettings().regularization_configs[1],
            ),
            reward_targets=("log_prob",),
            softmax_epochs=25,
        )

    if args.plots_only:
        regenerate_overlap_linearity_plots(
            output_dir=args.output_dir,
            settings=settings,
        )
        print(f"Regenerated overlap-linearity plots in: {args.output_dir.resolve()}")
        return

    run_overlap_linearity_experiment(
        output_dir=args.output_dir,
        settings=settings,
        log_level=log_level,
    )
    print(f"Saved overlap-linearity outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
