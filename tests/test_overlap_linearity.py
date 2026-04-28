from __future__ import annotations

import csv

from moe_bandit.experiments.overlap_linearity import (
    OverlapLinearitySettings,
    RegularizationConfig,
    run_overlap_linearity_experiment,
)


def test_overlap_linearity_smoke_outputs(tmp_path):
    settings = OverlapLinearitySettings(
        K=3,
        d=4,
        T=120,
        n_train_per_cluster=80,
        overlap_strengths=(0.25, 0.75),
        cluster_seps=(1.5,),
        seed_bundles=((11, 21, 31, 41),),
        regularization_configs=(
            RegularizationConfig("baseline", 0.0, 0.0, 1.0),
            RegularizationConfig("regularized", 1e-4, 0.05, 1.0),
        ),
        reward_targets=("log_prob",),
        softmax_hidden_dim=16,
        softmax_epochs=5,
        softmax_batch_size=32,
    )
    rows, _artifacts = run_overlap_linearity_experiment(
        output_dir=tmp_path,
        settings=settings,
    )
    # 2 overlap strengths * 1 sep * 1 seed * 2 reg configs * 1 reward target * 7 policies
    assert len(rows) == 28
    assert (tmp_path / "results_rows.csv").exists()
    assert (tmp_path / "results_rows.jsonl").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert (tmp_path / "linearity_summary.csv").exists()
    assert (tmp_path / "scatter_linucb_regret_vs_epsilon.png").exists()
    assert (tmp_path / "heatmap_linucb_rbf_regret_baseline_log_prob.png").exists()
    assert (tmp_path / "heatmap_linucb_normalized_regret_baseline_log_prob.png").exists()

    with (tmp_path / "linearity_summary.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        assert cols is not None
        for key in (
            "overlap_strength",
            "cluster_sep",
            "regularization_config",
            "reward_target",
            "mean_epsilon",
            "max_epsilon",
            "mean_rmse_epsilon",
            "max_rmse_epsilon",
            "mean_p95_abs_epsilon",
            "max_p95_abs_epsilon",
            "relative_epsilon",
            "linucb_regret",
            "linucb_rbf_regret",
            "linucb_router_feat_regret",
            "linucb_normalized_regret",
            "softmax_regret",
            "oracle_gap_mean",
            "cumulative_oracle_gap",
        ):
            assert key in cols
