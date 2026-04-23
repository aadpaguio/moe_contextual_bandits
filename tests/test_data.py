import numpy as np

from moe_bandit.data import generate_synthetic_data


def test_shapes_and_ranges():
    X, y, cluster_id = generate_synthetic_data(
        n_samples=500,
        K=4,
        d=6,
        cluster_sep=3.0,
        cluster_std=1.0,
        seed=7,
    )
    assert X.shape == (500, 6)
    assert y.shape == (500,)
    assert cluster_id.shape == (500,)
    assert np.all((y >= 0) & (y < 4))
    assert np.all((cluster_id >= 0) & (cluster_id < 4))
    assert np.array_equal(y, cluster_id)


def test_determinism_with_fixed_seed():
    X1, y1, c1 = generate_synthetic_data(
        n_samples=256,
        K=4,
        d=4,
        cluster_sep=2.5,
        cluster_std=0.9,
        seed=123,
    )
    X2, y2, c2 = generate_synthetic_data(
        n_samples=256,
        K=4,
        d=4,
        cluster_sep=2.5,
        cluster_std=0.9,
        seed=123,
    )
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(c1, c2)


def test_cluster_statistics_reasonable():
    K = 4
    d = 4
    cluster_sep = 3.0
    cluster_std = 1.0
    X, _, cluster_id = generate_synthetic_data(
        n_samples=20_000,
        K=K,
        d=d,
        cluster_sep=cluster_sep,
        cluster_std=cluster_std,
        seed=11,
    )

    # Means should be close to (cluster_sep * cluster_std) * e_i for d >= K.
    empirical_means = np.vstack([X[cluster_id == i].mean(axis=0) for i in range(K)])
    expected_means = np.zeros((K, d))
    expected_means[np.arange(K), np.arange(K)] = cluster_sep * cluster_std
    assert np.allclose(empirical_means, expected_means, atol=0.12)

    # Per-dimension stds in each cluster should be close to cluster_std.
    empirical_stds = np.vstack([X[cluster_id == i].std(axis=0) for i in range(K)])
    assert np.allclose(empirical_stds, cluster_std, atol=0.10)

    # Cluster frequencies should be near uniform.
    counts = np.bincount(cluster_id, minlength=K)
    probs = counts / counts.sum()
    assert np.all(np.abs(probs - 1.0 / K) < 0.03)


def test_cluster_means_are_well_separated():
    cluster_sep = 3.0
    cluster_std = 1.0
    X, _, cluster_id = generate_synthetic_data(
        n_samples=10_000,
        K=4,
        d=4,
        cluster_sep=cluster_sep,
        cluster_std=cluster_std,
        seed=0,
    )

    empirical_means = np.vstack([X[cluster_id == i].mean(axis=0) for i in range(4)])
    diffs = empirical_means[:, None, :] - empirical_means[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper_i, upper_j = np.triu_indices(4, k=1)
    min_pairwise_distance = distances[upper_i, upper_j].min()

    assert min_pairwise_distance >= 0.8 * cluster_sep * cluster_std
