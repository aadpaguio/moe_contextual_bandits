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


def test_cluster_geometry_is_stable_across_sample_seeds():
    K = 4
    d = 4
    cluster_sep = 2.5
    cluster_std = 1.0
    X1, _, c1 = generate_synthetic_data(
        n_samples=30_000,
        K=K,
        d=d,
        cluster_sep=cluster_sep,
        cluster_std=cluster_std,
        seed=123,
    )
    X2, _, c2 = generate_synthetic_data(
        n_samples=30_000,
        K=K,
        d=d,
        cluster_sep=cluster_sep,
        cluster_std=cluster_std,
        seed=456,
    )

    means1 = np.vstack([X1[c1 == i].mean(axis=0) for i in range(K)])
    means2 = np.vstack([X2[c2 == i].mean(axis=0) for i in range(K)])

    assert np.allclose(means1, means2, atol=0.07)


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

    # Means should lie on orthogonal directions with the requested radius.
    empirical_means = np.vstack([X[cluster_id == i].mean(axis=0) for i in range(K)])
    expected_radius = cluster_sep * cluster_std
    assert np.allclose(np.linalg.norm(empirical_means, axis=1), expected_radius, atol=0.12)

    gram = empirical_means @ empirical_means.T
    off_diag = gram[~np.eye(K, dtype=bool)]
    assert np.allclose(off_diag, 0.0, atol=0.20)

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
    pairwise_distances = distances[upper_i, upper_j]

    expected_distance = np.sqrt(2.0) * cluster_sep * cluster_std
    assert np.allclose(pairwise_distances, expected_distance, atol=0.18)


def test_cluster_means_fallback_when_k_exceeds_d():
    cluster_sep = 3.0
    cluster_std = 1.0
    X, _, cluster_id = generate_synthetic_data(
        n_samples=20_000,
        K=4,
        d=2,
        cluster_sep=cluster_sep,
        cluster_std=cluster_std,
        seed=5,
    )

    empirical_means = np.vstack([X[cluster_id == i].mean(axis=0) for i in range(4)])
    expected_radius = cluster_sep * cluster_std

    assert np.allclose(np.linalg.norm(empirical_means, axis=1), expected_radius, atol=0.15)

    diffs = empirical_means[:, None, :] - empirical_means[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    upper_i, upper_j = np.triu_indices(4, k=1)
    assert distances[upper_i, upper_j].min() > 0.2 * cluster_sep * cluster_std
