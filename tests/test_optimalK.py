# -*- coding: utf-8 -*-
import os
import pytest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MeanShift

from gap_statistic import OptimalK


def test_bad_init_config():
    """
    Cannot define own clustering function and try to use Rust backend
    """
    with pytest.raises(ValueError):
        OptimalK(parallel_backend="rust", clusterer=lambda x, k: print("just testing"))


@pytest.mark.parametrize("ClusterModel", [KMeans, MeanShift])
def test_alternative_clusting_method(ClusterModel):
    """
    Test that users can supply alternative clustering method as dep injection
    """

    def clusterer(X: np.ndarray, k: int, another_test_arg):
        """
        Function to wrap a sklearn model as a clusterer for OptimalK
        First two arguments are always the data matrix, and k, and can supply
        """
        m = ClusterModel()
        m.fit(X)
        assert another_test_arg == "test"
        return m.cluster_centers_, m.predict(X)

    optimalk = OptimalK(
        n_jobs=-1,
        parallel_backend="joblib",
        clusterer=clusterer,
        clusterer_kwargs={"another_test_arg": "test"},
    )
    X, y = make_blobs(n_samples=50, n_features=2, centers=3)
    n_clusters = optimalk(X, n_refs=3, cluster_array=np.arange(1, 5))
    assert isinstance(n_clusters, int)


@pytest.mark.parametrize(
    "parallel_backend, n_jobs, n_clusters",
    [
        pytest.param(
            "joblib", 1, 3, id="parallel_backend='joblib', n_jobs=1, n_clusters=3"
        ),
        pytest.param(None, 1, 3, id="parallel_backend=None, n_jobs=1, n_clusters=3"),
        # TODO: Add back this test param in rust side extension
        # pytest.param(
        #    "rust", 1, 3, id="parallel_backend='rust', n_jobs=1, n_clusters=3"
        # ),
    ],
)
def test_optimalk(parallel_backend, n_jobs, n_clusters):
    """
    Test core functionality of OptimalK using all backends.
    """

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=parallel_backend, n_jobs=n_jobs)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    suggested_clusters = optimalK(X, n_refs=3, cluster_array=np.arange(1, 10))

    assert np.allclose(
        suggested_clusters, n_clusters, 2
    ), "Correct clusters is {}, OptimalK suggested {}".format(
        n_clusters, suggested_clusters
    )


@pytest.mark.skipif(
    "TEST_RUST_EXT" not in os.environ, reason="Rust extension not built."
)
def test_optimalk_rust_ext():
    """
    Test core functionality of OptimalK using all backends.
    """

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend="rust", n_jobs=1)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    suggested_clusters = optimalK(X, n_refs=3, cluster_array=np.arange(1, 10))

    assert np.allclose(
        suggested_clusters, 3, 2
    ), "Correct clusters is {}, OptimalK suggested {}".format(3, suggested_clusters)


def test_optimalk_cluster_array_vs_data_sizes_error():
    """
    Test ValueError when cluster_array is larger than dataset.
    """
    import numpy as np
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=None, n_jobs=-1)

    # Create data
    X, y = make_blobs(n_samples=5, n_features=2, centers=3)

    with pytest.raises(ValueError) as excinfo:
        optimalK(X, cluster_array=np.arange(1, 10))
    assert "The number of suggested clusters to try" in str(excinfo.value)


def test_optimalk_cluster_array_values_error():
    """
    Test ValueError when cluster_array contains values less than 1
    """
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=None, n_jobs=-1)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    with pytest.raises(ValueError) as excinfo:
        optimalK(X, cluster_array=[0, -1, 1, 2, 3])
    assert "cluster_array contains values less than 1" in str(excinfo.value)


def test_optimalk_cluster_array_empty_error():
    """
    Test ValueError when cluster_array is empty.
    """
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=None, n_jobs=-1)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    with pytest.raises(ValueError) as excinfo:
        optimalK(X, cluster_array=[])
    assert "The supplied cluster_array has no values." in str(excinfo.value)


def test_dunders():
    """
    Test that implemented dunder methods don't return errors
    """
    from gap_statistic import OptimalK

    optimalK = OptimalK()
    optimalK.__str__()
    optimalK.__repr__()
    optimalK._repr_html_()
