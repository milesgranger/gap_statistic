# -*- coding: utf-8 -*-
import pytest


@pytest.mark.parametrize(
    "parallel_backend, n_jobs, n_clusters", [
        pytest.param('joblib', -1, 3, id="parallel_backend='joblib', n_jobs=-1, n_clusters=2"),
        pytest.param('multiprocessing', -1, 3, id="parallel_backend='multiprocessing', n_jobs=-1, n_clusters=2"),
        pytest.param(None, -1, 3, id="parallel_backend=None, n_jobs=-1, n_clusters=2"),
        pytest.param('rust', -1, 3, id="parallel_backend='rust', n_jobs=-1, n_clusters=2")
    ]
)
def test_optimalk(parallel_backend, n_jobs, n_clusters):
    """
    Test core functionality of OptimalK using all backends.
    """
    import numpy as np
    from sklearn.datasets.samples_generator import make_blobs
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=parallel_backend, n_jobs=n_jobs)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    suggested_clusters = optimalK(X, n_refs=3, cluster_array=np.arange(1, 10))

    assert np.allclose(suggested_clusters, n_clusters, 2), ('Correct clusters is {}, OptimalK suggested {}'
                                                            .format(n_clusters, suggested_clusters))


def test_optimalk_cluster_array_vs_data_sizes_error():
    """
    Test ValueError when cluster_array is larger than dataset.
    """
    import numpy as np
    from sklearn.datasets.samples_generator import make_blobs
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=None, n_jobs=-1)

    # Create data
    X, y = make_blobs(n_samples=5, n_features=2, centers=3)

    with pytest.raises(ValueError) as excinfo:
        optimalK(X, cluster_array=np.arange(1, 10))
    assert 'The number of suggested clusters to try' in str(excinfo.value)


def test_optimalk_cluster_array_values_error():
    """
    Test ValueError when cluster_array contains values less than 1
    """
    from sklearn.datasets.samples_generator import make_blobs
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=None, n_jobs=-1)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    with pytest.raises(ValueError) as excinfo:
        optimalK(X, cluster_array=[0, -1, 1, 2, 3])
    assert 'cluster_array contains values less than 1' in str(excinfo.value)


def test_optimalk_cluster_array_empty_error():
    """
    Test ValueError when cluster_array is empty.
    """
    from sklearn.datasets.samples_generator import make_blobs
    from gap_statistic import OptimalK

    # Create optimalK instance
    optimalK = OptimalK(parallel_backend=None, n_jobs=-1)

    # Create data
    X, y = make_blobs(n_samples=int(1e3), n_features=2, centers=3)

    with pytest.raises(ValueError) as excinfo:
        optimalK(X, cluster_array=[])
    assert 'The supplied cluster_array has no values.' in str(excinfo.value)


def test_dunders():
    """
    Test that implemented dunder methods don't return errors
    """
    from gap_statistic import OptimalK
    optimalK = OptimalK()
    optimalK.__str__()
    optimalK.__repr__()
    optimalK._repr_html_()
