# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Union, Iterable, Callable, Generator
from scipy.cluster.vq import kmeans2

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_FOUND = True
except ImportError:
    MATPLOTLIB_FOUND = False
    warnings.warn("matplotlib not installed; results plotting is disabled.")
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel, delayed = None, None
    warnings.warn(
        "joblib not installed, will be unavailable as a backend for parallel processing."
    )


GapCalcResult = namedtuple(
    "GapCalcResult", "gap_value n_clusters ref_dispersion_std sdk sk gap_star sk_star"
)


class OptimalK:
    """
    Obtain the optimal number of clusters a dataset should have using the gap statistic.
        Tibshirani, Walther, Hastie
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf

    Example:
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> from gap_statistic import OptimalK
    >>> X, y = make_blobs(n_samples=int(1e5), n_features=2, centers=3, random_state=100)
    >>> optimalK = OptimalK(parallel_backend='joblib')
    >>> optimalK(X, cluster_array=[1,2,3,4,5])
    3
    """

    gap_df = None

    def __init__(
        self,
        n_jobs: int = -1,
        parallel_backend: str = "joblib",
        clusterer: Callable = None,
        clusterer_kwargs: dict = None,
    ) -> None:
        """
        Construct OptimalK to use n_jobs (multiprocessing using joblib, multiprocessing, or single core.
        if parallel_backend == 'rust' it will use all cores.

        :param n_jobs:
        :param parallel_backend:
        :param clusterer:
        :param clusterer_kwargs:
        """
        if clusterer is not None and parallel_backend == "rust":
            raise ValueError(
                "Cannot use 'rust' backend with a user defined clustering function, only KMeans"
                " is supported on the rust implementation"
            )
        self.parallel_backend = (
            parallel_backend
            if parallel_backend in ["joblib", "multiprocessing", "rust"]
            else None
        )
        self.n_jobs = n_jobs if 1 <= n_jobs <= cpu_count() else cpu_count()  # type: int
        self.n_jobs = 1 if parallel_backend is None else self.n_jobs
        self.clusterer = clusterer if clusterer is not None else kmeans2
        self.clusterer_kwargs = (
            clusterer_kwargs or dict()
            if clusterer is not None
            else dict(iter=10, minit="points")
        )

    def __call__(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_refs: int = 3,
        cluster_array: Iterable[int] = (),
    ):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
        :param X - pandas dataframe or numpy array of data points of shape (n_samples, n_features)
        :param n_refs - int: Number of random reference data sets used as inertia reference to actual data.
        :param cluster_array - 1d iterable of integers; each representing n_clusters to try on the data.
        """

        # Convert the 1d array of n_clusters to try into an array
        # Raise error if values are less than 1 or larger than the unique sample in the set.
        cluster_array = np.array([x for x in cluster_array]).astype(int)
        if np.where(cluster_array < 1)[0].shape[0]:
            raise ValueError(
                "cluster_array contains values less than 1: {}".format(
                    cluster_array[np.where(cluster_array < 1)[0]]
                )
            )
        if cluster_array.shape[0] > X.shape[0]:
            raise ValueError(
                "The number of suggested clusters to try ({}) is larger than samples in dataset. ({})".format(
                    cluster_array.shape[0], X.shape[0]
                )
            )
        if not cluster_array.shape[0]:
            raise ValueError("The supplied cluster_array has no values.")

        # Array of resulting gaps.
        gap_df = pd.DataFrame({"n_clusters": [], "gap_value": []})

        # Define the compute engine; all methods take identical args and are generators.
        if self.parallel_backend == "joblib":
            engine = self._process_with_joblib
        elif self.parallel_backend == "multiprocessing":
            engine = self._process_with_multiprocessing
        elif self.parallel_backend == "rust":
            engine = self._process_with_rust
        else:
            engine = self._process_non_parallel

        # Calculate the gaps for each cluster count.
        for gap_calc_result in engine(X, n_refs, cluster_array):

            # Assign this loop's gap statistic to gaps
            gap_df = gap_df.append(
                {
                    "n_clusters": gap_calc_result.n_clusters,
                    "gap_value": gap_calc_result.gap_value,
                    "ref_dispersion_std": gap_calc_result.ref_dispersion_std,
                    "sdk": gap_calc_result.sdk,
                    "sk": gap_calc_result.sk,
                    "gap*": gap_calc_result.gap_star,
                    "sk*": gap_calc_result.sk_star,
                },
                ignore_index=True,
            )
            gap_df["gap_k+1"] = gap_df["gap_value"].shift(-1)
            gap_df["gap*_k+1"] = gap_df["gap*"].shift(-1)
            gap_df["sk+1"] = gap_df["sk"].shift(-1)
            gap_df["sk*+1"] = gap_df["sk*"].shift(-1)
            gap_df["diff"] = gap_df["gap_value"] - gap_df["gap_k+1"] + gap_df["sk+1"]
            gap_df["diff*"] = gap_df["gap*"] - gap_df["gap*_k+1"] + gap_df["sk*+1"]

        # drop auxilariy columns
        gap_df.drop(
            labels=["sdk", "gap_k+1", "gap*_k+1", "sk+1", "sk*+1"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        self.gap_df = gap_df.sort_values(by="n_clusters", ascending=True).reset_index(
            drop=True
        )
        self.n_clusters = int(
            self.gap_df.loc[np.argmax(self.gap_df.gap_value.values)].n_clusters
        )
        return self.n_clusters

    def plot_results(self):
        """
        Plots the results of the last run optimal K search procedure.
        Four plots are printed:
        (1) A plot of the Gap value - as defined in the original Tibshirani et
        al paper - versus n, the number of clusters.
        (2) A plot of diff versus n, the number of clusters, where diff =
        Gap(k) - Gap(k+1) + s_{k+1}. The original Tibshirani et al paper
        recommends choosing the smallest k such that this measure is positive.
        (3) A plot of the Gap* value - a variant of the Gap statistic suggested
        in "A comparison of Gap statistic definitions with and with-out
        logarithm function" [https://core.ac.uk/download/pdf/12172514.pdf],
        which simply removes the logarithm operation from the Gap calculation -
        versus n, the number of clusters.
        (4) A plot of the diff* value versus n, the number of clusters. diff*
        corresponds to the aforementioned diff value for the case of Gap*.
        """
        if not MATPLOTLIB_FOUND:
            print("matplotlib not installed; results plotting is disabled.")
            return
        if not hasattr(self, "gap_df") or self.gap_df is None:
            print("No results to print. OptimalK not called yet.")
            return

        # Gap values plot
        plt.plot(self.gap_df.n_clusters, self.gap_df.gap_value, linewidth=3)
        plt.scatter(
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].n_clusters,
            self.gap_df[self.gap_df.n_clusters == self.n_clusters].gap_value,
            s=250,
            c="r",
        )
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Gap Value")
        plt.title("Gap Values by Cluster Count")
        plt.show()

        # diff plot
        plt.plot(self.gap_df.n_clusters, self.gap_df["diff"], linewidth=3)
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Diff Value")
        plt.title("Diff Values by Cluster Count")
        plt.show()

        # Gap* plot
        max_ix = self.gap_df[self.gap_df["gap*"] == self.gap_df["gap*"].max()].index[0]
        plt.plot(self.gap_df.n_clusters, self.gap_df["gap*"], linewidth=3)
        plt.scatter(
            self.gap_df.loc[max_ix]["n_clusters"],
            self.gap_df.loc[max_ix]["gap*"],
            s=250,
            c="r",
        )
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Gap* Value")
        plt.title("Gap* Values by Cluster Count")
        plt.show()

        # diff* plot
        plt.plot(self.gap_df.n_clusters, self.gap_df["diff*"], linewidth=3)
        plt.grid(True)
        plt.xlabel("Cluster Count")
        plt.ylabel("Diff* Value")
        plt.title("Diff* Values by Cluster Count")
        plt.show()

    @staticmethod
    def _calculate_dispersion(
        X: Union[pd.DataFrame, np.ndarray], labels: np.ndarray, centroids: np.ndarray
    ) -> float:
        """
        Calculate the dispersion between actual points and their assigned centroids
        """
        disp = np.sum(
            np.sum(
                [np.abs(inst - centroids[label]) ** 2 for inst, label in zip(X, labels)]
            )
        )
        return disp

    def _calculate_gap(
        self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, n_clusters: int
    ) -> GapCalcResult:
        """
        Calculate the gap value of the given data, n_refs, and number of clusters.
        Return the resutling gap value and n_clusters
        """
        # Holder for reference dispersion results
        ref_dispersions = np.zeros(n_refs)

        # Compute the range of each feature
        X = np.asarray(X)
        a, b = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)

        # For n_references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(n_refs):

            # Create new random reference set uniformly over the range of each feature
            random_data = np.random.random_sample(size=X.shape) * (b - a) + a

            # Fit to it, getting the centroids and labels, and add to accumulated reference dispersions array.
            centroids, labels = self.clusterer(
                random_data, n_clusters, **self.clusterer_kwargs
            )  # type: Tuple[np.ndarray, np.ndarray]
            dispersion = self._calculate_dispersion(
                X=random_data, labels=labels, centroids=centroids
            )
            ref_dispersions[i] = dispersion

        # Fit cluster to original data and create dispersion calc.
        centroids, labels = self.clusterer(
            X, n_clusters, **self.clusterer_kwargs
        )  # type: Tuple[np.ndarray, np.ndarray]
        dispersion = self._calculate_dispersion(X=X, labels=labels, centroids=centroids)

        # Calculate gap statistic
        ref_log_dispersion = np.mean(np.log(ref_dispersions))
        log_dispersion = np.log(dispersion)
        gap_value = ref_log_dispersion - log_dispersion
        # compute standard deviation
        sdk = np.sqrt(np.mean((np.log(ref_dispersions) - ref_log_dispersion) ** 2.0))
        sk = np.sqrt(1.0 + 1.0 / n_refs) * sdk

        # Calculate Gap* statistic
        # by "A comparison of Gap statistic definitions with and
        # with-out logarithm function"
        # https://core.ac.uk/download/pdf/12172514.pdf
        gap_star = np.mean(ref_dispersions) - dispersion
        sdk_star = np.sqrt(np.mean((ref_dispersions - dispersion) ** 2.0))
        sk_star = np.sqrt(1.0 + 1.0 / n_refs) * sdk_star

        return GapCalcResult(
            gap_value,
            int(n_clusters),
            ref_dispersions.std(),
            sdk,
            sk,
            gap_star,
            sk_star,
        )

    def _process_with_rust(
        self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray
    ) -> Generator[GapCalcResult, None, None]:
        """
        Process gap stat using pure rust
        """
        import gapstat_rs

        for (
            n_clusters,
            gap_value,
            ref_dispersion_std,
            sdk,
            sk,
            gap_star,
            sk_star,
        ) in gapstat_rs.optimal_k(X.astype(np.float64), cluster_array.astype(np.int64)):
            yield GapCalcResult(
                gap_value, n_clusters, ref_dispersion_std, sdk, sk, gap_star, sk_star
            )

    def _process_with_joblib(
        self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray
    ) -> Generator[GapCalcResult, None, None]:
        """
        Process calling of .calculate_gap() method using the joblib backend
        """
        if Parallel is None:
            raise EnvironmentError(
                "joblib is not installed; cannot use joblib as the parallel backend!"
            )

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for gap_calc_result in parallel(
                delayed(self._calculate_gap)(X, n_refs, n_clusters)
                for n_clusters in cluster_array
            ):
                yield gap_calc_result

    def _process_with_multiprocessing(
        self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray
    ) -> Generator[GapCalcResult, None, None]:
        """
        Process calling of .calculate_gap() method using the multiprocessing library
        """
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:

            jobs = [
                executor.submit(self._calculate_gap, X, n_refs, n_clusters)
                for n_clusters in cluster_array
            ]
            for future in as_completed(jobs):
                yield future.result()

    def _process_non_parallel(
        self, X: Union[pd.DataFrame, np.ndarray], n_refs: int, cluster_array: np.ndarray
    ) -> Generator[GapCalcResult, None, None]:
        """
        Process calling of .calculate_gap() method using no parallel backend; simple for loop generator
        """
        for gap_calc_result in [
            self._calculate_gap(X, n_refs, n_clusters) for n_clusters in cluster_array
        ]:
            yield gap_calc_result

    def __str__(self):
        return 'OptimalK(n_jobs={}, parallel_backend="{}")'.format(
            self.n_jobs, self.parallel_backend
        )

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        return "<p>{}</p>".format(self.__str__())
