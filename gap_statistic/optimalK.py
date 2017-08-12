import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from scipy.cluster.vq import kmeans2
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed


class OptimalK:

    def __init__(self, n_jobs: int=-1) -> None:
        """
        Construct OptimalK to use n_jobs (multiprocessing usign joblib.
        """
        self.n_jobs = n_jobs if n_jobs >= 1 else cpu_count()  # type: int

    @staticmethod
    def calculate_gap(X, n_refs, gap_index, k):

        # Holder for reference dispersion results
        ref_dispersions = np.zeros(n_refs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(n_refs):
            # Create new random reference set
            random_data = np.random.random_sample(size=X.shape)

            # Fit to it
            centeroids, labels = kmeans2(random_data, k, iter=10, minit='points')
            disp = np.sum(np.sum([np.abs(inst - centeroids[label]) ** 2 for inst, label in zip(random_data, labels)]))
            ref_dispersions[i] = disp

        # Fit cluster to original data and create dispersion
        centeroids, labels = kmeans2(X, k, iter=10, minit='points')
        disp = np.sum(np.sum([np.abs(inst - centeroids[label]) ** 2 for inst, label in zip(X, labels)]))

        # Calculate gap statistic
        gap = np.log(np.mean(ref_dispersions)) - np.log(disp)

        return gap, gap_index, k

    def __call__(self, data, n_refs=3, maxClusters=15):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (optimalK, gaps_dataframe)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        df_results = pd.DataFrame({'clusterCount': [], 'gap': []})

        with Parallel(n_jobs=self.n_jobs) as parallel:
            for gap_result, gap_index, k in parallel(delayed(self.calculate_gap)(data, n_refs, gap_index, k)
                                                     for gap_index, k in enumerate(range(1, maxClusters))):
                # Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap_result

                df_results = df_results.append({'clusterCount': k, 'gap': gap_result}, ignore_index=True)

        # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        #
        #     jobs = {executor.submit(self.calculate_gap, data, nrefs, gap_index, k): k
        #             for gap_index, k in enumerate(range(1, maxClusters))
        #             }
        #
        #     for future in as_completed(jobs):
        #         cluster_count = jobs[future]
        #         try:
        #             gap_result, gap_index, k = future.result()
        #         except Exception as exc:
        #             print('Got the following exception:\n{}'.format(exc))
        #         else:
        #             # Assign this loop's gap statistic to gaps
        #             gaps[gap_index] = gap_result
        #
        #             df_results = df_results.append({'clusterCount': k, 'gap': gap_result}, ignore_index=True)

        return (gaps.argmax() + 1, df_results)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
