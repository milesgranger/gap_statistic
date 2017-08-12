import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


class OptimalK:

    def __init__(self, n_jobs: int=-1) -> None:
        """
        Construct OptimalK to use n_jobs (multiprocessing usign joblib.
        """
        self.n_jobs = n_jobs if n_jobs >= 1 else cpu_count()  # type: int

    @staticmethod
    def calculate_gap(data, nrefs, gap_index, k):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        return gap, gap_index, k

    def __call__(self, data, nrefs=3, maxClusters=15):
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

            for gap_result, gap_index, k in parallel(delayed(self.calculate_gap)(data, nrefs, gap_index, k)
                                                     for gap_index, k in enumerate(range(1, maxClusters))):

                # Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap_result

                df_results = df_results.append({'clusterCount': k, 'gap': gap_result}, ignore_index=True)

        return (gaps.argmax() + 1, df_results)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
