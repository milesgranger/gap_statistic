
## gap_statistic
### Python implementation of the [Gap Statistic]('gap statistic' http://www.web.stanford.edu/~hastie/Papers/gap.pdf) using pandas, numpy, and SciKit-Learn

---
#### Purpose
Dynamically identify the number of optimal clusters given data using KMeans iteratively to calculate the gap statistic for each possible cluster count from 1 to maxClusters. Identifies the optimal cluster count in that range by returning cluster count which resulted in the highest gap statistic

---

#### Usage:

<p><code>from gap_statistic.optimalK import optimalK</code></p>

Parameters:
- data: ndarray of shape (n_samples, n_features)
- nrefs: number of random sample data sets to produce
- maxClusters: maximum number of clusters to look for

---
#### Install:

cd into site-packages directory and run: <p><code>git clone https://github.com/milesgranger/gap_statistic.git</code></p>
