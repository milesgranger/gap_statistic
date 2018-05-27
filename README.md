
### Python implementation of the [Gap Statistic](http://www.web.stanford.edu/~hastie/Papers/gap.pdf)

![Build Status](https://travis-ci.org/milesgranger/gap_statistic.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/milesgranger/gap_statistic/badge.svg)](https://coveralls.io/github/milesgranger/gap_statistic)
[![Code Health](https://landscape.io/github/milesgranger/gap_statistic/master/landscape.svg?style=flat)](https://landscape.io/github/milesgranger/gap_statistic/master)


[![Anaconda](https://anaconda.org/milesgranger/gap-stat/badges/version.svg)](https://anaconda.org/milesgranger/gap-stat)
[![Anaconda](https://anaconda.org/milesgranger/gap-stat/badges/installer/conda.svg)](https://anaconda.org/milesgranger/gap-stat)
---
#### Purpose
Dynamically identify the suggested number of clusters in a data-set
using the gap statistic.

---

### Full example available in a notebook [HERE](Example.ipynb)

---
#### Install:  
Bleeding edge: **Will require Rust the latest rust nightly; and highly unstable!**
```commandline
pip install git+git://github.com/milesgranger/gap_statistic.git
```

PyPi:  
```commandline
pip install --upgrade gap-stat
```

Anaconda
```commandline
conda install -c milesgranger gap-stat
```

---
#### Uninstall:
```commandline
pip uninstall gap-stat
```


Change Log:

- 1.0.1
    - Sept 2017 
    - Initial release; pure python featuring parallel backends 'joblib' and 'multiprocessing'
    - Dependencies: 
        - numpy
        - scipy
        - pandas
        - joblib (optional)
    - Install: `pip install gap-stat==1.0.1`
  
- 1.5.0a1
    - 27-May-2018
    - First alpha of using Rust. `optimalK` now takes `"rust"` as an argument for the parallel_backend
      This is the fastest backend, where joblib w/ scipy had ~7.5s in benchmark, Rust is ~4.5s
    - Dependencies:
        - numpy
        - scipy
        - pandas
        - joblib (optional)
    - Install: `pip install gap-stat==1.5.0a1` (Not available on Windows)
