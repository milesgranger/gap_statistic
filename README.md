
### Python implementation of the [Gap Statistic](http://www.web.stanford.edu/~hastie/Papers/gap.pdf)

Linux & OSX: [![Build Status](https://travis-ci.org/milesgranger/gap_statistic.svg?branch=master)](https://travis-ci.org/milesgranger/gap_statistic)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Windows: [![Build status](https://ci.appveyor.com/api/projects/status/fbnv8del2qkx56j1?svg=true)](https://ci.appveyor.com/project/milesgranger/gap-statistic)  

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
Bleeding edge: **Will require Rust the latest rust nightly**
```commandline
pip install git+git://github.com/milesgranger/gap_statistic.git
```

PyPi:  
```commandline
pip install --upgrade gap-stat
```

Anaconda (Only available for 1.0.1 release (no rust backend))
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
  
- 1.5.0a(1,2)
    - 27-May-2018
    - First alpha of using Rust. `optimalK` now takes `"rust"` as an argument for the parallel_backend
      This is the fastest backend, where joblib w/ scipy had ~7.5s in benchmark, Rust is ~4.5s  
      *Pre-built wheels for Linux & OSX for Python 3.5 & 3.6. Windows users can install but will need
      Rust nightly in order to compile.*
    - Dependencies:
        - numpy
        - scipy
        - pandas
        - joblib (optional)
    - Install: `pip install gap-stat==1.5.0a2` (Not available on Windows)
    
- 1.5.0
    - June-2018
    - More tests, add best init in rust kmeans impl for better stability between runs
    - Dependencies:
        - numpy
        - scipy
        - pandas
        - joblib (optional)
    - Install `pip install gap-stat==1.5.0` 
        - *(Windows users will not have the Rust backend  
      unless they have Rust nightly available; OSX and Linux have pre-compiled wheels)*
