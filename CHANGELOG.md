#### Changelog

All noteable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## 1.7.1 - 2019.08.18
### Added
  - Optimize Rust backend
  - Benchmark setup
  - Create a separate CHANGELOG

---

## 1.7.0 - 2019.07
## Added
  - Add OSX support to CI/CD [#31](https://github.com/milesgranger/gap_statistic/pull/31)
## Changed
  - Fix passing of null dataset in python gap value calculations [#34](https://github.com/milesgranger/gap_statistic/pull/34)
  - Use advanced method for choosing k, documentation updates, plotting, other enhancements. [#28](https://github.com/milesgranger/gap_statistic/pull/28)
  - Change license to dual Unlicense / MIT [#30](https://github.com/milesgranger/gap_statistic/pull/30)
  
## 1.6.1 - 2019.05
## Added
  - Return the reference distributions' standard deviation in the dataframe results
  
## 1.6 - 2019.05
### Added
  - Support user defined functions for the clustering algorithm used in the gap statistic process
  - Migrate to Azure Pipelines C
  
## 1.5.2 - 2018.08
### Changed
  - Fix calculation of gap statistic

## 1.5.1 - 2018.06
### Added
  - First stable release with Rust backend. OptimalK now takes `backend="rust"`
  - More tests, add best init in rust kmeans impl for better stability between runs
  
## 1.5.0a(1,2) - 2018.05.27
### Added
  - First alpha of using Rust. `optimalK` now takes `"rust"` as an argument for the parallel_backend
    This is the fastest backend, where joblib w/ scipy had ~7.5s in benchmark, Rust is ~4.5s  
    *Pre-built wheels for Linux & OSX for Python 3.5 & 3.6. Windows users can install but will need
    Rust nightly in order to compile.*
    
## 1.0.1 - 2017.09
### Added
  - First release - pure python implmentation
  - Install: `pip install gap-stat==1.0.1`
  - OR conda install -c milesgranger gap-stat==1.0.1
