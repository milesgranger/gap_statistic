#![feature(proc_macro, specialization)]

pub extern crate pyo3;
#[macro_use] pub extern crate ndarray;
pub extern crate ndarray_rand;
pub extern crate ndarray_linalg;
pub extern crate rand;
pub extern crate statrs;
pub extern crate num_traits;


use pyo3::prelude::*;
use pyo3::py::modinit as pymodinit;

#[cfg(test)]
mod tests;
pub mod kmeans;
pub mod gap_statistic;


#[pymodinit(gapstat)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "optimal_k")]
    fn gap_statistic_py(data: Vec<Vec<f64>>, cluster_range: Vec<u32>) -> PyResult<Vec<(u32, f64)>> {
       Ok(gap_statistic::optimal_k(data, cluster_range))
    }

    Ok(())
}

