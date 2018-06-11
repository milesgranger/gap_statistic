#![feature(proc_macro, specialization)]

pub extern crate pyo3;
#[macro_use] pub extern crate ndarray;
pub extern crate ndarray_rand;
pub extern crate ndarray_linalg;
pub extern crate ndarray_parallel;
pub extern crate rayon;
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

    #[pyfn(m, "kmeans")]
    fn kmeans_py(data: Vec<Vec<f64>>, k: u32, max_iter: u32, iter: u32) -> PyResult<Vec<u32>> {
        let data = gap_statistic::convert_2d_vec_to_array(data);
        let (_centroids, labels) = gap_statistic::kmeans(&data, k, max_iter, iter);
        Ok(labels)
    }

    #[pyfn(m, "optimal_k")]
    fn gap_statistic_py(data: Vec<Vec<f64>>, cluster_range: Vec<u32>, iter: Option<u32>) -> PyResult<Vec<(u32, f64)>> {
        if let Some(iterations) = iter {
            Ok(gap_statistic::optimal_k(data, cluster_range, iterations))
        } else {
            Ok(gap_statistic::optimal_k(data, cluster_range, 10))
        }
    }

    Ok(())
}

