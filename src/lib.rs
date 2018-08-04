#![feature(use_extern_macros, specialization)]


#[macro_use] pub extern crate pyo3;
#[macro_use] pub extern crate ndarray;
pub extern crate ndarray_rand;
pub extern crate ndarray_parallel;
pub extern crate rayon;
pub extern crate rand;
pub extern crate statrs;
pub extern crate num_traits;

use pyo3::prelude::*;

#[cfg(test)]
mod tests;
pub mod kmeans;
pub mod gap_statistic;

#[pyfunction]
fn kmeans(data: Vec<Vec<f64>>, k: u32, max_iter: u32, iter: u32) -> PyResult<Vec<u32>> {
    let data = gap_statistic::convert_2d_vec_to_array(data);
    let (_centroids, labels) = gap_statistic::kmeans(&data, k, max_iter, iter);
    Ok(labels)
}

#[pyfunction]
fn optimal_k(data: Vec<Vec<f64>>, cluster_range: Vec<u32>, iter: Option<u32>) -> PyResult<Vec<(u32, f64)>> {
    if let Some(iterations) = iter {
        Ok(gap_statistic::optimal_k(data, cluster_range, iterations))
    } else {
        Ok(gap_statistic::optimal_k(data, cluster_range, 10))
    }
}


#[pymodinit]
fn gapstat(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_function!(kmeans))?;
    m.add_function(wrap_function!(optimal_k))?;
    Ok(())
}
