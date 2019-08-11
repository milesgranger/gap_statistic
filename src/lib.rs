#![feature(specialization, test)]

#[macro_use]
pub extern crate pyo3;
#[macro_use]
pub extern crate ndarray;
pub extern crate ndarray_parallel;
pub extern crate ndarray_rand;
pub extern crate num_traits;
pub extern crate rand;
pub extern crate rayon;
pub extern crate statrs;
extern crate test;

use pyo3::prelude::*;

pub mod gap_statistic;
pub mod kmeans;
#[cfg(test)]
mod tests;

#[pyfunction]
fn kmeans(data: Vec<Vec<f64>>, k: u32, max_iter: u32, iter: u32) -> PyResult<Vec<u32>> {
    let data = gap_statistic::convert_2d_vec_to_array(data);
    let (_centroids, labels) = gap_statistic::kmeans(&data, k, max_iter, iter);
    Ok(labels)
}

#[pyfunction]
fn optimal_k(
    data: Vec<Vec<f64>>,
    cluster_range: Vec<u32>,
    iter: Option<u32>,
) -> PyResult<Vec<(u32, f64, f64, f64, f64, f64, f64)>> {
    let gapcalcs = if let Some(iterations) = iter {
        gap_statistic::optimal_k(data, cluster_range, iterations)
    } else {
        gap_statistic::optimal_k(data, cluster_range, 10)
    };
    let result = gapcalcs
        .into_iter()
        .map(|gapcalc| {
            (
                gapcalc.n_clusters,
                gapcalc.gap_value,
                gapcalc.ref_dispersion_std,
                gapcalc.sdk,
                gapcalc.sk,
                gapcalc.gap_star,
                gapcalc.sk_star,
            )
        })
        .collect();
    Ok(result)
}

#[pymodinit]
fn gapstat(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_function!(kmeans))?;
    m.add_function(wrap_function!(optimal_k))?;
    Ok(())
}
