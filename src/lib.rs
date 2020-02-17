#![feature(specialization, test)]

#[macro_use]
pub extern crate ndarray;
pub extern crate ndarray_parallel;
pub extern crate ndarray_rand;
pub extern crate num_traits;
pub extern crate numpy;
pub extern crate pyo3;
pub extern crate rand;
pub extern crate rayon;
pub extern crate statrs;
extern crate test;

use kmeans::KMeans;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

pub mod gap_statistic;
pub mod kmeans;
#[cfg(test)]
mod tests;

#[pymodule]
fn gapstat_rs(_py: Python, m: &PyModule) -> Result<(), pyo3::PyErr> {
    #[pyfn(m, "kmeans")]
    fn kmeans(data: Vec<Vec<f64>>, k: u32, max_iter: u32, iter: u32) -> PyResult<Vec<u32>> {
        let data = gap_statistic::convert_2d_vec_to_array(data);
        let km = KMeans::new(k, 0.00005, max_iter, iter);
        let labels = km.predict(&data.view());
        Ok(labels)
    }

    #[pyfn(m, "optimal_k")]
    fn optimal_k(
        py: Python,
        data: &PyArray2<f64>,
        cluster_range: &PyArray1<i64>,
        n_iter: Option<i32>,
        n_refs: Option<i32>,
    ) -> PyResult<Vec<(i64, f64, f64, f64, f64, f64, f64)>> {
        let x = data.as_array();
        let cr = cluster_range.as_array();
        let n_refs = n_refs.unwrap_or_else(|| 10) as u32;
        let n_iter = n_iter.unwrap_or_else(|| 10) as u32;

        let gapcalcs =
            py.allow_threads(move || gap_statistic::optimal_k(&x.view(), cr, n_iter, n_refs));

        let results = gapcalcs
            .into_iter()
            .map(|gapcalc| {
                (
                    gapcalc.n_clusters as i64,
                    gapcalc.gap_value,
                    gapcalc.ref_dispersion_std,
                    gapcalc.sdk,
                    gapcalc.sk,
                    gapcalc.gap_star,
                    gapcalc.sk_star,
                )
            })
            .collect();

        Ok(results)
    }

    Ok(())
}
