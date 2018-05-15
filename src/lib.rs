#![feature(proc_macro, specialization)]

extern crate pyo3;

extern crate ndarray;
extern crate num;

use pyo3::prelude::*;
use pyo3::py::modinit as pymodinit;


#[pymodinit(rustgap)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "optimalK")]
    fn optimalK_py(data: Vec<Vec<f64>>, clusters: Vec<u32>) -> PyResult(u32) {
        Ok(optimalK())
    }
    Ok(())
}

fn optimalK() -> u32 {
    2
}