#![allow(dead_code, unused)]

use ndarray::Array2;

// Centroid struct; hold place inside data
// TODO: Implement this
pub struct Centroid<'a> {
    pub data: &'a Array2<f64>
}

struct KMeans {
    k: u32,
    tolerance: f64,
    max_iterations: u32
}