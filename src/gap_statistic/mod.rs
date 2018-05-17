#![allow(dead_code, unused)]  // TODO: Remove this when things settle into place.

use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use statrs::statistics::Mean;
use statrs::statistics::Statistics;

use kmeans::Centroid;

// Kmeans Entry point
// TODO: Implement this
fn kmeans<'a>(data: &'a Array2<f64>, k: u32, iter: u32, minit: &str) -> (Vec<Centroid<'a>>, Vec<u32>) {

    (vec![Centroid{data: &data}, Centroid{data: &data}], vec![1, 2])
}

// Obtain the optimal clusters for the given dataset.
pub fn optimal_k(data: Vec<Vec<f64>>, cluster_range: Vec<u32>) -> u32 {

    // Convert vector to Array
    // TODO: deal with this better than using unwrap, pass error if present
    let data = Array2::from_shape_vec(
    (data.len(), data[0].len()), data
    ).unwrap();

    5
}


// Calculate the gap value
fn calculate_gap(data: &Array2<f64>, n_clusters: u32) -> f64 {

    let n_refs = 5; // TODO: Add this as parameter
    let mut ref_dispersions = Array1::zeros((n_refs,));

    // For each reference check, run k-means on random data resembling shape of data.
    for i in 0..n_refs {

        // Generate some random data for this round
        let random_data = Array2::random(data.dim(), Range::new(0.0, 1.0));

        // Get centroids from data, each centroid contains .point() and .label()
        let (centroids, labels) = kmeans(&random_data, n_clusters, 10, "points");
        ref_dispersions[i] = calculate_dispersion(&random_data, labels, centroids);
    }

    // Do calculations for the actual data
    let (centroids, labels) = kmeans(&data, n_clusters, 10, "points");
    let dispersion = calculate_dispersion(&data, labels, centroids);

    // Calculate and return gap value
    let gap_value = ref_dispersions.into_iter().mean().log2() - dispersion.log2();
    gap_value
}


// Calculate the dispersion
// TODO: Implement this
fn calculate_dispersion(data: &Array2<f64>, labels: Vec<u32>, centroids: Vec<Centroid>) -> f64 {
    1.1
}