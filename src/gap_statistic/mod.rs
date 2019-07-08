use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use num_traits::pow::Pow;
use rand::distributions::Range;
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::iter::FromIterator;

use kmeans::{Centroid, KMeans};

// Kmeans Entry point
pub fn kmeans<'a>(
    data: &'a Array2<f64>,
    k: u32,
    max_iter: u32,
    iter: u32,
) -> (Vec<Centroid>, Vec<u32>) {
    let mut kmeans = KMeans::new(k, 0.001, max_iter, iter);
    kmeans.fit(&data);
    let labels = kmeans.predict(&data);
    (
        kmeans
            .centroids
            .expect("No centroids inside of KMeans model!")
            .clone(),
        labels.clone(),
    )
}

pub fn convert_2d_vec_to_array(data: Vec<Vec<f64>>) -> Array2<f64> {
    // Convert vector to Array
    let shape = (data.len(), data[0].len());
    //let mut array = Array2::zeros(shape);
    let data = Array1::from_iter(data.iter().flat_map(|v| v.clone())) // TODO: Do better than clone...
        .into_shape(shape)
        .expect("Failed to reshape!");
    data
}

pub fn optimal_k(data: Vec<Vec<f64>>, cluster_range: Vec<u32>, iter: u32) -> Vec<GapCalcResult> {
    /*
        Given 2d data and a cluster range, return a vector of tuples
        where the first element represents n_clusters, gap value, and last is the ref disp std.
        (Higher the gap value, the better!)
    */
    let data = convert_2d_vec_to_array(data);
    let cluster_range = Array1::from_vec(cluster_range);

    // Get gap values for each cluster in range.

    let gap_values = cluster_range
        .into_iter()
        .map(|n_clusters| calculate_gap(&data, n_clusters.clone(), iter))
        .collect::<Vec<GapCalcResult>>();

    gap_values
}

pub struct GapCalcResult {
    pub(crate) gap_value: f64,
    pub(crate) n_clusters: u32,
    pub(crate) ref_dispersion_std: f64,
    pub(crate) sdk: f64,
    pub(crate) sk: f64,
    pub(crate) gap_star: f64,
    pub(crate) sk_star: f64,
}

// Calculate the gap value
fn calculate_gap(data: &Array2<f64>, n_clusters: u32, iter: u32) -> GapCalcResult {
    let n_refs = 5; // TODO: Add this as parameter
    let mut ref_dispersions = Array1::zeros((n_refs,));

    // For each reference check, run k-means on random data resembling shape of data.
    for i in 0..n_refs {
        // Generate some random data for this round
        let random_data = Array2::random(data.dim(), Range::new(-1_f64, 1_f64));

        // Get centroids from data, each centroid contains .point() and .label()
        let (centroids, labels) = kmeans(&random_data, n_clusters, 5, iter);
        ref_dispersions[i] = calculate_dispersion(&random_data, labels, centroids);
    }

    // Do calculations for the actual data
    let (centroids, labels) = kmeans(&data, n_clusters, 5, iter);

    let ref_dispersions = ref_dispersions.to_vec();

    let dispersion = calculate_dispersion(&data, labels, centroids);
    let ref_log_dispersion = ref_dispersions.iter().map(|v| v.log2()).mean();
    let log_dispersion = dispersion.log2();
    let gap_value = ref_log_dispersion - log_dispersion;
    let sdk = ref_dispersions
        .iter()
        .map(|v| (v.log2() - ref_log_dispersion).powf(2.0))
        .mean()
        .sqrt();
    let sk = (1. + 1. / n_refs as f64).sqrt() * sdk;
    let gap_star = ref_dispersions.iter().mean() - dispersion;
    let sdk_star = ref_dispersions
        .iter()
        .map(|v| (v - dispersion).powf(2.0))
        .mean()
        .sqrt();
    let sk_star = (1. + 1. / n_refs as f64).sqrt() * sdk_star;
    let ref_dispersion_std = ref_dispersions.iter().std_dev();

    GapCalcResult {
        gap_value,
        n_clusters,
        ref_dispersion_std,
        sdk,
        sk,
        gap_star,
        sk_star,
    }
}

// Calculate the dispersion
// TODO: Implement this
fn calculate_dispersion(data: &Array2<f64>, labels: Vec<u32>, centroids: Vec<Centroid>) -> f64 {
    // Place centroids in hashmap for looking up
    let centroid_lookup = HashMap::<u32, Array1<f64>>::from_iter(
        centroids
            .iter()
            .map(|centroid| (centroid.label as u32, centroid.center.clone())),
    );

    let dispersion: f64 = labels
        .iter()
        .zip(data.outer_iter())
        .map(|(label, point)| {
            point
                .iter()
                .zip(
                    centroid_lookup
                        .get(label as &u32)
                        .expect(&format!("Couldn't find point for label: {}!", label))
                        .iter(),
                )
                .map(|(a, b): (&f64, &f64)| (a - b).abs().pow(2))
                .sum::<f64>()
        })
        .sum();
    dispersion
}
