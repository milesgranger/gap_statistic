use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_parallel::prelude::*;
use ndarray_rand::RandomExt;
use num_traits::pow::Pow;
use rand::distributions::Uniform;
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::iter::FromIterator;

use kmeans::{Centroid, KMeans};

pub fn convert_2d_vec_to_array(data: Vec<Vec<f64>>) -> Array2<f64> {
    // Convert vector to Array
    let shape = (data.len(), data[0].len());
    //let mut array = Array2::zeros(shape);
    let data = Array1::from_iter(data.into_iter().flat_map(|v| v))
        .into_shape(shape)
        .expect("Failed to reshape!");
    data
}

pub fn optimal_k(
    data: &ArrayView2<f64>,
    cluster_range: ArrayView1<i64>,
    iter: u32,
    n_refs: u32,
) -> Vec<GapCalcResult> {
    /*
        Given 2d data and a cluster range, return a vector of tuples
        where the first element represents n_clusters, gap value, and last is the ref disp std.
        (Higher the gap value, the better!)
    */

    // Get gap values for each cluster in range.
    let gap_values = cluster_range
        .into_par_iter()
        .map(|n_clusters| calculate_gap(data, *n_clusters as u32, iter, n_refs))
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

/// Generate a reference dataset from the original data
pub(crate) fn ref_dataset(data: &ArrayView2<f64>) -> Array2<f64> {
    let column_dists: Vec<_> = data
        .axis_iter(Axis(1))
        .map(|col| {
            Array::random(col.dim(), Uniform::new(col.min(), col.max())).insert_axis(Axis(0))
        })
        .collect();
    let column_dists_views: Vec<_> = column_dists.iter().map(|a| a.view()).collect();
    // Unwraps should be ok, since we haven't change the data/column sizes
    ndarray::stack(Axis(0), &column_dists_views)
        .unwrap()
        .t()
        .to_owned()
}

// Calculate the gap value
fn calculate_gap(data: &ArrayView2<f64>, n_clusters: u32, iter: u32, n_refs: u32) -> GapCalcResult {
    let mut km = KMeans::new(n_clusters, 0.00005, 20, iter);

    let ref_dispersions = (0..n_refs)
        .map(|_| {
            let random_data = ref_dataset(data);

            // Get centroids from data, each centroid contains .point() and .label()
            km.fit(&random_data.view());
            let labels = km.predict(&random_data.view());
            let centroids = km.centroids.as_ref().unwrap();
            calculate_dispersion(&random_data.view(), labels, centroids)
        })
        .collect::<Vec<f64>>();

    // Do calculations for the actual data
    km.fit(&data);
    let labels = km.predict(&data);
    let centroids = km.centroids.unwrap();

    let dispersion = calculate_dispersion(data, labels, &centroids);
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
fn calculate_dispersion(
    data: &ArrayView2<f64>,
    labels: Vec<u32>,
    centroids: &Vec<Centroid>,
) -> f64 {
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
