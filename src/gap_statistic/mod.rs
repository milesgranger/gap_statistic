#![allow(dead_code, unused)]  // TODO: Remove this when things settle into place.

use std::collections::HashMap;
use std::iter::FromIterator;
use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use num_traits::pow::Pow;
use statrs::statistics::Mean;
use statrs::statistics::Statistics;

use kmeans::{KMeans, Centroid};

// Kmeans Entry point
// TODO: Implement this
fn kmeans<'a>(data: &'a Array2<f64>, k: u32, max_iter: u32, minit: &str) -> (Vec<Centroid>, Vec<u32>) {

    let mut kmeans = KMeans::new(k, 0.001, max_iter);
    kmeans.fit(&data);
    let labels = kmeans.predict(&data);
    (kmeans.centroids.expect("No centroids inside of KMeans model!").clone(), labels.clone())
}


pub fn optimal_k(data: Vec<Vec<f64>>, cluster_range: Vec<u32>) -> Vec<(u32, f64)> {
    /*
        Given 2d data and a cluster range, return a vector of tuples
        where the first element represents n_clusters, and second represents the gap value.
        (Higher the gap value, the better!)
    */
    // Convert vector to Array
    let mut array = Array2::zeros((data.len(), data[0].len()));
    for vec in data {
        array.assign(&Array1::from_vec(vec));
    }

    // Get gap values for each cluster in range.
    let gap_values = cluster_range
        .iter()
        .map(|n_clusters| (*n_clusters, calculate_gap(&array, *n_clusters)))
        .collect::<Vec<(u32, f64)>>();

    gap_values
}


// Calculate the gap value
fn calculate_gap(data: &Array2<f64>, n_clusters: u32) -> f64 {

    let n_refs = 5; // TODO: Add this as parameter
    let mut ref_dispersions = Array1::zeros((n_refs,));

    // For each reference check, run k-means on random data resembling shape of data.
    for i in 0..n_refs {

        // Generate some random data for this round
        let random_data = Array2::random(data.dim(), Range::new(-100_f64, 100_f64));
        println!("Random data: {:?}", &random_data);

        // Get centroids from data, each centroid contains .point() and .label()
        let (centroids, labels) = kmeans(&random_data, n_clusters, 10, "points");
        ref_dispersions[i] = calculate_dispersion(&random_data, labels, centroids);
    }

    // Do calculations for the actual data
    let (centroids, labels) = kmeans(&data, n_clusters, 10, "points");
    let dispersion = calculate_dispersion(&data, labels, centroids);


    // Calculate and return gap value
    let gap_value = (ref_dispersions.into_iter().mean() + 1f64).log2() - (dispersion + 1f64).log2();
    gap_value
}


// Calculate the dispersion
// TODO: Implement this
fn calculate_dispersion(data: &Array2<f64>, labels: Vec<u32>, centroids: Vec<Centroid>) -> f64 {

    // Place centroids in hashmap for looking up
    let centroid_lookup = HashMap::<u32, Array1<f64>>::from_iter(
        centroids.iter().map(|centroid| (centroid.label as u32, centroid.center.clone()))
    );

    println!("Keys: {:?}", &centroid_lookup.keys());

    let dispersion: f64 = labels.iter()
        .zip(data.outer_iter())
        .map(|(label, point)|
                 point.iter()
                     .zip(centroid_lookup
                         .get(label as &u32)
                         .expect(&format!("Couldn't find point for label: {}!", label))
                         .iter())
                     .map(|(a, b): (&f64, &f64)| (a - b).abs().pow(2))
                     .sum::<f64>())
        .sum();
    dispersion
}