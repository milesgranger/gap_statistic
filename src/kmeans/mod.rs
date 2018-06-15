#![allow(dead_code, unused)]

use std::f64;
use std::ops::Index;
use std::cmp::PartialOrd;
use std::iter;
use std::iter::Sum;
use std::collections::HashMap;
use ndarray::{Array1, Array2, Ix2, Zip, Axis, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use rand::distributions::{Range};
use rand;
use rand::Rng;
use statrs::statistics::Min;


/// Centroid struct
/// Hold the current centroid location along with status such as if the centroid is stable
/// within the data given a tolerance.
#[derive(Clone, Debug)]
pub struct Centroid {
    pub center: Array1<f64>,
    pub label: u32,
    pub stable: bool,
    pub tolerance: f64
}

/// Implement the centroid, allow for updating centroid location based on new data points assigned
/// to it. As well as computing a distance between one point and the current center of the centroid.
impl Centroid {

    pub fn new(center: Array1<f64>, tolerance: f64, label: u32) -> Self {
        Centroid{
            center,
            label,
            tolerance,
            stable: false
        }
    }

    pub fn distance(point_a: &ArrayView1<f64>, point_b: &ArrayView1<f64>) -> f64 {
        /*
            Compuet the distance of a point from the center of this centroid
            // TODO: Implement metric other than Euclidean; and implement this better?
        */

        // Calculate Euclidean distance
        let distance: f64 = Sum::sum(
            point_a.into_iter()
                .zip(point_b.into_iter())
                .map(|(a, b): (&f64, &f64)| (a - b).powf(2f64))
        );
        distance.sqrt()
    }

    pub fn update(&mut self, data: &Array2<f64>) {
        /*
            Given new data (points assigned to this centroid), update the centroid
        */
        if self.stable {
            return
        }

        // Determine the average of this data.
        //println!("Data: {:?}", &data);
        let center = data.mean_axis(Axis(0));

        // Calculate the difference between new center and previous, set stable if <= tolerance
        let diff: f64 = Sum::sum(
            center.into_iter()
                .zip(self.center.into_iter())
                .map(|(current, original): (&f64, &f64)| (current - original) / original * 100f64)
        );

        self.center = center;
        self.stable = diff <= self.tolerance;
    }

}

/// KMeans Struct; implement K-Means based given k clusters and a tolerance for the centroids.
/// Max iterations cuts centroid convergance if it takes longer than max_iter.
pub struct KMeans {
    pub k: u32,
    pub tolerance: f64,
    pub max_iter: u32,
    pub centroids: Option<Vec<Centroid>>,
    pub iter: u32
}

impl KMeans {

    pub fn new(k: u32, tolerance: f64, max_iter: u32, iter: u32) -> Self {
        /*
            k:          Number of clusters
            tolerance:  A centroid must move this much before being updated, otherwise considered stable
            max_iter:   The maximum iterations over dataset to update centroids
            iter:       Number of times to repeat kmeans to find best init cycle to try and minimize dist
                        between points and their centroids
        */
        KMeans{
            k,
            tolerance,
            max_iter,
            iter,
            centroids: None,
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        /*
            Fit KMeans model to data
        */
        let mut lowest_error = f64::MAX;
        let mut labels = Vec::with_capacity(data.shape()[0]);

        for n_iteration in 0..self.iter {

            // Initialize centroids; keep track of the starting point
            let idx = rand::thread_rng().gen_range(0, data.shape()[0]) as usize;
            self.centroids = Some(self.init_centroids(&data, idx));


            // Try to converge centroids up to max_iter
            for i in 0..self.max_iter {

                // Get current centroid assignments for data
                labels = self.predict(&data);
                let mut n_stable = 0;


                // Assuming we have initialized centroids...
                if let Some(ref mut centroids) = self.centroids {

                    // Iter over centroids, collecting points assigned to that cluster, and then
                    // update the centroid center.
                    for ref mut centroid in centroids {

                        // Find indexes of points beloning to current centroid
                        let filtered_points = labels.iter()
                            .zip(data.outer_iter())
                            .enumerate()
                            .filter_map(|(idx, (label, point))| {
                                match *label == centroid.label {
                                    true => Some(idx as usize),
                                    false => None
                                }
                            })
                            .collect::<Vec<usize>>();

                        // Fetch those points and update centroid
                        let points = data.select(Axis(0), &filtered_points);

                        // Check if cluster has any assigned points, update if so.
                        if !&points.is_empty() {
                            centroid.update(&points);
                        }

                        if centroid.stable {
                            n_stable += 1;
                        }
                    }
                }

                // Check if all centroids are converged, and break if so.
                if n_stable == self.k {
                    break
                }

            }  // End of max iter attempts for centroid stabilization

            let centroids = self.centroids.clone().unwrap();
            let error = iter::repeat(&centroids)
                    .zip(labels.iter())
                    .zip(data.outer_iter())
                    .map(|((centroids, label), point)|
                             Centroid::distance(&centroids[*label as usize].center.view(), &point))
                    .sum::<f64>();
            if error < lowest_error {
                lowest_error = error;
                self.centroids = Some(centroids);
            }

        }  // End of kmeans iterations for best init

    }

    pub fn predict(&self, data: &Array2<f64>) -> Vec<u32> {
        /*
            Calculate which centroid each data point belongs to.
        */
        //let mut classifications = Vec::new();
        // Sort the data into which centroid it should be pushed to
        let mut classifications = Vec::new();
        for point in data.outer_iter() {

            if let Some(ref centroids) = self.centroids {
                let distances = centroids
                    .iter()
                    .map(|centroid| Centroid::distance(&centroid.center.view(), &point))
                    .collect::<Vec<f64>>();
                let mut min = f64::MAX;
                let mut label= 0;
                for (dist, centroid) in distances.iter().zip(centroids.iter()) {
                    if dist < &min {
                        min = *dist;
                        label = centroid.label;
                    }
                }
                classifications.push(label);
            } else {
                panic!("Centroids are non-existant!");
            }
        }
        classifications
    }

    fn calculate_error(&self) -> f64 {
        /*
            Given centroids and their assigned points, calculate the total distance as error.
        */
        6.5
    }

    fn init_centroids(&mut self, data: &Array2<f64>, start_idx: usize) -> Vec<Centroid> {

        // Set vector of indices representing points to be assigned to centroids
        let mut indices = Vec::with_capacity(self.k as usize);

        // Choose first point at random if not passed directly
        indices.push(start_idx);

        // Start choosing new centroid locations based on k-means++
        let distances = None;
        while indices.len() < self.k as usize {

            // Get the most recent added center
            let center = data.slice(s![indices[indices.len()-1], ..]);

            // Get normalized distances from center
            let distances = Self::normed_distances_from_point(&center, &data, distances);

            // Choose new center based on normed distances
            let new_centroid_idx = Self::choose_next_centroid_idx(distances.expect("Didn't get distances back!").clone());
            indices.push(new_centroid_idx);

        }

        let centroids = data.select(Axis(0), &indices)
            .outer_iter()
            .enumerate()
            .map(|(i, point)| Centroid::new(point.to_owned(), self.tolerance, i as u32))
            .collect::<Vec<Centroid>>();
        centroids
    }

    fn choose_next_centroid_idx(mut normed_distances: Array1<f64>) -> usize {
        /*
            Based on normed distances, choose the new centroid by returning the index to select on from dataset
        */


        let sum = normed_distances.sum_axis(Axis(0));
        normed_distances = normed_distances / &sum;
        let cumsum = normed_distances
            .iter()
            .scan(0.0, |cs: &mut f64, value| {
                *cs += *value;
                Some(*cs)
            })
            .collect::<Vec<f64>>();

        let random_prob = rand::thread_rng().gen_range(0_f64, 1_f64);
        let new_centroid_idx = cumsum
            .iter()
            .enumerate()
            .filter_map(|(i, prob)| {
                if prob >= &random_prob {
                    Some(i)
                } else {
                    None
                }
            })
            .next()
            .expect(&format!("No probabilities found greater than {}", &random_prob));

        new_centroid_idx

    }

    fn normed_distances_from_point(center: &ArrayView1<f64>, points: &Array2<f64>, previous_distances: Option<&Array1<f64>>) -> Option<Array1<f64>> {
        /*
            Calculate the abs distance for every point from center and return a normal distribution
            from there; only if the new distance is less than the previous distance (if any supplied)
        */

        // Compute normed distances from each point to center
        let mut distances = points
            .outer_iter()
            .map(|point|
                (point.to_owned() - center.to_owned())
                    .to_vec()
                    .iter()
                    .map(|v| v.abs().powf(2_f64))
                    .sum()
            )
            .map(|normalized: f64| (normalized).powf(2_f64))
            .collect::<Vec<f64>>();

        // If previous distances were passed, return the lowest dist when compared against new distances
        // otherwise return the current distances.
        match previous_distances {
            Some(prev_distances) => {
                let distances = prev_distances
                    .iter()
                    .zip(distances.iter())
                    .map(|(old_dist, new_dist)| {
                       if new_dist < old_dist {
                           *new_dist
                       } else {
                           *old_dist
                       }
                    })
                    .collect();
                Some(Array1::from_vec(distances))
            },
            None => {
                Some(Array1::from_vec(distances))
            }
        }
    }

}