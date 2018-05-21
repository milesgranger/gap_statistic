#![allow(dead_code, unused)]

use std::iter::Sum;
use ndarray::{Array1, Array2, Ix2, Zip, Axis, ArrayView1, ArrayView2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use statrs::statistics::Statistics;

/// Centroid struct
/// Hold the current centroid location along with status such as if the centroid is stable
/// within the data given a tolerance.
#[derive(Clone)]
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

    pub fn distance(&self, point: &ArrayView1<f64>) -> f64 {
        /*
            Compuet the distance of a point from the center of this centroid
            // TODO: Implement metric other than Euclidean; and implement this better?
        */

        // Calculate Euclidean distance
        let distance: f64 = Sum::sum(
            point.into_iter()
                .zip(self.center.into_iter())
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
        let center: Array1<f64> = data.mean_axis(Axis(0));

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
    pub centroids: Option<Vec<Centroid>>
}

impl KMeans {

    pub fn new(k: u32, tolerance: f64, max_iter: u32) -> Self {

        KMeans{
            k,
            tolerance,
            max_iter,
            centroids: None
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        /*
            Fit KMeans model to data
        */

        // Initialize centroids based on data passed
        self.centroids = Some(self.init_centroids(&data));

        // Try to converge centroids up to max_iter
        for i in 0..self.max_iter {

            // Get current centroid assignments for data
            let labels = self.predict(&data);
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
                    centroid.update(&points);
                    if centroid.stable {
                        n_stable += 1;
                    }
                }
            }

            // Check if all centroids are converged, and break if so.
            if n_stable == self.k {
                break
            }
        }
    }

    pub fn predict(&self, data: &Array2<f64>) -> Vec<u32> {
        /*
            Calculate which centroid each data point belongs to.
        */
        //let mut classifications = Vec::new();
        // Sort the data into which centroid it should be pushed to
        let mut classifications = Vec::new();
        for point in data.axis_iter(Axis(1)) {

            if let Some(ref centroids) = self.centroids {
                let distances = centroids
                    .iter()
                    .map(|centroid| centroid.distance(&point))
                    .collect::<Vec<f64>>();
                let max: f64 = Sum::sum(distances.iter());
                let label = distances
                    .iter()
                    .position(|&x| x == max)
                    .unwrap() as u32;
                classifications.push(label);
            } else {
                panic!("Centroids are non-existant!");
            }
        }
        classifications
    }

    fn init_centroids(&mut self, data: &Array2<f64>) -> Vec<Centroid> {

        // Create a sampling index from the data size
        // TODO: Implement random sampling without replacement.
        let rand_sample = Array1::random(
            (data.dim().0,), Range::new(0, data.len() - 1)
        );

        // Take this random sampling and pull out points inside of the data for each centroid
        // to start with
        let centroids = rand_sample
            .iter()
            .enumerate()
            .map(|(label, idx)| Centroid::new(data.slice(s![*idx as i32, ..]).to_owned(), self.tolerance, label as u32))
            .collect::<Vec<Centroid>>();

        centroids
    }
}