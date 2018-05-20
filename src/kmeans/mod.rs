#![allow(dead_code, unused)]

use std::iter::Sum;
use ndarray::{Array1, Array2, Ix2, Zip, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Range;
use statrs::statistics::Statistics;

// Centroid struct; hold place inside data
#[derive(Clone)]
pub struct Centroid {
    pub center: Array1<f64>,
    pub stable: bool,
    pub tolerance: f64
}


impl Centroid {

    pub fn new(center: Array1<f64>, tolerance: f64) -> Self {
        Centroid{
            center,
            tolerance,
            stable: false
        }
    }

    pub fn distance(&self, point: &Array1<f64>) -> f64 {
        /*
            Compuet the distance of a point from the center of this centroid
            // TODO: Implement metric other than Euclidean; and implement this better?
        */

        // Calculate Euclidean distance
        let distance: f64 = Sum::sum(
            point.into_iter()
                .zip(self.center.into_iter())
                .map(|(a, b): (&f64, &f64)| (a - b).powf(2.0))
        );
        distance.sqrt()
    }

    pub fn update(self, data: &Array2<f64>) -> Self {

        if self.stable {
            return Centroid{
                center: self.center,
                tolerance: self.tolerance,
                stable: false
            }
        }

        Centroid{
            center: self.center,
            tolerance: self.tolerance,
            stable: false
        }
    }

}

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

    pub fn fit(&mut self, data: &Array2<f64>) -> (){

        // Initialize centroids based on data passed
        self.centroids = Some(self.init_cenroids(&data));

        // for each centroid update location up until max_iter
        // TODO: Implement break if all centroids are stable.
        for _ in 0..self.max_iter {
            self.centroids = Some(
                self.centroids
                    .clone()
                    .unwrap()
                    .iter()
                    .map(|cent| cent.clone().update(&data))
                    .collect::<Vec<Centroid>>()
            )
        }
    }

    fn init_cenroids(&mut self, data: &Array2<f64>) -> Vec<Centroid> {

        // Create a sampling index from the data size
        let rand_sample = Array1::random(
            (data.dim().0,), Range::new(0, data.len() - 1)
        );

        // Take this random sampling and pull out points inside of the data for each centroid
        // to start with
        let centroids = rand_sample
            .iter()
            .map(|idx| Centroid::new(data.slice(s![*idx as i32, ..]).to_owned(), self.tolerance))
            .collect::<Vec<Centroid>>();

        centroids
    }
}