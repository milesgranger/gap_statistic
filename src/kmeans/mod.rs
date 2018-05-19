#![allow(dead_code, unused)]

use ndarray::{Array1, Array2, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

// Centroid struct; hold place inside data
#[derive(Clone)]
pub struct Centroid {
    pub center: Array1<f64>,
    pub stable: bool
}


impl Centroid {

    pub fn new(center: Array1<f64>) -> Self {
        Centroid{
            center,
            stable: false
        }
    }

    pub fn update(self, data: &Array2<f64>) -> Self {
        Centroid{
            center: self.center,
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

    pub fn fit(mut self, data: &Array2<f64>) -> () {

        // Initialize centroids based on data passed
        self.centroids = Some(self.init_cenroids(&data));

        // for each centroid update location up until max_iter
        for _ in 0..self.max_iter {
            self.centroids = Some(
                self.centroids
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
            .map(|idx| Centroid::new(data.slice(s![*idx as i32, ..]).to_owned()))
            .collect::<Vec<Centroid>>();

        centroids
    }
}