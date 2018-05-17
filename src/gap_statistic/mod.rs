use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Range;

// Obtain the optimal clusters for the given dataset.
pub fn optimal_k(data: Vec<Vec<f64>>, cluster_range: Vec<u32>) -> u32 {

    // Convert vector to Array
    let data = Array2::from_shape_vec(
        (data.len(), data[0].len()), data
    ).unwrap(); // TODO: deal with this better than using unwrap, pass error if present

    5
}


// Calculate the gap value
fn calculate_gap(data: Array2<f64>, n_clusters: u32) -> f64 {
    let n_refs = 5; // TODO: Add this as parameter

    for i in 0..n_refs {
        Array2::random((2,2), Range::new(0, 5));
    }

    5.6
}