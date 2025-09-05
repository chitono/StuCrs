use ndarray::*;
use rand;
use rand::seq::{index, SliceRandom};

struct DataLoader {
    dataset: ArrayD<f32>,
    batch_size: usize,
    shuffle: bool,
    data_size: usize,
    indices: Vec<usize>,
}

impl Iterator for DataLoader {
    fn next(&mut self) -> Option<Self::Item> {}
}

impl DataLoader {
    fn new(datasets: ArrayD<f32>, batch_size: usize, shuffle: bool) -> Self {
        let x_len = datasets.view().shape()[0];
        let mut indices: Vec<usize>;
        if shuffle == true {
            indices = (0..x_len).collect();
            indices.shuffle(&mut rand::thread_rng());
        } else {
            indices = (0..x_len).collect();
        }

        Self {
            dataset: datasets,
            batch_size: batch_size,
            shuffle: shuffle,
            data_size: x_len,
            indices: indices,
        }
    }
}
