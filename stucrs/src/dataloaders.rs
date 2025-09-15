use ndarray::*;
use rand;
use rand::seq::{index, SliceRandom};

use crate::core_new::{ArrayDToRcVariable, RcVariable};

struct DataLoader {
    x_data_set: ArrayD<f32>,
    t_data_set: ArrayD<f32>,
    batch_size: usize,
    shuffle: bool,
    data_size: usize,
    index: Vec<usize>,
    count: usize,
    max_count: usize,
}

/*

impl Iterator for DataLoader {
    type Item = (RcVariable, RcVariable);
    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.max_count {
            None
        } else {
            let i = self.count;
            for chunk_indices in self.index.chunks(self.batch_size) {
                let x_batch = self.x_data_set.select(Axis(0), chunk_indices).rv();
                let t_batch = self.t_data_set.select(Axis(0), chunk_indices).rv();
            }
        }
    }
}

*/
impl DataLoader {
    fn new(
        x_data_set: ArrayD<f32>,
        t_data_set: ArrayD<f32>,
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        let x_len = x_data_set.view().shape()[0];
        let max_count = ((x_len / batch_size) as f32).ceil();
        let mut index: Vec<usize>;
        if shuffle == true {
            index = (0..x_len).collect();
            index.shuffle(&mut rand::thread_rng());
        } else {
            index = (0..x_len).collect();
        }

        Self {
            x_data_set: x_data_set,
            t_data_set: t_data_set,
            batch_size: batch_size,
            shuffle: shuffle,
            data_size: x_len,
            index: index,
            count: 0,
            max_count: max_count as usize,
        }
    }
}
