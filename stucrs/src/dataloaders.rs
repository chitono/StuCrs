use ndarray::*;
use rand;
use rand::seq::{index, SliceRandom};

use crate::core_new::{ArrayDToRcVariable, RcVariable};


#[derive(Clone)]
pub struct DataLoader {
    x_data_set: ArrayD<f32>,
    y_data_set: ArrayD<f32>,
    batch_size: usize,
    data_size: usize,
    index: Vec<usize>,
    current_count: usize,
    shuffle: bool,
}

impl Iterator for DataLoader {
    type Item = (RcVariable, RcVariable);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_count >= self.index.len() {
            return None;
        }

        let batch_end_index = (self.current_count + self.batch_size).min(self.index.len());

        let chunk_index = &self.index[self.current_count..batch_end_index];

        self.current_count = batch_end_index;

        let x_batch = self
            .x_data_set
            .select(Axis(0), &chunk_index)
            .to_owned()
            .rv();
        let y_batch = self
            .y_data_set
            .select(Axis(0), &chunk_index)
            .to_owned()
            .rv();

        Some((x_batch, y_batch))
    }
}

// assert_eq!などで不正な行列の入力を防ぐを実装したい

impl DataLoader {
    pub fn new(
        x_data_set: ArrayD<f32>,
        y_data_set: ArrayD<f32>,
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        assert_eq!(
            x_data_set.shape()[0],
            y_data_set.shape()[0],
            "x_data_setとy_data_setのデータ数が異なります。"
        );
        let data_size = x_data_set.view().shape()[0];
        //let max_count = ((data_size / batch_size) as f32).ceil();
        let mut index: Vec<usize>;
        if shuffle == true {
            index = (0..data_size).collect();
            index.shuffle(&mut rand::thread_rng());
        } else {
            index = (0..data_size).collect();
        }

        Self {
            x_data_set: x_data_set,
            y_data_set: y_data_set,
            batch_size: batch_size,
            data_size: data_size,
            index: index,
            current_count: 0,
            shuffle: shuffle,
        }
    }
}
