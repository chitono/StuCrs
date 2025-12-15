use core::panic;

//use std::clone;

//use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::*;
use ndarray_rand::rand_distr::Standard;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;

use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::config::get_test_flag_status;
use crate::core_new::*;
use crate::datasets::arr1d_to_one_hot;

pub fn dropout(x: &RcVariable, ratio: f32) -> RcVariable {
    if get_test_flag_status() == false {
        let random_array: Array<f32, IxDyn> = Array::random(x.data().shape(), Standard);
        let mask = random_array
            .mapv(|x| if x > ratio { 1.0f32 } else { 0.0f32 })
            .rv();
        let scale = array![1.0f32 - ratio].rv();
        let y = x.clone() * mask / scale;
        y
    } else {
        x.clone()
    }
}

pub fn accuracy(y: ArrayView2<f32>, t: ArrayView2<f32>) -> f32 {
    if y.shape() != t.shape() {
        panic!("交差エントロピー誤差でのxとtの形状が異なります。tがone-hotベクトルでない可能性があります。")
    }
    let data_size = y.shape()[0] as f32;
    let num_class = t.shape()[1];
    let argmax_vec: Vec<u32> = y
        .outer_iter()
        .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| row.argmax().unwrap() as u32)
        .collect();
    let max_index = Array::from_vec(argmax_vec);
    let one_hot_y = arr1d_to_one_hot(max_index.view(), num_class);

    assert_eq!(one_hot_y.shape(), t.shape());

    let acc_matrix = &one_hot_y * &t;

    let accuracy = acc_matrix.sum() / data_size;

    accuracy
}
