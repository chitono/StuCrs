use core::panic;

//use std::clone;

//use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::*;

use ndarray_stats::QuantileExt;

//use std::thread;
//use std::time::Duration;

use crate::config::get_test_flag_status;
use crate::core_new::*;
use crate::datasets::{arr1d_to_one_hot, tensor2d_to_one_hot};
use crate::error::FrameResult;
use crate::functions::matrix::matmul;
use crate::tensor::lib::{Tensor, TensorOps};

pub fn linear_simple(
    x: &RcVariable,
    w: &RcVariable,
    b: &Option<RcVariable>,
) -> FrameResult<RcVariable> {
    let t = matmul(&x, &w)?;

    let y;

    if let Some(b_rc) = b {
        y = t + b_rc.clone();
    } else {
        y = t;
    }

    Ok(y)
}

pub fn dropout(x: &RcVariable, ratio: f32) -> FrameResult<RcVariable> {
    if get_test_flag_status() == false {
        let random_tensor = Tensor::standard_normal(x.data().shape().dims.clone())?;
        //let random_array: Array<f32, IxDyn> = Array::random(x.data().shape(), Standard);
        /*
        let mask = random_tensor
            .mapv(|x| if x > ratio { 1.0f32 } else { 0.0f32 })
            .rv();
        */
        let mask = random_tensor.max_mask(ratio)?.rv();
        let scale = (1.0f32 - ratio).rv();

        let y = x.clone() * mask / scale;
        Ok(y)
    } else {
        Ok(x.clone())
    }
}

pub fn accuracy(y: ArrayView2<f32>, t: ArrayView2<f32>) -> FrameResult<f32> {
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

    Ok(accuracy)
}

pub fn tensor_accuracy(y: &Tensor, t: &Tensor) -> FrameResult<f32> {
    if y.shape() != t.shape() {
        panic!("交差エントロピー誤差でのxとtの形状が異なります。tがone-hotベクトルでない可能性があります。")
    }
    let data_size = y.shape().dims()[0] as f32;
    let num_class = t.shape().dims()[1];

    // 二つのone_hotの行列を書けることで正しく解けた個数を求めている
    // TODO: もっと効率の良い処理に変更したい。
    let argmax_tensor = y.argmax_axis(1)?;

    let one_hot_y = tensor2d_to_one_hot(argmax_tensor, num_class)?;

    assert_eq!(one_hot_y.shape(), t.shape());

    let acc_matrix = (one_hot_y * t.clone())?;

    let accuracy = acc_matrix.sum(None, false)?.to_vec()?[0] / data_size;

    Ok(accuracy)
}
