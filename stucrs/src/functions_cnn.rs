use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;

use ndarray::{
    array, s, Array, Array3, ArrayBase, ArrayD, ArrayView4, ArrayViewD, Dimension, IxDyn,
    OwnedRepr, Shape,
};
use std::rc::{Rc, Weak};
use std::{usize, vec};

use crate::config::{get_grad_status, id_generator, set_grad_false, set_grad_true};
use crate::functions_new::*;

pub fn get_conv_outsize(
    input_size: (usize, usize),
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> (usize, usize) {
    let oh = (input_size.0 + pad_size.0 * 2 - kernel_size.0) / stride_size.0 + 1;
    let ow = (input_size.1 + pad_size.1 * 2 - kernel_size.1) / stride_size.1 + 1;

    (oh, ow)
}

pub fn conv2d_array(
    input: ArrayView4<f32>,
    weight: ArrayView4<f32>,
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    // inputから形状のデータを取り出す。
    let n = input_shape[0]; //バッチ数
    let c = input_shape[1]; //チャンネル数
    let h = input_shape[2]; //縦
    let w = input_shape[3]; //横

    // weightから形状のデータを取り出す。
    let oc = weight_shape[0];
    let c_wt = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    // チャンネル数がinputとweightで一致しているか確認。
    if c != c_wt {
        panic!("Conv2d: inputのチャンネル数とweightのチャンネル数が一致しません。");
    }

    let (stride_h, stride_w) = stride_size;
    let (pad_h, pad_w) = pad_size;

    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

    let mut cols = Array3::<f32>::zeros((n, c * kh * kw, oh * ow));

    for b in 0..n {
        let img = input.slice(s![b, .., .., ..]);
        let mut col = cols.slice_mut(s![b, .., ..]);
        let mut col_idx = 0;

        for y in 0..oh {
            let y_first = y as isize * stride_h as isize - pad_h as isize;
        }
    }
}
