use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;

use ndarray::{
    array, s, Array, Array1, Array3, Array4, ArrayBase, ArrayD, ArrayView4, ArrayViewD, Dimension,
    IxDyn, OwnedRepr, Shape,
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
) -> ArrayD<f32> {
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

    //im2colの出力となる行列を初期化。
    let mut cols = Array3::<f32>::zeros((n, c * kh * kw, oh * ow));

    //im2colの処理
    for b in 0..n {
        let img = input.slice(s![b, .., .., ..]);
        let mut col = cols.slice_mut(s![b, .., ..]);
        let mut col_idx = 0;

        for y in 0..oh {
            for x in 0..ow {
                let y_start = y as isize * stride_h as isize - pad_h as isize;
                let x_start = x as isize * stride_w as isize - pad_w as isize;

                let mut patch = Vec::<f32>::with_capacity(c * kh * kw);

                for c_idx in 0..c {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;

                            // paddingしたところは0にする。
                            let value = if in_y >= 0
                                && (in_y as usize) < h
                                && in_x >= 0
                                && (in_x as usize) < w
                            {
                                img[(c_idx, in_y as usize, in_x as usize)]
                            } else {
                                0.0
                            };
                            patch.push(value);
                        }
                    }
                }
                for (i, v) in patch.into_iter().enumerate() {
                    col[(i, col_idx)] = v;
                }
                col_idx += 1;
            }
        }
    }
    //weightを1列に展開し、並べて2次元の行列に変形させる。
    let weights_2d = weight.into_shape_with_order((oc, c * kh * kw)).unwrap();

    // Wx = (oc,c*kh*kw) × (n,c*kh*kw,oh*ow) -> (n,oc,oh*ow)      (oc,c*kh*kw) × (c*kh*kw,oh*ow)の行列積をバッチn個として計算する。
    let mut out = Array3::<f32>::zeros((n, oc, oh * ow));

    for b in 0..n {
        let col = cols.slice(s![b, .., ..]);
        let result = weights_2d.dot(&col);
        out.slice_mut(s![b, .., ..]).assign(&result);
    }

    let mut out4d = Array4::<f32>::zeros((n, oc, oh, ow));

    for b in 0..n {
        for co in 0..oc {
            for idx in 0..(oh * ow) {
                let y = idx / ow;
                let x = idx % ow;
                out4d[(b, co, y, x)] = out[(b, co, idx)];
            }
        }
    }
    out4d.into_dyn()
}
