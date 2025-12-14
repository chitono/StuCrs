use core::f32;
use std::cell::RefCell;
//use std::clone;
//use std::collections::HashSet;
use std::fmt::Debug;

use ndarray::*;
use ndarray_stats::QuantileExt;
use std::rc::{Rc, Weak};
use std::{usize, vec};

use crate::config::{get_grad_status, id_generator};
use crate::core_new::{ArrayDToRcVariable, Function, RcVariable, Variable};
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

pub fn conv2d_simple(
    input: &RcVariable,
    weight: &RcVariable,
    _bias: Option<RcVariable>,
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let input_data = input.data();
    let weight_data = weight.data();

    let input_shape = input_data.shape();
    let weight_shape = weight_data.shape();

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    // weightから形状のデータを取り出す。
    let oc = weight_shape[0];
    let c_wt = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    // チャンネル数がinputとweightで一致しているか確認。
    if c != c_wt {
        panic!("Conv2d: inputのチャンネル数とweightのチャンネル数が一致しません。");
    }

    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), stride_size, pad_size);

    let cols = im2col_simple(input, (kh, kw), stride_size, pad_size);

    let weights_2d = weight.reshape(IxDyn(&[oc, c * kh * kw]));

    let out = tensordot(&weights_2d, &cols);

    let out4d = out.reshape(IxDyn(&[n, oc, oh, ow]));

    out4d
}

pub fn max_pool2d_simple(
    input: &RcVariable,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let input_data = input.data();

    let input_shape = input_data.shape();

    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    let (kh, kw) = kernel_size;

    let (oh, ow) = get_conv_outsize((h, w), kernel_size, stride_size, pad_size);

    let cols = im2col_simple(input, kernel_size, stride_size, pad_size);

    let cols = permute_axes(&cols, vec![0, 2, 1]);

    let cols = cols.reshape(IxDyn(&[n, c * oh * ow, kh * kw]));

    let y = max(&cols, Some(2));

    let output = y
        .reshape(IxDyn(&[n, oh, ow, c]))
        .permute_axes(vec![0, 3, 1, 2]);

    output
}

pub fn conv2d_array(
    input: ArrayView4<f32>,
    weight: ArrayView4<f32>,
    _bias: Option<Array1<f32>>,
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

    // im2col関数でcolsを出力。
    let cols = im2col_array(input.view(), (kh, kw), stride_size, pad_size);

    //weightを1列に展開し、並べて2次元の行列に変形させる。
    let weights_2d = weight.into_shape_with_order((oc, c * kh * kw)).unwrap();

    // Wx = (oc,c*kh*kw) × (n,c*kh*kw,oh*ow) -> (n,oc,oh*ow)      (oc,c*kh*kw) × (c*kh*kw,oh*ow)の行列積をバッチn個として計算する。
    let mut out = Array3::<f32>::zeros((n, oc, oh * ow));

    for b in 0..n {
        let col = cols.slice(s![b, .., ..]);
        let result = weights_2d.dot(&col);
        out.slice_mut(s![b, .., ..]).assign(&result);
    }

    //let mut out4d = Array4::<f32>::zeros((n, oc, oh, ow));
    let out4d = out.into_shape_with_order((n, oc, oh, ow)).unwrap();
    //(N,OC,OH*OW) -> (N,OC,OH,OW)に変換

    /*
    for b in 0..n {
        for co in 0..oc {
            for idx in 0..(oh * ow) {
                let y = idx / ow;
                let x = idx % ow;
                out4d[(b, co, y, x)] = out[(b, co, idx)];
            }
        }
    } */
    out4d.into_dyn()
}

pub fn max_pool2d_array(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> ArrayD<f32> {
    let input_shape = input.shape();
    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];
    let (kh, kw) = kernel_size;

    let (oh, ow) = get_conv_outsize((h, w), kernel_size, stride_size, pad_size);

    let cols = im2col_array(input, kernel_size, stride_size, pad_size);

    // (N,c*kh*kw,oh*ow) -> (N,oh*ow,c*kh*kw)
    let cols = cols.permuted_axes([0, 2, 1]);

    let cols = cols
        .to_owned()
        .into_shape_clone((n, c * oh * ow, kh * kw))
        .unwrap();
    let mut out = Array2::<f32>::zeros((n, c * oh * ow));
    for b in 0..n {
        let rows = cols.slice(s![b, .., ..]);
        let max: Array1<f32> = rows
            .outer_iter()
            .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| row.max().unwrap().clone())
            .collect();

        out.slice_mut(s![b, ..]).assign(&max);
    }

    let max_cols = out.into_shape_with_order((n, oh, ow, c)).unwrap();
    let output = max_cols.permuted_axes([0, 3, 1, 2]);

    output.into_dyn()
}

#[derive(Debug, Clone)]
struct Im2col {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    generation: i32,
    id: usize,
}

impl Function for Im2col {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Im2colは一変数関数です。inputsの個数が一つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[RcVariable]) -> RcVariable {
        let x = &xs[0];
        //ここは後に動的のままim2colに渡す予定。
        let x_data = x.data().into_dimensionality::<Ix4>().unwrap();

        let y_data = im2col_array(
            x_data.view(),
            self.kernel_size,
            self.stride_size,
            self.pad_size,
        );

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x_data = &self.inputs[0].data();
        let x_shape = x_data.shape();
        let x_shape = x_shape
            .try_into()
            .expect("Im2colのxの次元が4ではありません。");

        let gx = col2im_simple(
            gy,
            x_shape,
            self.kernel_size,
            self.stride_size,
            self.pad_size,
        );
        let gxs = vec![gx];

        gxs
    }

    fn get_inputs(&self) -> &[RcVariable] {
        &self.inputs
    }

    fn get_output(&self) -> RcVariable {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        RcVariable(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
}
impl Im2col {
    fn new(
        inputs: &[RcVariable],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn im2col_f(
    xs: &[RcVariable],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    Im2col::new(xs, kernel_size, stride_size, pad_size)
        .borrow_mut()
        .call()
}

pub fn im2col_simple(
    x: &RcVariable,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let y = im2col_f(&[x.clone()], kernel_size, stride_size, pad_size);
    y
}

#[derive(Debug, Clone)]
struct Col2im {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    input_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    generation: i32,
    id: usize,
}

impl Function for Col2im {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Col2imは一変数関数です。inputsの個数が一つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[RcVariable]) -> RcVariable {
        let x = &xs[0];
        //ここは後に動的のままcol2imに渡す予定。
        let x_data = x.data().into_dimensionality::<Ix3>().unwrap();

        let y_data = col2im_array(
            x_data.view(),
            self.input_shape,
            self.kernel_size,
            self.stride_size,
            self.pad_size,
        );

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let gx = im2col_simple(gy, self.kernel_size, self.stride_size, self.pad_size);
        let gxs = vec![gx];

        gxs
    }

    fn get_inputs(&self) -> &[RcVariable] {
        &self.inputs
    }

    fn get_output(&self) -> RcVariable {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        RcVariable(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
}
impl Col2im {
    fn new(
        inputs: &[RcVariable],
        input_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            input_shape: input_shape,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn col2im_f(
    xs: &[RcVariable],
    input_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    Col2im::new(xs, input_shape, kernel_size, stride_size, pad_size)
        .borrow_mut()
        .call()
}

pub fn col2im_simple(
    x: &RcVariable,
    input_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> RcVariable {
    let y = col2im_f(
        &[x.clone()],
        input_shape,
        kernel_size,
        stride_size,
        pad_size,
    );
    y
}

/// imageからcolsに変換するndarray関数。
/// inputには画像データ(4次元のndarray行列)を渡す。
pub fn im2col_array(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> Array3<f32> {
    let input_shape = input.shape();

    // inputから形状のデータを取り出す。
    let n = input_shape[0]; //バッチ数
    let c = input_shape[1]; //チャンネル数
    let h = input_shape[2]; //縦
    let w = input_shape[3]; //横

    let (kh, kw) = kernel_size;
    let (stride_h, stride_w) = stride_size;
    let (pad_h, pad_w) = pad_size;

    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

    let mut cols = Array3::<f32>::zeros((n, c * kh * kw, oh * ow));

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
    cols
}

/// colsからimageに変更する関数。
/// inputにはcolsを渡す。
/// im_shapeは元のimageのshape(N,C,H,W)を渡す。
pub fn col2im_array(
    input: ArrayView3<f32>,
    im_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> Array4<f32> {
    let (kh, kw) = kernel_size;
    let (stride_h, stride_w) = stride_size;
    let (pad_h, pad_w) = pad_size;

    let (n, c, h, w) = (im_shape[0], im_shape[1], im_shape[2], im_shape[3]); //元のimageの形状を取得。
    let (oh, ow) = get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

    let mut imgs = Array4::<f32>::zeros((n, c, h, w));

    for b in 0..n {
        let col = input.slice(s![b, .., ..]);
        let mut img = imgs.slice_mut(s![b, .., .., ..]);
        let mut col_idx = 0;

        for y in 0..oh {
            for x in 0..ow {
                let y_start = y as isize * stride_h as isize - pad_h as isize;
                let x_start = x as isize * stride_w as isize - pad_w as isize;

                let mut patch_row_idx = 0;

                for c_idx in 0..c {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;

                            // paddingしていないところか判定。
                            if in_y >= 0 && (in_y as usize) < h && in_x >= 0 && (in_x as usize) < w
                            {
                                let value = col[(patch_row_idx, col_idx)];
                                // imgの対応するところに加算する。
                                img[(c_idx, in_y as usize, in_x as usize)] += value;
                            }
                            patch_row_idx += 1;
                        }
                    }
                }

                col_idx += 1;
            }
        }
    }
    imgs
}
