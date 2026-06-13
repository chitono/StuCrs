//! # StuCrs
//!
//! A deep learning framework implemented in pure Rust from scratch to explore and understand its underlying principles.
//!
//! ## Example
//!
//! ```rust
//! use rand::seq::SliceRandom;
//! use rand::*;
//! use std::time::Instant;
//! use stucrs::config;
//! use stucrs::core::TensorToRcVariable;
//! use stucrs::datasets::{tensor2d_to_one_hot, MNIST};
//! use stucrs::error::FrameResult;
//! use stucrs::functions::loss::softmax_cross_entropy_simple;
//! use stucrs::functions::neural_funcs::tensor_accuracy;
//! use stucrs::layers::{Activation, Dense, Linear};
//! use stucrs::models::{BaseModel, Model};
//! use stucrs::optimizers::{Optimizer, SGD};
//! use stucrs::tensor::ops::TensorOps;
//!
//! fn main() -> FrameResult<()> {
//!    let mnist = MNIST::new()?;
//!    let x_train = mnist.train_img;
//!    let y_train = mnist.train_label;
//!    let x_test = mnist.test_img;
//!    let y_test = mnist.test_label;
//!
//!    println!("backend = {:?}", x_train.backend_type());
//!
//!    let _image_num = 0;
//!
//!    let x_train = x_train.reshape(vec![50000, 28 * 28])?;
//!    let x_test = x_test.reshape(vec![10000, 28 * 28])?;
//!
//!    let y_train = tensor2d_to_one_hot(y_train, 10)?;
//!    let y_test = tensor2d_to_one_hot(y_test, 10)?;
//!
//!    let max_epoch = 5;
//!    let lr = 0.01;
//!    let batch_size = 100;
//!
//!    let data_size = x_train.shape().dims()[0];
//!    println!("data_size={}", data_size);
//!
//!    let mut model = BaseModel::new();
//!    model.stack(Dense::new(1000, true, None, Activation::Relu)?);
//!    model.stack(Dense::new(1000, true, None, Activation::Relu)?);
//!    model.stack(Linear::new(10, false, None)?);
//!
//!    let mut optimizer = SGD::new(lr);
//!    optimizer.setup(&model);
//!    let start = Instant::now();
//!    for epoch in 0..max_epoch {
//!        let mut indices: Vec<usize> = (0..data_size).collect();
//!        let mut rng = thread_rng();
//!        indices.shuffle(&mut rng);
//!
//!        let mut sum_loss = 0.0f32;
//!        let mut sum_acc = 0.0f32;
//!
//!        for chunk_indices in indices.chunks(batch_size) {
//!            let x_batch = x_train.axis_slice(0, chunk_indices)?.rv();
//!            let y_batch = y_train.axis_slice(0, chunk_indices)?.rv();
//!
//!            let y = model.call(&x_batch)?;
//!            let mut loss = softmax_cross_entropy_simple(&y, &y_batch)?;
//!
//!            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;
//!            model.cleargrad();
//!
//!            loss.backward(false)?;
//!
//!            optimizer.update()?;
//!
//!            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);
//!
//!            sum_loss = sum_loss + epoch_loss;
//!            sum_acc = sum_acc + acc * (y_batch.len() as f32);
//!        }
//!
//!        let average_loss = sum_loss / (data_size as f32);
//!        let average_acc = sum_acc / (data_size as f32);
//!
//!        println!(
//!            "epoch = {:?}, train_loss = {:?}, accuracy = {}",
//!            epoch + 1,
//!            average_loss,
//!            average_acc
//!        );
//!
//!        //推論
//!        config::set_grad_false();
//!        let test_data_size = x_test.shape().dims()[0];
//!        let mut indices: Vec<usize> = (0..test_data_size).collect();
//!        let mut rng = thread_rng();
//!        indices.shuffle(&mut rng);
//!
//!        let mut sum_loss = 0.0f32;
//!        let mut sum_acc = 0.0f32;
//!
//!        for chunk_indices in indices.chunks(batch_size) {
//!            let x_batch = x_test.axis_slice(0, chunk_indices)?.rv();
//!            let y_batch = y_test.axis_slice(0, chunk_indices)?.rv();
//!
//!            let y = model.call(&x_batch)?;
//!            let loss = softmax_cross_entropy_simple(&y, &y_batch)?;
//!            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;
//!
//!            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);
//!
//!            sum_loss = &sum_loss + &epoch_loss;
//!            sum_acc = sum_acc + acc * (y_batch.len() as f32);
//!        }
//!
//!        let average_loss = sum_loss / (test_data_size as f32);
//!        let average_acc = sum_acc / (test_data_size as f32);
//!
//!        println!(
//!            "epoch = {:?}, test_loss = {:?}, test_accuracy = {}",
//!            epoch + 1,
//!            average_loss,
//!            average_acc
//!        );
//!
//!        config::set_grad_true();
//!    }
//!    let end = Instant::now();
//!    let duration = end.duration_since(start);
//!    println!("処理時間{:?}", duration);
//!
//!    Ok(())
//!}
//!
//!
//!
//! ```
//!

//use ndarray::{array, ArrayBase, Dimension, OwnedRepr};

use core::RcVariable;
use core::{add, div, mul, neg, sub};
use std::ops::{Add, Div, Mul, Neg, Sub};
//use std::sync::atomic::{AtomicU32, Ordering};

//演算子のオーバーロード

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        if let Ok(add_y) = add(&[self.clone(), rhs.clone()]) {
            return add_y;
        } else {
            panic!("add演算子でエラーが発生しました。")
        }
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        if let Ok(mul_y) = mul(&[self.clone(), rhs.clone()]) {
            return mul_y;
        } else {
            panic!("mul演算子でエラーが発生しました。")
        }
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        if let Ok(sub_y) = sub(&[self.clone(), rhs.clone()]) {
            return sub_y;
        } else {
            panic!("sub演算子でエラーが発生しました。")
        }
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        match div(&[self.clone(), rhs.clone()]) {
            Ok(div_y) => div_y,
            Err(e) => {
                panic!("{:?}", e);
            }
        }
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        if let Ok(neg_y) = neg(&[self.clone()]) {
            return neg_y;
        } else {
            panic!("neg演算子でエラーが発生しました。")
        }
    }
}
pub mod config;
pub mod core;
//pub mod dataloaders;
pub mod datasets;
pub mod error;
pub mod functions;
pub mod functions_cnn;
pub mod layers;
pub mod models;
pub mod optimizers;
pub mod tensor;

#[cfg(test)]
mod tests {

    use std::time::Instant;

    use ndarray::array;

    use crate::{
        core::TensorToRcVariable,
        error::FrameResult,
        functions::{
            activation_funcs::relu,
            math::{clamp, cos, exp, log, max, sin, square, tanh},
            matrix::{matmul, permute_axes, reshape, sum, tensordot, transpose},
        },
        functions_cnn::{
            col2im_simple, conv2d_array, conv2d_simple, im2col_simple, max_pool2d_simple,
        },
        models::Model,
        tensor::{lib::TensorOps, shape::Shape, tensor::Tensor},
    };

    #[test]
    fn add_test() -> FrameResult<()> {
        let a = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3])?.rv();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3])?.rv();

        let mut c = a.clone() + b.clone();

        println!("c = {}", c.data());

        c.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data());
        println!("b_grad = {:?}", b.grad().unwrap().data());
        Ok(())
    }

    #[test]
    fn add_with_broadcast_test() -> FrameResult<()> {
        let a = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3])?.rv();
        let b = Tensor::from_vec(vec![2.0], vec![1])?.rv();

        let mut c = a.clone() + b.clone();

        println!("c = {}", c.data()); // [3,3,3,3,3]

        c.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // [1.0,1.0,1.0]
        println!("b_grad = {:?}", b.grad().unwrap().data()); // [3.0]

        Ok(())
    }

    #[test]
    fn mul_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3]).unwrap().rv();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]).unwrap().rv();

        let c = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap().rv();

        let mut y = (a.clone() * b.clone()) + c.clone();

        println!("c = {}", y.data()); // 7.0

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 2.0
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 3.0
        Ok(())
    }

    #[test]
    fn sub_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3]).unwrap().rv();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]).unwrap().rv();

        let c = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]).unwrap().rv();

        let mut y = (a.clone() * b.clone()) - c.clone();

        println!("y = {}", y.data());

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data());
        println!("b_grad = {:?}", b.grad().unwrap().data());
        println!("c_grad = {:?}", c.grad().unwrap().data());

        Ok(())
    }

    #[test]
    fn div_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3]).unwrap().rv();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3]).unwrap().rv();

        let mut y = a.clone() / b.clone();

        println!("y = {}", y.data()); // 1.5

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 0.5
        println!("b_grad = {:?}", b.grad().unwrap().data()); // -0.75
        Ok(())
    }

    #[test]
    fn back_clear_test() -> FrameResult<()> {
        let mut a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3])?.rv();

        let mut c = a.clone() + a.clone();

        println!("c = {}", c.data());

        c.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data());

        a.cleargrad();

        let mut d = a.clone() + a.clone() + a.clone();

        d.backward(false)?;

        println!("a_grad2 = {}", a.grad().unwrap().data());
        //println!("b_grad = {:?}", b.grad().unwrap().data());
        Ok(())
    }

    #[test]
    fn pow_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3]).unwrap().rv();

        let mut y = a.clone().pow(2.0)?;

        println!("y = {}", y.data()); // 9.0

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
        Ok(())
    }

    #[test]
    fn square_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3]).unwrap().rv();

        let mut y = square(&a)?;

        println!("y = {}", y.data()); // 9.0

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
        Ok(())
    }

    #[test]
    fn exp_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();
        let b = Tensor::from_vec(vec![2.0, 2.0, 2.0], vec![3])?.rv();

        let mut y0 = exp(&a)?;
        let mut y1 = b.clone().exp()?;

        println!("y0= {}", y0.data()); // 20.0855...
        println!("y1= {}", y1.data()); // 7.3890...

        y0.backward(false)?;
        y1.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 20.0855
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 7.3890
        Ok(())
    }

    #[test]
    fn sin_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();

        let mut y = sin(&a)?;

        println!("y = {}", y.data()); // 0.1411

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // -0.9899
        Ok(())
    }

    #[test]
    fn cos_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();

        let mut y = cos(&a)?;

        println!("y = {}", y.data()); // -0.9899

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // -0.1411
        Ok(())
    }

    #[test]
    fn tanh_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();

        let mut y = tanh(&a)?;

        println!("y = {}", y.data()); // 0.9950...

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 9.866...e-3
        Ok(())
    }

    #[test]
    fn log_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();
        let b = Tensor::from_vec(vec![3.0, 3.0, 3.0], vec![3])?.rv();

        let mut y0 = log(&a, None)?; //底がe
        let mut y1 = log(&b, Some(2.0))?; //底が2.0

        println!("y0 = {}", y0.data()); // 1.098...
        println!("y1 = {}", y1.data()); // 1.584...

        y0.backward(false)?;
        y1.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 0.3333...
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 0.4808...

        Ok(())
    }

    #[test]
    fn reshape_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let mut y0 = reshape(&a, &Shape::new(vec![1, 6])?)?;

        let mut y1 = b.reshape(&Shape::new(vec![1, 6])?)?;

        println!("y0 = {}", y0.data()); //[[1,2,3,4,5,6]] shape(1,6)
        println!("y1 = {}", y1.data()); //[[1,2,3,4,5,6]] shape(1,6)

        y0.backward(false)?;
        y1.backward(false)?;
        println!("a_grad = {:?}", a.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
        println!("b_grad = {:?}", b.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
        Ok(())
    }

    #[test]
    fn transpose_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let mut y0 = transpose(&a)?;

        let mut y1 = b.t()?;

        println!("y0 = {}", y0.data()); //[[1,4],[2,5],[3,6]] shape(3,2)
        println!("y1 = {}", y1.data()); //[[1,4],[2,5],[3,6]] shape(3,2)

        y0.backward(false)?;
        y1.backward(false)?;
        println!("a_grad = {:?}", a.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
        println!("b_grad = {:?}", b.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)

        Ok(())
    }

    #[test]
    fn sum_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();
        let c = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let mut y0 = sum(&a, None)?;
        let mut y1 = sum(&b, Some(0))?;
        let mut y2 = sum(&c, Some(1))?;

        println!("y0 = {}", y0.data()); // 21.0
        println!("y1 = {}", y1.data()); //
        println!("y2 = {}", y2.data()); //

        y0.backward(false)?;
        y1.backward(false)?;
        y2.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); //
        println!("b_grad = {:?}", b.grad().unwrap().data()); //
        println!("c_grad = {:?}", c.grad().unwrap().data()); //

        Ok(())
    }

    #[test]
    fn broadcast_to_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let mut y = a.clone().pow(2.0)?;

        println!("y = {}", y.data()); // 9.0

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
        Ok(())
    }

    #[test]
    fn sum_to_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let mut y = a.clone().pow(2.0)?;

        println!("y = {}", y.data()); // 9.0

        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
        Ok(())
    }

    #[test]
    fn matmul_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?.rv();

        let mut y = matmul(&a, &b)?;

        println!("y = {}", y.data()); // [[58,64],[139,154]]
        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data());
        println!("b_grad = {:?}", b.grad().unwrap().data());

        Ok(())
    }

    #[test]
    fn tensordot_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3])?.rv();

        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?.rv();

        println!("a_shape = {:?}", a.data().shape()); //[1,2,3]

        println!("b_shape = {:?}", b.data().shape()); //[3,2]

        let mut y = tensordot(&a, &b)?;

        println!("y = {:?}", y.data()); //[[[58.0, 64.0],[139.0, 154.0]]]
        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); //[[[15.0, 19.0, 23.0],[15.0, 19.0, 23.0]]]
        println!("b_grad = {:?}", b.grad().unwrap().data()); //[[5.0, 5.0],[7.0, 7.0],[9.0, 9.0]]

        Ok(())
    }

    #[test]
    fn permute_axes2d_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let mut y = permute_axes(&a, vec![1, 0])?;

        println!("y = {}", y.data());
        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data());
        Ok(())
    }

    #[test]
    fn permute_axes3d_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3])?.rv();

        println!("a_shape = {:?}", a.data().shape()); //[1,2,3]

        let mut y = permute_axes(&a, vec![1, 2, 0])?;

        println!("y = {:?}", y.data());
        y.backward(false)?;

        println!("a_grad = {:?}", a.grad().unwrap().data()); //[1,2,3] a_shapeと同じならok

        Ok(())
    }

    #[test]
    fn max_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![1.0, 2.0, 30.0, 4.0, 5.0, 6.0], vec![6])?.rv();
        let b = Tensor::from_vec(vec![1.0, 5.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?.rv();

        let c = Tensor::from_vec(
            vec![
                1.0, 3.0, 2.0, 6.0, 5.0, 4.0, 10.0, 20.0, 30.0, 60.0, 50.0, 40.0,
            ],
            vec![2, 2, 3],
        )?
        .rv(); // 3次元

        let mut y1 = max(&b, Some(1))?;
        println!("計算2完了");
        let mut y2 = max(&c, Some(2))?;
        println!("計3完了");

        println!("y1 = {}", y1.data()); // [[5],[6]]
        println!("y2 = {}", y2.data()); //

        y1.backward(false)?;
        println!("微分2完了");
        y2.backward(false)?;
        println!("微分3完了");

        println!("b_grad = {}", b.grad().unwrap().data()); //[[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]
        println!("c_grad = {}", c.grad().unwrap().data()); //[[[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]],[[0.0, 0.0, 1.0],[1.0, 0.0, 0.0]]]

        Ok(())
    }

    #[test]
    fn clamp_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![0.0, -1.0, 3.0, 1.0, -4.0], vec![5])?.rv();

        println!("a Backend = {:?}", a.data().backend_type());

        let mut y = clamp(&a, 1.0e-4, 1.0)?;

        println!("y = {}", y.data()); // [0.0001, 0.0001, 1.0000, 1.0000, 0.0001]
        println!("y Backend = {:?}", y.data().backend_type());

        y.backward(false)?;

        println!("a_grad = {}", a.grad().unwrap().data()); // [0.0, 0.0, 0.0, 1.0, 0.0]
        Ok(())
    }

    #[test]
    fn relu_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let a = Tensor::from_vec(vec![0.0, -1.0, 3.0, 1.0, -4.0], vec![5])?.rv();

        let mut y = relu(&a)?;

        println!("y = {}", y.data()); // [0.0000, 0.0000, 3.0000, 1.0000, 0.0000]

        y.backward(false)?;

        println!("a_grad = {}", a.grad().unwrap().data()); // [0.0, 0.0, 1.0, 1.0, 0.0]
        Ok(())
    }

    #[test]
    fn get_conv_outsize_test() {
        use crate::functions_cnn::get_conv_outsize;

        let input_size = (4, 4);
        let kernel_size = (3, 3);
        let stride_size = (1, 1);
        let pad_size = (1, 1);

        let output_size = get_conv_outsize(input_size, kernel_size, stride_size, pad_size);

        assert_eq!(output_size, (4, 4));
    }

    #[test]
    fn dim_test() {
        let input = array![[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]];
        let input_2 = array![[[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]];

        assert_eq!(input.ndim(), 4);
        assert_eq!(input_2.ndim(), 4);
    }

    #[test]
    fn conv2d_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let input_tensor = Tensor::ones(vec![2, 5, 15, 15])?;
        let weight_tensor = Tensor::ones(vec![8, 5, 3, 3])?;

        let input = input_tensor.rv();
        let weight = weight_tensor.rv();

        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let mut output = conv2d_simple(&input, &weight, None, stride_size, pad_size)?;

        println!("output_shape = {:?}", output.data().shape()); //shape = (1,8,15,15)

        output.backward(false)?;

        println!(
            "input_grad_shape = {:?}",
            input.grad().unwrap().data().to_vec()?
        ); //shape = (1,5,15,15)

        Ok(())
    }

    #[test]
    fn conv2d_array_1ch_test() -> FrameResult<()> {
        let _input = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        )?;
        let _kernel = Tensor::from_vec(
            vec![1.0f32, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            vec![1, 1, 3, 3],
        )?;
        //let input = array![[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
        //let kernel = array![[[[1.0f32, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]]];

        let _stride_size = (1, 1);
        let _pad_size = (1, 1);

        //let output = conv2d_array(input.view(), kernel.view(), None, stride_size, pad_size);
        //println!("{:?}", output);

        //assert_eq!(output_size, (4, 4));

        Ok(())
    }

    #[test]
    fn conv2d_array_2ch_test() {
        let input = array![[[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]];
        let kernel = array![[[[1.0f32, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]];

        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = conv2d_array(input.view(), kernel.view(), None, stride_size, pad_size);
        println!("{:?}", output); //55.0
    }

    #[test]
    fn max_pool2d_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;

        let input_tensor = Tensor::from_vec(
            vec![
                4.0f32, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
                4.0, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
            ],
            vec![2, 1, 4, 4],
        )?;

        /*
        let input_array: Array4<f32> = array![[
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ],
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ]
        ]]; */

        println!("input_shape = {:?}", input_tensor.shape());

        let input = input_tensor.rv();
        let kernel_size = (2, 2);
        let stride_size = (2, 2);
        let pad_size = (0, 0);

        let mut output = max_pool2d_simple(&input, kernel_size, stride_size, pad_size)?;

        println!("output = {}", output.data()); //shape = (1,2,3,3)

        output.backward(false)?;

        println!("input_grad= {}", input.grad().unwrap().data()); //shape = (1,5,15,15)

        Ok(())
    }

    #[test]
    fn max_pool2d_array_1ch_test() {
        use crate::functions_cnn::max_pool2d_array;

        let input = array![[[
            [4.0f32, 1.0, 5.0, 3.0],
            [7.0, 3.0, 2.0, 3.0],
            [7.0, 2.0, 3.0, 4.0],
            [1.0, 5.0, 3.0, 9.0]
        ]]];
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = max_pool2d_array(input.view(), kernel_size, stride_size, pad_size);
        println!("output = {:?}", output);
        //assert_eq!(output_size, (4, 4));
    }

    #[test]
    fn max_pool2d_array_2ch_test() {
        use crate::functions_cnn::max_pool2d_array;
        let input = array![[
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ],
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ]
        ]];
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = max_pool2d_array(input.view(), kernel_size, stride_size, pad_size);
        println!("output = {:?}", output);
    }

    #[test]
    fn im2col_test() -> FrameResult<()> {
        use crate::functions_cnn::im2col_array;

        let input_tensor = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![1, 1, 4, 4],
        )?;

        let input_tensor2 = Tensor::ones(Shape::new(vec![100, 3, 28, 28])?)?;

        let input_array = array![[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]];
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output_array = im2col_array(input_array.view(), kernel_size, stride_size, pad_size);
        let start = Instant::now();
        let output_tensor = input_tensor.im2col(kernel_size, stride_size, pad_size)?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("output_array = {:?}", output_array); //shape (1,4,9)
        println!("output_tensor = {:?}", output_tensor.to_vec()?); //shape (1,4,9)
        Ok(())
    } // input_tensor2 ...cpu(78ms) cuda(13µs)

    #[test]
    fn col2im_test() -> FrameResult<()> {
        use crate::functions_cnn::col2im_array;

        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]];

        println!("input shape = {:?}", input.shape());

        let input_tensor = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0,
                11.0, 12.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0, 6.0, 7.0, 8.0, 10.0,
                11.0, 12.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(vec![1, 4, 9])?,
        )?;

        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = col2im_array(
            input.view(),
            [1, 1, 4, 4],
            kernel_size,
            stride_size,
            pad_size,
        );

        let output_tensor =
            input_tensor.col2im([1, 1, 4, 4], kernel_size, stride_size, pad_size)?;
        println!("output = {:?}", output);
        println!("output_tensor = {:?}", output_tensor.to_vec()?);
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */

        Ok(())
    }

    #[test]
    fn im2col_function_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        let input = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![1, 1, 4, 4],
        )?
        .rv();

        let input2 = Tensor::ones(vec![1, 3, 15, 15])?;

        /*
        let input = array![[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]]
        .rv();
        */

        let kernel_size = (3, 3);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let mut output = im2col_simple(&input, kernel_size, stride_size, pad_size)?;

        println!("output = {}", output.data()); //shape (1,4,9)

        output.backward(false)?;
        println!("input_grad = {}", input.grad().unwrap().data());

        Ok(())
    }

    #[test]
    fn col2im_function_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0,
                11.0, 12.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0, 6.0, 7.0, 8.0, 10.0,
                11.0, 12.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 4, 9],
        )?
        .rv();

        /*
        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]]
        .rv(); */

        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let input_shape = [1, 1, 4, 4];

        let mut output = col2im_simple(&input, input_shape, kernel_size, stride_size, pad_size)?;

        println!("output = {}", output.data());
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */

        output.backward(false)?;
        println!("input_grad = {}", input.grad().unwrap().data());
        Ok(())
    }

    #[test]
    fn conv2d_layer_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        use crate::layers as L;
        use crate::models::BaseModel;

        let mut model = BaseModel::new();
        model.stack(L::Conv2d::new(4, (3, 3), (1, 1), (0, 0), false)?);

        let input_tensor = Tensor::ones(vec![2, 3, 15, 15])?;

        let input = input_tensor.rv();

        let mut y = model.call(&input)?;

        println!("y = {}", y.data()); // shape = [1,4,13,13]
        y.backward(false)?;

        println!("input_grad = {}", input.grad().unwrap().data()); // shape = [1,3,15,15]
        Ok(())
    }

    #[test]
    fn maxpool2d_layer_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        use crate::layers as L;
        use crate::models::BaseModel;

        let input_tensor = Tensor::from_vec(
            vec![
                4.0f32, 1.0, 5.0, 3.0, 8.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
                4.0, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 8.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
            ],
            vec![2, 1, 4, 4],
        )?;

        /*
        let input_array = array![[
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [8.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ],
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [8.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ]
        ]]; */
        let kernel_size = (2, 2);
        let stride_size = (2, 2);
        let pad_size = (0, 0);

        let mut model = BaseModel::new();
        model.stack(L::Maxpool2d::new(kernel_size, stride_size, pad_size));

        let input = input_tensor.rv();

        let mut y = model.call(&input)?;

        println!("y = {}", y.data()); // shape = [1,4,13,13]
        y.backward(false)?;

        println!("input_grad = {}", input.grad().unwrap().data()); // shape = [1,3,15,15]

        Ok(())
    }

    #[test]
    fn dropout_layer_test() -> FrameResult<()> {
        use crate::config::set_test_flag_true;
        use crate::core::TensorToRcVariable;
        use crate::layers as L;
        use crate::models::BaseModel;

        //set_test_flag_true();

        let input_tensor = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0, 1.0], vec![1, 5])?;

        //let input_array = array![[4.0f32, 1.0, 5.0, 3.0], [1.0, 5.0, 3.0, 9.0]];

        let ratio = 0.5f32;

        let mut model = BaseModel::new();
        model.stack(L::Dropout::new(ratio));

        let input = input_tensor.rv();

        let mut y = model.call(&input)?;

        println!("y = {}", y.data());
        y.backward(false)?;

        println!("input_grad = {}", input.grad().unwrap().data());

        Ok(())
    }

    #[test]
    fn flatten_layer_test() -> FrameResult<()> {
        use crate::core::TensorToRcVariable;
        use crate::layers as L;
        use crate::models::BaseModel;

        let input_tensor = Tensor::from_vec(
            vec![
                4.0f32, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
                4.0, 1.0, 5.0, 3.0, 7.0, 3.0, 2.0, 3.0, 7.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 9.0,
            ],
            vec![1, 2, 4, 4],
        )?;

        /*
        let input_array = array![[
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ],
            [
                [4.0f32, 1.0, 5.0, 3.0],
                [7.0, 3.0, 2.0, 3.0],
                [7.0, 2.0, 3.0, 4.0],
                [1.0, 5.0, 3.0, 9.0]
            ]
        ]]; */

        println!("input_shape = {:?}", input_tensor.shape()); //[1,2,4,4]

        let mut model = BaseModel::new();
        model.stack(L::Flatten::new());

        let input = input_tensor.rv();

        let mut y = model.call(&input)?;

        println!("y = {}", y.data()); // shape = [1,32]
        y.backward(false)?;

        println!("input_grad = {}", input.grad().unwrap().data()); // shape = [1,2,4,4]

        Ok(())
    }
}
