//use ndarray::{array, ArrayBase, Dimension, OwnedRepr};

use core_new::RcVariable;
use core_new::{add, div, mul, neg, sub};
use std::ops::{Add, Div, Mul, Neg, Sub};
//use std::sync::atomic::{AtomicU32, Ordering};

//演算子のオーバーロード

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[Some(self.clone()), Some(rhs.clone())]);
        add_y
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[Some(self.clone()), Some(rhs.clone())]);
        mul_y
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[Some(self.clone()), Some(rhs.clone())]);
        sub_y
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[Some(self.clone()), Some(rhs.clone())]);
        div_y
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[Some(self.clone()), None]);
        neg_y
    }
}

pub mod core_new;
//pub mod core_hdv;
pub mod config;
//pub mod dataloaders;
pub mod datasets;
pub mod functions_new;
pub mod layers;
pub mod models;
pub mod optimizers;
//pub mod functions_hdv;

#[cfg(test)]
mod tests {

    //use super::*;

    use tensor_frame::{Shape, Tensor, TensorOps};

    use crate::{
        core_new::TensorToRcVariable,
        functions_new::{relu, sigmoid_simple, sum},
    };

    #[test]
    fn sigmoid_test() {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap()
            .to_backend("CUDA")
            .expect("cudaだめ");

        let a = tensor.rv();
        let mut b = sigmoid_simple(&a);

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("b = {}", b.data());

        b.backward(false);

        println!("backward_result = {}", a.grad().unwrap().data());
    }
    #[test]
    fn reshape_test() {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap()
            .to_backend("CUDA")
            .expect("cudaだめ");

        let a = tensor.rv();
        let mut b = a.reshape(Shape::new(vec![1, 6]).unwrap());

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("b = {}", b.data());

        b.backward(false);

        println!("backward_result = {}", a.grad().unwrap().data());
    }

    #[test]
    fn sum_test() {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .to_backend("CUDA")
            .expect("cudaだめ");

        let a = tensor.rv();
        let mut b = sum(&a, Some(0));

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("b = {}", b.data());

        b.backward(false);

        println!("backward_result = {}", a.grad().unwrap().data());
    }



    #[test]
    fn relu_test() {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, -4.0, -5.0, 6.0], vec![2, 3])
            .unwrap()
            .to_backend("CUDA")
            .expect("cudaだめ");

        let a = tensor.rv();
        let mut b = relu(&a);

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("b = {}", b.data());

        b.backward(false);

        println!("backward_result = {}", a.grad().unwrap().data());
    }
}
