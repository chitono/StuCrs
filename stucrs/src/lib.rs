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
pub mod dataloaders;
pub mod datasets;
pub mod functions_cnn;
pub mod functions_new;
pub mod layers;
pub mod models;
pub mod optimizers;

#[cfg(test)]
mod tests {

    use ndarray::{array, Array4};

    use crate::{config::set_test_flag_true, functions_cnn::conv2d_array};

    use super::*;

    #[test]
    fn dropout_test() {
        use crate::{core_new::ArrayDToRcVariable, functions_new::dropout};
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]

        let a = array![1.0, 1.0, 1.0, 1.0, 1.0].rv();

        let b = dropout(&a, 0.5);

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("b = {}", b.data());

        set_test_flag_true();

        let c = dropout(&a, 0.5);

        println!("c = {}", c.data());
    }

    #[test]
    fn get_conv_outsize_test() {
        use crate::{core_new::ArrayDToRcVariable, functions_cnn::get_conv_outsize};

        let input_size = (4, 4);
        let kernel_size = (3, 3);
        let stride_size = (1, 1);
        let pad_size = (1, 1);

        let output_size = get_conv_outsize(input_size, kernel_size, stride_size, pad_size);

        assert_eq!(output_size, (4, 4));
    }

    #[test]
    fn dim_test() {
        use crate::{core_new::ArrayDToRcVariable, functions_cnn::get_conv_outsize};

        let input = array![[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]];
        let input_2 = array![[[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]];

        assert_eq!(input.ndim(), 4);
        assert_eq!(input_2.ndim(), 4);

        //let output = conv2d_array(input, weight, stride_size, pad_size)

        //assert_eq!(output_size, (4, 4));
    }

    #[test]
    fn conv2d_array_1ch_test() {
        use crate::{core_new::ArrayDToRcVariable, functions_cnn::get_conv_outsize};

        let input = array![[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
        let kernel = array![[[[1.0f32, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]]];

        let stride_size = (1, 1);
        let pad_size = (1, 1);

        let output = conv2d_array(input.view(), kernel.view(), stride_size, pad_size);
        println!("{:?}", output);

        //assert_eq!(output_size, (4, 4));
    }

    #[test]
    fn conv2d_array_2ch_test() {
        use crate::{core_new::ArrayDToRcVariable, functions_cnn::get_conv_outsize};

        let input = array![[[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]];
        let kernel = array![[[[1.0f32, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]];

        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = conv2d_array(input.view(), kernel.view(), stride_size, pad_size);
        println!("{:?}", output); //55.0
    }
}
