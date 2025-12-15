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
        let add_y = add(&[self.clone(), rhs.clone()]);
        add_y
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[self.clone(), rhs.clone()]);
        mul_y
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[self.clone(), rhs.clone()]);
        sub_y
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[self.clone(), rhs.clone()]);
        div_y
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[self.clone()]);
        neg_y
    }
}

pub mod config;
pub mod core_new;
pub mod dataloaders;
pub mod datasets;
pub mod functions;
pub mod functions_cnn;
pub mod layers;
pub mod models;
pub mod optimizers;

#[cfg(test)]
mod tests {

    use ndarray::{array, Array, Array4, Dim, IxDyn};

    use crate::{
        functions::activation_funcs::relu,
        functions::math::{cos, exp, log, max, sin, square, tanh},
        functions::matrix::{
            argmax_array, matmul, permute_axes, reshape, sum, tensordot, transpose,
        },
        functions_cnn::{
            col2im_simple, conv2d_array, conv2d_simple, im2col_simple, max_pool2d_simple,
        },
    };

    #[test]
    fn add_test() {
        use crate::core_new::ArrayDToRcVariable;
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]

        let a = array![1.0, 1.0, 1.0, 1.0, 1.0].rv();

        let b = array![2.0, 2.0, 2.0, 2.0, 2.0].rv();

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        let mut c = a.clone() + b.clone();

        println!("c = {}", c.data());

        c.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data());
        println!("b_grad = {:?}", b.grad().unwrap().data());
    }

    #[test]
    fn add_with_broadcast_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![1.0, 1.0, 1.0, 1.0, 1.0].rv();

        let b = array![2.0].rv();

        let mut c = a.clone() + b.clone();

        println!("c = {}", c.data()); // [3,3,3,3,3]

        c.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // [1.0,1.0,1.0,1.0,1.0]
        println!("b_grad = {:?}", b.grad().unwrap().data()); // [5.0]
    }

    #[test]
    fn mul_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let b = array![2.0, 2.0, 2.0, 2.0, 2.0].rv();

        let c = array![1.0, 1.0, 1.0, 1.0, 1.0].rv();

        let mut y = (a.clone() * b.clone()) + c.clone();

        println!("c = {}", y.data()); // 7.0

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 2.0
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 3.0
    }

    #[test]
    fn sub_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let b = array![2.0, 2.0, 2.0, 2.0, 2.0].rv();

        let c = array![1.0, 1.0, 1.0, 1.0, 1.0].rv();

        let mut y = (a.clone() * b.clone()) - c.clone();

        println!("y = {}", y.data());

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data());
        println!("b_grad = {:?}", b.grad().unwrap().data());
        println!("c_grad = {:?}", c.grad().unwrap().data());
    }

    #[test]
    fn div_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let b = array![2.0, 2.0, 2.0, 2.0, 2.0].rv();

        let mut y = a.clone() / b.clone();

        println!("y = {}", y.data()); // 1.5

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 0.5
        println!("b_grad = {:?}", b.grad().unwrap().data()); // -0.75
    }

    #[test]
    fn pow_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let mut y = a.clone().pow(2.0);

        println!("y = {}", y.data()); // 9.0

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
    }

    #[test]
    fn square_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let mut y = square(&a);

        println!("y = {}", y.data()); // 9.0

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
    }

    #[test]
    fn exp_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0].rv();
        let b = array![2.0, 2.0, 2.0].rv();

        let mut y0 = exp(&a);
        let mut y1 = b.clone().exp();

        println!("y0= {}", y0.data()); // 20.0855...
        println!("y1= {}", y1.data()); // 7.3890...

        y0.backward(false);
        y1.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 20.0855
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 7.3890
    }

    #[test]
    fn sin_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0].rv();

        let mut y = sin(&a);

        println!("y = {}", y.data()); // 0.1411

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // -0.9899
    }

    #[test]
    fn cos_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0].rv();

        let mut y = cos(&a);

        println!("y = {}", y.data()); // -0.9899

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // -0.1411
    }

    #[test]
    fn tanh_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0].rv();

        let mut y = tanh(&a);

        println!("y = {}", y.data()); // 0.9950...

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 9.866...e-3
    }

    #[test]
    fn log_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0].rv();
        let b = array![3.0, 3.0, 3.0].rv();

        let mut y0 = log(&a, None); //底がe
        let mut y1 = log(&b, Some(2.0)); //底が2.0

        println!("y0 = {}", y0.data()); // 1.098...
        println!("y1 = {}", y1.data()); // 1.584...

        y0.backward(false);
        y1.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 0.3333...
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 0.4808...
    }

    #[test]
    fn reshape_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();

        let mut y0 = reshape(&a, Dim(IxDyn(&[1, 6])));

        let mut y1 = b.reshape(Dim(IxDyn(&[1, 6])));

        println!("y0 = {}", y0.data()); //[[1,2,3,4,5,6]] shape(1,6)
        println!("y1 = {}", y1.data()); //[[1,2,3,4,5,6]] shape(1,6)

        y0.backward(false);
        y1.backward(false);
        println!("a_grad = {:?}", a.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
        println!("b_grad = {:?}", b.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
    }

    #[test]
    fn transpose_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();

        let mut y0 = transpose(&a);

        let mut y1 = b.t();

        println!("y0 = {}", y0.data()); //[[1,4],[2,5],[3,6]] shape(3,2)
        println!("y1 = {}", y1.data()); //[[1,4],[2,5],[3,6]] shape(3,2)

        y0.backward(false);
        y1.backward(false);
        println!("a_grad = {:?}", a.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
        println!("b_grad = {:?}", b.grad().unwrap().data()); // [[1.0,1.0,1.0],[1.0,1.0,1.0]] shape(2,3)
    }

    #[test]
    fn sum_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();
        let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();
        let c = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();

        let mut y0 = sum(&a, None);
        let mut y1 = sum(&b, Some(0));
        let mut y2 = sum(&c, Some(1));

        println!("y0 = {}", y0.data()); // 21.0
        println!("y1 = {}", y1.data()); //
        println!("y2 = {}", y2.data()); //

        y0.backward(false);
        y1.backward(false);
        y2.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); //
        println!("b_grad = {:?}", b.grad().unwrap().data()); //
        println!("c_grad = {:?}", c.grad().unwrap().data()); //
    }

    #[test]
    fn broadcast_to_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let mut y = a.clone().pow(2.0);

        println!("y = {}", y.data()); // 9.0

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
    }

    #[test]
    fn sum_to_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0, 3.0, 3.0].rv();

        let mut y = a.clone().pow(2.0);

        println!("y = {}", y.data()); // 9.0

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 6.0
    }

    #[test]
    fn matmul_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();

        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].rv();

        let mut y = matmul(&a, &b);

        println!("y = {}", y.data()); // [[58,64],[139,154]]
        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data());
        println!("b_grad = {:?}", b.grad().unwrap().data());
    }

    #[test]
    fn tensordot_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]].rv();
        println!("a_shape = {:?}", a.data().shape()); //[1,2,3]

        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].rv();
        println!("b_shape = {:?}", b.data().shape()); //[3,2]

        let mut y = tensordot(&a, &b);

        println!("y = {:?}", y.data()); //[[[58.0, 64.0],[139.0, 154.0]]]
        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); //[[[15.0, 19.0, 23.0],[15.0, 19.0, 23.0]]]
        println!("b_grad = {:?}", b.grad().unwrap().data()); //[[5.0, 5.0],[7.0, 7.0],[9.0, 9.0]]
    }

    #[test]
    fn permute_axes2d_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();

        let mut y = permute_axes(&a, vec![1, 0]);

        println!("y = {}", y.data());
        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data());
    }

    #[test]
    fn permute_axes3d_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]].rv();
        println!("a_shape = {:?}", a.data().shape()); //[1,2,3]

        let mut y = permute_axes(&a, vec![1, 2, 0]);

        println!("y = {:?}", y.data());
        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); //[1,2,3] a_shapeと同じならok
    }

    #[test]
    fn max_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![1.0, 2.0, 30.0, 4.0, 5.0, 6.0].rv(); //1次元
        let b = array![[1.0, 5.0, 3.0], [4.0, 5.0, 6.0]].rv(); //2次元
        let c = array![
            [[1.0, 3.0, 2.0], [6.0, 5.0, 4.0]],
            [[10.0, 20.0, 30.0], [60.0, 50.0, 40.0]]
        ]
        .rv(); //3次元

        let mut y0 = max(&a, None);
        println!("計算1完了");
        let mut y1 = max(&b, Some(1));
        println!("計算2完了");
        let mut y2 = max(&c, Some(2));
        println!("計3完了");

        println!("y0 = {}", y0.data()); // [30]
        println!("y1 = {}", y1.data()); // [[5],[6]]
        println!("y2 = {}", y2.data()); //

        y0.backward(false);
        println!("微分1完了");

        y1.backward(false);
        println!("微分2完了");
        y2.backward(false);
        println!("微分3完了");

        println!("a_grad = {:?}", a.grad().unwrap().data()); // [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        println!("b_grad = {:?}", b.grad().unwrap().data()); //[[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]
        println!("c_grad = {:?}", c.grad().unwrap().data()); //[[[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]],[[0.0, 0.0, 1.0],[1.0, 0.0, 0.0]]]
    }

    #[test]
    fn relu_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![0.0, -1.0, 3.0, 1.0, -4.0].rv();

        let mut y = relu(&a);

        println!("y = {}", y.data());

        y.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data());
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

        //let output = conv2d_array(input, weight, stride_size, pad_size)

        //assert_eq!(output_size, (4, 4));
    }

    #[test]
    fn conv2d_test() {
        use crate::core_new::ArrayDToRcVariable;

        let input_array: Array4<f32> = Array::from_elem((1, 5, 15, 15), 2.0);
        let weight_array: Array4<f32> = Array::from_elem((8, 5, 3, 3), 3.0);

        let input = input_array.rv();
        let weight = weight_array.rv();

        let stride_size = (1, 1);
        let pad_size = (1, 1);

        let mut output = conv2d_simple(&input, &weight, None, stride_size, pad_size);

        println!("output_shape = {:?}", output.data().shape()); //shape = (1,8,15,15)

        output.backward(false);

        println!(
            "input_grad_shape = {:?}",
            input.grad().unwrap().data().shape()
        ); //shape = (1,5,15,15)
    }

    #[test]
    fn conv2d_array_1ch_test() {
        let input = array![[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]];
        let kernel = array![[[[1.0f32, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]]];

        let stride_size = (1, 1);
        let pad_size = (1, 1);

        let output = conv2d_array(input.view(), kernel.view(), None, stride_size, pad_size);
        println!("{:?}", output);

        //assert_eq!(output_size, (4, 4));
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
    fn max_pool2d_test() {
        use crate::core_new::ArrayDToRcVariable;

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
        ]];

        println!("input_shape = {:?}", input_array.shape());

        let input = input_array.rv();
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let mut output = max_pool2d_simple(&input, kernel_size, stride_size, pad_size);

        println!("output = {}", output.data()); //shape = (1,8,15,15)

        output.backward(false);

        println!("input_grad= {}", input.grad().unwrap().data()); //shape = (1,5,15,15)
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
    fn im2col_test() {
        use crate::functions_cnn::im2col_array;

        let input = array![[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]];
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let output = im2col_array(input.view(), kernel_size, stride_size, pad_size);
        println!("output = {:?}", output); //shape (1,4,9)
    }

    #[test]
    fn col2im_test() {
        use crate::functions_cnn::col2im_array;

        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]];

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
        println!("output = {:?}", output);
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */
    }

    #[test]
    fn im2col_function_test() {
        use crate::core_new::ArrayDToRcVariable;

        let input = array![[[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]]
        .rv();
        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let mut output = im2col_simple(&input, kernel_size, stride_size, pad_size);

        println!("output = {:?}", output); //shape (1,4,9)

        output.backward(false);
        println!("input_grad = {:?}", input.grad().unwrap().data());
    }

    #[test]
    fn col2im_function_test() {
        use crate::core_new::ArrayDToRcVariable;

        // im2col_testの出力。(output)
        let input = array![[
            [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
            [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
            [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
        ]]
        .rv();

        let kernel_size = (2, 2);
        let stride_size = (1, 1);
        let pad_size = (0, 0);

        let input_shape = [1, 1, 4, 4];

        let mut output = col2im_simple(&input, input_shape, kernel_size, stride_size, pad_size);

        println!("output = {:?}", output);
        /*output = [[[[1.0, 4.0, 6.0, 4.0],
        [10.0, 24.0, 28.0, 16.0],
        [18.0, 40.0, 44.0, 24.0],
        [13.0, 28.0, 30.0, 16.0]]]] */

        output.backward(false);
        println!("input_grad = {:?}", input.grad().unwrap().data());
    }

    #[test]
    fn array_argmax2d_test() {
        let input = array![[1.0f32, 2.0], [4.0, 3.0], [10.0, 20.0], [30.0, 40.0]].into_dyn();

        let output = argmax_array(input.view(), Some(1));
        println!("{:?}", output);
    }

    #[test]
    fn array_argmax3d_test() {
        let input = array![[[1.0f32, 2.0], [4.0, 3.0]], [[10.0, 20.0], [30.0, 40.0]]].into_dyn();

        let output = argmax_array(input.view(), Some(2));
        println!("{:?}", output); //axis = 1 ...[[1, 1],[1, 1]] 返す行列はusize型
                                  //axis = 2 ...[[1, 0],[1, 1]]
    }
}
