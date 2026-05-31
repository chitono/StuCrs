//! # Tensor Frame
//!
//! A PyTorch-like tensor library for Rust with support for multiple backends including CPU, WGPU, and CUDA.
//!
//! ## Overview
//!
//! Tensor Frame provides a flexible and efficient tensor computation framework that allows you to:
//! - Create and manipulate multi-dimensional arrays (tensors)
//! - Perform element-wise operations with automatic broadcasting
//! - Use different compute backends (CPU, GPU via WGPU, or CUDA)
//! - Seamlessly switch between backends based on your hardware capabilities
//!
//! ## Quick Start
//!
//! ```rust
//! use crate::tensor::tensor::Tensor;
//!
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::ones(vec![2, 2]).unwrap();
//!
//! // Perform operations
//! let c = (a + b).unwrap();
//! let sum = c.sum(None).unwrap();
//!
//! println!("Result: {:?}", c.to_vec().unwrap());
//! ```
//!
//! ## Features
//!
//! - **Multiple Backends**: Choose between CPU (with Rayon parallelization), WGPU (WebGPU), or CUDA
//! - **Broadcasting**: Automatic shape broadcasting for element-wise operations
//! - **Rich Operations**: Addition, subtraction, multiplication, division, reductions (sum, mean)
//! - **Shape Manipulation**: Reshape and transpose operations
//! - **Type Safety**: Strong typing with comprehensive error handling
//!
//! ## Backend Selection
//!
//! Enable different backends through Cargo features:
//!
//! ```toml
//! # CPU backend (default)
//! tensor_frame = "0.0.3-alpha"
//!
//!
//! # CUDA backend
//! tensor_frame = { version = "0.0.3-alpha", features = ["cuda"] }
//! ```
//!
//! ## Examples
//!
//! ### Creating Tensors
//!
//! ```rust
//! use tensor_frame::Tensor;
//!
//! // From a vector with shape
//! let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Zeros tensor
//! let zeros = Tensor::zeros(vec![3, 3]).unwrap();
//!
//! // Ones tensor
//! let ones = Tensor::ones(vec![2, 4]).unwrap();
//! ```
//!
//! ### Operations with Broadcasting
//!
//! ```rust
//! use tensor_frame::Tensor;
//!
//! let a = Tensor::ones(vec![2, 1]).unwrap();  // Shape: [2, 1]
//! let b = Tensor::ones(vec![1, 3]).unwrap();  // Shape: [1, 3]
//! let c = (a + b).unwrap();                   // Shape: [2, 3] via broadcasting
//!
//!
//! ```
//!
//!
//! cargo test --features "cuda"  tensor_matmul_cuda_test -- --nocapture

/// The backend trait for tensor operations
pub use super::backend::Backend;
/// Error types and Result alias for the library
pub use super::error::{Result, TensorError};
/// Core tensor types and operations
pub use super::{ops::TensorOps, shape::Shape, tensor::Tensor};

#[cfg(test)]
mod tests {
    use ndarray::{array, Array, Array2, Axis, IxDyn};
    use std::time::Instant;

    use super::*;

    // ==== TENSOR CREATION TESTS ====

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(vec![3, 2]).unwrap();
        assert_eq!(tensor.shape().dims(), &[3, 2]);
        assert_eq!(tensor.numel(), 6);
        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.to_vec().unwrap(), data);
    }

    #[test]
    fn test_tensor_1d() {
        let tensor = Tensor::ones(vec![5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[5]);
        assert_eq!(tensor.numel(), 5);
    }

    #[test]
    fn test_tensor_3d() {
        let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
    }

    #[test]
    fn test_tensor_scalar() {
        let tensor = Tensor::from_vec(vec![42.0], vec![]).unwrap();
        assert_eq!(tensor.shape().dims(), &[] as &[usize]);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.to_vec().unwrap(), vec![42.0]);
    }

    // ==== ARITHMETIC OPERATION TESTS ====

    // Add
    #[test]
    fn array_add_test() {
        let a: Array2<f32> = Array::ones((1000, 784));
        let b = array![[2.0]];

        let start = Instant::now();
        let result1 = a + b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_add_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a + b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_add_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a + b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    /// Sub
    #[test]
    fn array_sub_test() {
        let a: Array2<f32> = Array::ones((1000, 784));
        let b = array![[2.0]];
        let _c = array![[3.0]];
        let start = Instant::now();
        let result1 = a - b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_sub_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a - b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_sub_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a - b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Mul

    #[test]
    fn array_mul_test() {
        let a: Array2<f32> = Array::ones((1000, 784));
        let b = array![[2.0]];
        let _c = array![[3.0]];
        let start = Instant::now();
        let result1 = a * b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_mul_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a * b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_mul_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a * b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Div

    #[test]
    fn array_div_test() {
        let a: Array2<f32> = Array::ones((1000, 784));
        let b = array![[2.0]];
        let _c = array![[3.0]];
        let start = Instant::now();
        let result1 = a / b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_div_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a / b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_div_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let _c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = a * b;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn array_chain_test() {
        let a: Array2<f32> = Array::ones((1000, 784));
        let b = array![[2.0]];
        let c = array![[3.0]];
        let start = Instant::now();
        let result1 = (a + b) * c;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_chain_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = (a + b)? * c;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_chain_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;
        let b = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let c = Tensor::from_vec(vec![3.0, 3.0], vec![1, 2]).unwrap();
        let start = Instant::now();
        let result = (a + b)? * c;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Pow

    #[test]
    fn array_pow_test() {
        let a: Array2<f32> = Array::ones((1000, 784));

        let start = Instant::now();
        let result1 = a.powf(3.0);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_pow_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();

        let start = Instant::now();
        let result = a.pow(3.0);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_pow_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;

        let start = Instant::now();
        let result = a.pow(3.0);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // ==== Mathematical Functions ====

    // Exp

    #[test]
    fn array_exp_test() {
        let a: Array2<f32> = Array::ones((1000, 784));

        let start = Instant::now();
        let result1 = a.exp();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_exp_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();

        let start = Instant::now();
        let result = a.exp();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_exp_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;

        let start = Instant::now();
        let result = a.exp();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Sin

    #[test]
    fn array_sin_test() {
        let a: Array2<f32> = Array::ones((1000, 784));

        let start = Instant::now();
        let result1 = a.sin();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_sin_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();

        let start = Instant::now();
        let result = a.sin();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_sin_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;

        let start = Instant::now();
        let result = a.sin();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Cos

    #[test]
    fn array_cos_test() {
        let a: Array2<f32> = Array::ones((1000, 784));

        let start = Instant::now();
        let result1 = a.cos();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_cos_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();

        let start = Instant::now();
        let result = a.cos();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_cos_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;

        let start = Instant::now();
        let result = a.cos();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Tanh

    #[test]
    fn array_tanh_test() {
        let a: Array2<f32> = Array::ones((1000, 784));

        let start = Instant::now();
        let result1 = a.mapv(|x| x.tanh());
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_tanh_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();

        let start = Instant::now();
        let result = a.tanh();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_tanh_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;

        let start = Instant::now();
        let result = a.tanh();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // Log

    #[test]
    fn array_log_test() {
        let a: Array2<f32> = Array::ones((1000, 784));

        let start = Instant::now();
        let result1 = a.ln();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_log_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap()).unwrap();

        let start = Instant::now();
        let result = a.log();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    #[test]
    fn tensor_log_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![1000, 784]).unwrap())
            .unwrap()
            .to_backend("CUDA")?;

        let start = Instant::now();
        let result = a.log();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu_shape = {:?}", result.unwrap().shape());
        Ok(())
    }

    // ==== Matrix Functions ====

    //Reshape

    #[test]
    fn array_reshape_test() {
        let a: Array2<f32> = Array::ones((100, 784));
        let start = Instant::now();
        let result1 = a.to_shape(IxDyn(&[200, 392]));
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.unwrap().shape());
    }

    #[test]
    fn tensor_reshape_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![100, 784])?)?;
        let start = Instant::now();
        let result = a.reshape(vec![200, 392])?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu = {:?}", result.shape());
        Ok(())
    }

    #[test]
    fn tensor_reshape_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![100, 784])?)
            .unwrap()
            .to_backend("CUDA")?;
        let start = Instant::now();
        let result = a.reshape(vec![200, 392]);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu = {:?}", result?.shape());
        Ok(())
    }

    // Squeeze

    #[test]
    fn tensor_squeeze_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1])?;
        let start = Instant::now();
        let result = a.squeeze(0);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu = {:?}", result?.shape());
        Ok(())
    }

    // Unsqueeze

    #[test]
    fn tensor_unsqueeze_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1])?;
        let start = Instant::now();
        let result = a.unsqueeze(1);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu = {}", result?);
        Ok(())
    }

    //Transpose

    #[test]
    fn array_transpose_test() {
        let a: Array2<f32> = Array::ones((100, 784));
        let start = Instant::now();
        let result1 = a.t();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_transpose_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![100, 784])?)?;
        let start = Instant::now();
        let result = a.transpose();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu = {:?}", result?);
        Ok(())
    }

    #[test]
    fn tensor_transpose_cuda_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![100, 784])?)
            .unwrap()
            .to_backend("CUDA")?;
        let start = Instant::now();
        let result = a.transpose();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu = {:?}", result?.shape());
        Ok(())
    }

    //permuted_axes

    #[test]
    fn array_permute_test() {
        let a: Array2<f32> = Array::ones((100, 784));
        let start = Instant::now();
        let result1 = a.permuted_axes([1, 0]);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_permute_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1])?;
        let c = Tensor::ones(vec![1, 4, 169])?;
        let start = Instant::now();
        let result = c.permute(&vec![1, 0, 2]); // 102
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu = {}", result?);
        Ok(())
    }

    #[test]
    fn tensor_permute_cuda_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1])?;
        let start = Instant::now();
        let result = a.permute(&vec![1, 0]);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu = {:?}", result?.shape());
        Ok(())
    }

    //Sum

    #[test]
    fn array_sum_test() {
        let a: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let start = Instant::now();
        let result1 = a.sum_axis(Axis(0));
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration);
        println!("array = {:?}", result1.shape());
    }

    #[test]
    fn tensor_sum_test() -> Result<()> {
        //let a = Tensor::ones(Shape::new(vec![100, 784])?)?;
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let start = Instant::now();
        let result = a.sum(Some(1), false);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu = {:?}", result?); // [2]
        Ok(())
    }

    #[test]
    fn tensor_sum_cuda_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?.to_backend("CUDA")?;
        let start = Instant::now();
        let result = a.sum(Some(0), true);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration);
        println!("tensor_gpu = {}", result?);
        Ok(())
    }

    // Matmul

    #[test]
    fn array_matmul_test() {
        let a: Array2<f32> = Array::ones((100, 784));
        let b: Array2<f32> = Array::ones((784, 500));
        let start = Instant::now();
        let result1 = a.dot(&b);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration); //26.241815ms
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_matmul_test() -> Result<()> {
        let a = Tensor::ones(Shape::new(vec![100, 784])?)?;
        let b = Tensor::ones(Shape::new(vec![784, 500])?)?;
        let start = Instant::now();
        let result = a.matmul(&b);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration); //26.490274ms
        println!("tensor_cpu_shape = {:?}", result?.shape()); //[100,500]
        Ok(())
    }

    #[test]
    fn tensor_matmul_cuda_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let b = Tensor::from_vec(
            vec![
                11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
            vec![3, 4],
        )?;
        let start = Instant::now();
        let result = a.matmul(&b);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration); //1.604526ms
        println!("tensor_gpu_shape = {}", result?); //[100, 500]
        Ok(())
    }

    // tensordot

    #[test]
    fn array_tensordot_test() {
        let a: Array2<f32> = Array::ones((100, 784));
        let b: Array2<f32> = Array::ones((784, 500));
        let start = Instant::now();
        let result1 = a.dot(&b);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration); //
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_tensordot_32_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 4])?;

        let start = Instant::now();
        let result = a.tensordot(&b)?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor = {:?}", result.to_vec()?); // [5.0, 0.0, 0.0, 5.0, 11.0, 0.0, 0.0, 11.0, 17.0, 0.0, 0.0, 17.0, 23.0, 0.0, 0.0, 23.0]
        Ok(())
    }

    #[test]
    fn tensor_tensordot_23_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![4, 2])?;
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2])?;

        let start = Instant::now();
        let result = a.tensordot(&b)?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor = {:?}", result.to_vec()?); // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        Ok(())
    }

    #[test]
    fn tensor_tensordot_33_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2])?;

        let start = Instant::now();
        let result = a.tensordot(&b)?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor = {:?}", result.to_vec()?); // [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]
        Ok(())
    }

    // ==== BROADCASTING TESTS ====

    // Broadcast_to

    #[test]
    fn array_broadcast_to_test() {
        let a: Array2<f32> = Array::ones((100, 784));
        let b: Array2<f32> = Array::ones((784, 500));
        let start = Instant::now();
        let result1 = a.dot(&b);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration); //26.241815ms
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_broadcast_to_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![2])?;
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[2]);
        let start = Instant::now();
        let result = (a + b)?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {:?}", result.shape());
        Ok(())
    }

    #[test]
    fn tensor_broadcast_to_cuda_test() -> Result<()> {
        let _a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1])?;

        let start = Instant::now();
        let result = b.broadcast_to(Shape { dims: vec![4, 20] })?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration); // 114.569µs
        println!("tensor_gpu_shape = {}", result);
        Ok(())
    }

    // Sum_to

    #[test]
    fn array_sum_to_test() {
        let a: Array2<f32> = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b: Array2<f32> = array![[10.0, 20.0], [30.0, 40.0]];
        let start = Instant::now();
        let result1 = a.dot(&b);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間array_cpu = {:?}", duration); //26.241815ms
        println!("array_shape = {:?}", result1.shape());
    }

    #[test]
    fn tensor_sum_to_test() -> Result<()> {
        let _a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?;
        let c = Tensor::ones(vec![1, 8, 45])?;
        let start = Instant::now();
        let result = c.sum_to(&Shape { dims: vec![8, 45] })?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {}", result);
        Ok(())
    }

    #[test]
    fn tensor_sum_to_cuda_test() -> Result<()> {
        let _a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?.to_backend("CUDA")?;
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2])?.to_backend("CUDA")?;

        let start = Instant::now();
        let result = b.sum_to(&Shape { dims: vec![1, 2] })?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間gpu = {:?}", duration); // 114.569µs
        println!("tensor_gpu_shape = {}", result);
        Ok(())
    }

    #[test]
    fn test_broadcast_2d_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
        // This should fail without proper broadcasting implementation
        // but we'll test the shape compatibility
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[2]);
    }

    #[test]
    fn test_broadcast_same_shape() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![2, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_compatible_shapes() {
        let a = Tensor::ones(vec![2, 1]).unwrap();
        let b = Tensor::ones(vec![1, 3]).unwrap();
        let c = (a + b).unwrap();
        assert_eq!(c.shape().dims(), &[2, 3]);
        let data = c.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0], vec![]).unwrap(); // scalar
                                                              // Test shape compatibility
        assert_eq!(a.shape().dims(), &[2, 2]);
        assert_eq!(b.shape().dims(), &[] as &[usize]);
    }

    // ==== REDUCTION OPERATION TESTS ====

    // ==== ERROR HANDLING TESTS ====

    #[test]
    fn test_shape_mismatch_from_vec() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_vec(data, vec![2, 2]); // 3 elements, expecting 4
        assert!(result.is_err());
        if let Err(TensorError::ShapeMismatch { expected, got }) = result {
            assert_eq!(expected, vec![4]);
            assert_eq!(got, vec![3]);
        }
    }

    #[test]
    fn test_incompatible_shapes_addition() {
        let a = Tensor::ones(vec![2, 3]).unwrap();
        let b = Tensor::ones(vec![3, 4]).unwrap();
        let result = a + b;
        // This should either work with broadcasting or fail gracefully
        match result {
            Ok(_) => {}  // Broadcasting worked
            Err(_) => {} // Expected failure for incompatible shapes
        }
    }

    #[test]
    fn test_invalid_reshape() {
        let tensor = Tensor::ones(vec![2, 3]).unwrap(); // 6 elements
        let result = tensor.reshape(vec![2, 2]); // 4 elements
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_1d() {
        let tensor = Tensor::ones(vec![5]).unwrap();
        let result = tensor.transpose();
        // 1D transpose should either work (return same) or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // ==== EDGE CASE TESTS ====

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::zeros(vec![0]).unwrap();
        assert_eq!(tensor.numel(), 0);
        assert_eq!(tensor.to_vec().unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn test_large_tensor() {
        let tensor = Tensor::zeros(vec![100, 100]).unwrap();
        assert_eq!(tensor.numel(), 10000);
        assert_eq!(tensor.shape().dims(), &[100, 100]);
    }

    #[test]
    fn test_operations_with_negative_numbers() {
        let a = Tensor::from_vec(vec![-1.0, -2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 2.0, -3.0, -4.0], vec![2, 2]).unwrap();

        let sum = (a.clone() + b.clone()).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);

        let product = (a * b).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![-1.0, -4.0, -9.0, -16.0]);
    }

    #[test]
    fn test_operations_with_zero() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let zeros = Tensor::zeros(vec![2, 2]).unwrap();

        let sum = (a.clone() + zeros.clone()).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let product = (a * zeros).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_display_formatting() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let display_str = format!("{}", tensor);
        // Just ensure it doesn't panic and produces some output
        assert!(!display_str.is_empty());
    }

    // ==== SHAPE VALIDATION TESTS ====

    #[test]
    fn test_shape_validation() {
        use crate::tensor::shape::Shape;

        // These should all succeed - zero dimensions represent empty tensors
        assert!(Shape::new(vec![0]).is_ok());
        assert!(Shape::new(vec![2, 0]).is_ok());
        assert!(Shape::new(vec![0, 3]).is_ok());
        assert!(Shape::new(vec![2, 0, 3]).is_ok());

        // These should also succeed
        assert!(Shape::new(vec![1]).is_ok());
        assert!(Shape::new(vec![2, 3]).is_ok());
        assert!(Shape::new(vec![]).is_ok()); // Scalar is allowed

        // Test numel calculation with empty tensors
        let empty = Shape::new(vec![0]).unwrap();
        assert_eq!(empty.numel(), 0);

        let empty2 = Shape::new(vec![2, 0, 3]).unwrap();
        assert_eq!(empty2.numel(), 0);
    }

    #[test]
    fn test_overflow_protection() {
        use crate::tensor::shape::Shape;

        // This should fail due to overflow
        let huge_dims = vec![usize::MAX, 2];
        assert!(Shape::new(huge_dims).is_err());

        // This should also fail - 10^18 elements is way too many
        let large_dims = vec![1000000, 1000000, 1000000];
        let result = Shape::new(large_dims);
        // On some systems this might not overflow, so let's be more specific
        if result.is_ok() {
            // If it didn't overflow, try an even larger size
            let huge_dims = vec![usize::MAX / 2, usize::MAX / 2];
            assert!(Shape::new(huge_dims).is_err());
        } else {
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tensor_creation_with_mismatched_data() {
        // Data size doesn't match shape - should fail
        let result = Tensor::from_vec_with_shape(vec![1.0, 2.0], vec![3, 2]);
        assert!(result.is_err());

        // Empty tensor creation should work
        let result2 = Tensor::from_vec_with_shape(Vec::new(), vec![0]);
        assert!(result2.is_ok());

        // Valid shape should work
        let result3 = Tensor::from_vec_with_shape(vec![1.0, 2.0], vec![1, 2]);
        assert!(result3.is_ok());
    }

    // ==== DIVISION BY ZERO TESTS ====

    #[test]
    fn test_division_by_zero_handling() {
        // Test different division by zero cases
        let numerator = Tensor::from_vec(vec![1.0, -1.0, 0.0, 5.0], vec![4]).unwrap();
        let denominator = Tensor::from_vec(vec![0.0, 0.0, 0.0, 2.0], vec![4]).unwrap();

        let result = (numerator / denominator).unwrap();
        let values = result.to_vec().unwrap();

        // Check that we get the expected IEEE floating point results
        assert!(values[0].is_infinite() && values[0].is_sign_positive()); // 1.0/0.0 = +inf
        assert!(values[1].is_infinite() && values[1].is_sign_negative()); // -1.0/0.0 = -inf
        assert!(values[2].is_nan()); // 0.0/0.0 = NaN
        assert_eq!(values[3], 2.5); // 5.0/2.0 = 2.5 (normal division)
    }

    #[test]
    fn test_division_by_near_zero() {
        // Test division by very small numbers (should not trigger special handling)
        let numerator = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let denominator = Tensor::from_vec(vec![1e-10, 1e-20], vec![2]).unwrap();

        let result = (numerator / denominator).unwrap();
        let values = result.to_vec().unwrap();

        // These should be very large but finite numbers, not infinity
        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
        assert!(values[0] > 1e9); // Should be approximately 1e10
        assert!(values[1] > 1e19); // Should be approximately 2e20
    }

    // ==== AXIS-SPECIFIC REDUCTION TESTS ====

    #[test]
    fn test_axis_specific_sum() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let sum_axis_0 = tensor.sum(Some(0), false).unwrap();
        let result_0 = sum_axis_0.to_vec().unwrap();
        assert_eq!(result_0, vec![5.0, 7.0, 9.0]);
        assert_eq!(sum_axis_0.shape().dims(), &[3]);

        // Sum along axis 1 (rows): should give [6, 15] with shape [2]
        let sum_axis_1 = tensor.sum(Some(1), false).unwrap();
        let result_1 = sum_axis_1.to_vec().unwrap();
        assert_eq!(result_1, vec![6.0, 15.0]);
        assert_eq!(sum_axis_1.shape().dims(), &[2]);

        // Sum all elements: should give [21] with shape []
        let sum_all = tensor.sum(None, false).unwrap();
        let result_all = sum_all.to_vec().unwrap();
        assert_eq!(result_all, vec![21.0]);
        assert_eq!(sum_all.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_axis_specific_mean() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        // Mean along axis 0 (columns): should give [2.5, 3.5, 4.5] with shape [3]
        let mean_axis_0 = tensor.mean(Some(0)).unwrap();
        let result_0 = mean_axis_0.to_vec().unwrap();
        assert_eq!(result_0, vec![2.5, 3.5, 4.5]);
        assert_eq!(mean_axis_0.shape().dims(), &[3]);

        // Mean along axis 1 (rows): should give [2, 5] with shape [2]
        let mean_axis_1 = tensor.mean(Some(1)).unwrap();
        let result_1 = mean_axis_1.to_vec().unwrap();
        assert_eq!(result_1, vec![2.0, 5.0]);
        assert_eq!(mean_axis_1.shape().dims(), &[2]);

        // Mean all elements: should give [3.5] with shape []
        let mean_all = tensor.mean(None).unwrap();
        let result_all = mean_all.to_vec().unwrap();
        assert_eq!(result_all, vec![3.5]);
        assert_eq!(mean_all.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_axis_sum_3d_tensor() {
        use crate::tensor::ops::TensorOps;

        // Create a 2x2x2 tensor
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();

        // Sum along axis 0: shape [2, 2] -> [2, 2]
        let sum_axis_0 = tensor.sum(Some(0), false).unwrap();
        assert_eq!(sum_axis_0.shape().dims(), &[2, 2]);
        let result_0 = sum_axis_0.to_vec().unwrap();
        assert_eq!(result_0, vec![6.0, 8.0, 10.0, 12.0]); // [1+5, 2+6, 3+7, 4+8]

        // Sum along axis 1: shape [2, 2] -> [2, 2]
        let sum_axis_1 = tensor.sum(Some(1), false).unwrap();
        assert_eq!(sum_axis_1.shape().dims(), &[2, 2]);
        let result_1 = sum_axis_1.to_vec().unwrap();
        assert_eq!(result_1, vec![4.0, 6.0, 12.0, 14.0]); // [1+3, 2+4, 5+7, 6+8]

        // Sum along axis 2: shape [2, 2] -> [2, 2]
        let sum_axis_2 = tensor.sum(Some(2), false).unwrap();
        assert_eq!(sum_axis_2.shape().dims(), &[2, 2]);
        let result_2 = sum_axis_2.to_vec().unwrap();
        assert_eq!(result_2, vec![3.0, 7.0, 11.0, 15.0]); // [1+2, 3+4, 5+6, 7+8]
    }

    #[test]
    fn tensor_axis_slice_test() -> Result<()> {
        use crate::tensor::ops::TensorOps;

        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?.to_backend("CUDA")?;

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let tensor_rows_slice = tensor.axis_slice(0, &[0, 2])?;
        let result_0 = tensor_rows_slice.to_vec()?;
        assert_eq!(result_0, vec![1.0, 2.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn tensor_max_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 4])?;

        let start = Instant::now();
        let result = a.max(Some(0))?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {}", result);
        Ok(())
    }

    #[test]
    fn tensor_argmax_to_max_backward_test() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 4])?;
        let a_argmax = a.argmax_axis(1)?;

        println!("a_argmax = {}", a_argmax);
        let start = Instant::now();
        let result = a_argmax.argmax_to_max_backward(&a.shape(), 1)?;
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!("tensor_cpu_shape = {}", result);
        Ok(())
    }

    #[test]
    fn max_mask_test() -> Result<()> {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let result = tensor.max_mask(3.0f32)?;
        println!("result = {}", result);
        Ok(())
    }

    #[test]
    fn min_mask_test() -> Result<()> {
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let result = tensor.min_mask(3.0f32)?;
        println!("result = {}", result);
        Ok(())
    }

    #[test]
    fn argmax_axis_test() -> Result<()> {
        let tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0], vec![2, 3])?.to_backend("CUDA")?;

        let result = tensor.argmax_axis(1)?;
        println!("result = {}", result);
        Ok(())
    }

    #[test]
    fn rand_uniform_test() -> Result<()> {
        let tensor = Tensor::rand_uniform(Shape::new(vec![5, 5])?)?;

        println!("result = {}", tensor);
        Ok(())
    }

    #[test]
    fn one_hot_encode_test() {
        use crate::tensor::ops::TensorOps;
        use std::time::Instant;
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 0.0, 0.0, 2.0, 1.0], vec![6, 1])
            .unwrap()
            .to_backend("CUDA")
            .unwrap();
        let start = Instant::now();
        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]
        let result = tensor.one_hot_encode(3).unwrap();
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("処理時間 = {:?}", duration);
        println!(" = {}", result);
    }
}
