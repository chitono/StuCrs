//! Tensor operations trait defining common tensor operations.
//!
//! This module provides the [`TensorOps`] trait which defines the interface
//! for various tensor operations including reductions, shape manipulations,
//! and transformations.

use crate::tensor::error::Result;
use crate::tensor::shape::Shape;

/// Trait defining common operations on tensors.
///
/// This trait provides a standard interface for tensor operations that can be
/// implemented by different tensor types. All operations return a `Result` to
/// handle potential errors gracefully.
///
/// # Examples
///
/// ```
/// use tensor_frame::{Tensor, TensorOps};
///
/// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
///
/// // Sum all elements
/// let sum = tensor.sum(None).unwrap();
/// assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
///
/// // Reshape the tensor
/// let reshaped = tensor.reshape(vec![4]).unwrap();
/// assert_eq!(reshaped.shape().dims(), &[4]);
/// ```
pub trait TensorOps {
    /// Computes the sum of tensor elements.
    ///
    /// # Arguments
    ///
    /// * `axis` - Optional axis along which to sum. If `None`, sums all elements.
    ///
    /// # Returns
    ///
    /// A tensor containing the sum. If summing all elements, returns a scalar tensor.
    ///
    /// # Attention!
    ///
    /// keepdimsは現在falseのみで使用してください。
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let sum = tensor.sum(None).unwrap();
    /// assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
    /// ```
    fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<Self>
    where
        Self: Sized;

    /// Computes the mean of tensor elements.
    ///
    /// # Arguments
    ///
    /// * `axis` - Optional axis along which to compute mean. If `None`, computes mean of all elements.
    ///
    /// # Returns
    ///
    /// A tensor containing the mean. If computing mean of all elements, returns a scalar tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
    /// let mean = tensor.mean(None).unwrap();
    /// assert_eq!(mean.to_vec().unwrap(), vec![5.0]);
    /// ```
    ///
    ///
    fn broadcast_to(&self, to_shape: Shape) -> Result<Self>
    where
        Self: Sized;

    fn sum_to(&self, to_shape: &Shape) -> Result<Self>
    where
        Self: Sized;

    fn axis_slice(&self, axis: usize, indices: &[usize]) -> Result<Self>
    where
        Self: Sized;

    fn mean(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Reshapes the tensor to a new shape.
    ///
    /// The new shape must have the same total number of elements as the original.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The desired shape
    ///
    /// # Returns
    ///
    /// A tensor with the new shape containing the same data.
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape has a different number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let reshaped = tensor.reshape(vec![3, 2]).unwrap();
    /// assert_eq!(reshaped.shape().dims(), &[3, 2]);
    /// ```
    fn reshape(&self, new_shape: Vec<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Transposes the tensor.
    ///
    /// Currently only supports 2D tensors. For a 2D tensor, swaps rows and columns.
    ///
    /// # Returns
    ///
    /// A new tensor with transposed dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2D.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    /// let transposed = tensor.transpose().unwrap();
    /// assert_eq!(transposed.shape().dims(), &[3, 2]);
    /// ```
    fn transpose(&self) -> Result<Self>
    where
        Self: Sized;

    fn permute(&self, axes: &Vec<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Removes dimensions of size 1 from the tensor shape.
    ///
    /// # Arguments
    ///
    /// * `axis` - Optional specific axis to squeeze. If `None`, removes all dimensions of size 1.
    ///
    /// # Returns
    ///
    /// A tensor with squeezed dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the specified axis doesn't have size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::ones(vec![2, 1, 3]).unwrap();
    /// let squeezed = tensor.squeeze(Some(1)).unwrap();
    /// assert_eq!(squeezed.shape().dims(), &[2, 3]);
    /// ```
    fn squeeze(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Adds a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    ///
    /// * `axis` - The position where to insert the new dimension
    ///
    /// # Returns
    ///
    /// A tensor with an additional dimension of size 1.
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::ones(vec![2, 3]).unwrap();
    /// let unsqueezed = tensor.unsqueeze(1).unwrap();
    /// assert_eq!(unsqueezed.shape().dims(), &[2, 1, 3]);
    /// ```
    fn unsqueeze(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;

    /// Matrix multiplication for 2D tensors.
    /// 2次元の行列積を求める関数。
    ///
    /// Performs matrix multiplication between two 2D tensors.
    /// The dimensions must be compatible: (M, K) × (K, N) → (M, N).
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand side tensor for multiplication
    ///
    /// # Returns
    ///
    /// A new tensor containing the matrix multiplication result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either tensor is not 2D
    /// - The inner dimensions don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    /// let result = a.matmul(&b).unwrap();
    /// assert_eq!(result.shape().dims(), &[2, 2]);
    /// ```
    fn matmul(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Batched matrix multiplication for 2D tensors.
    ///
    /// Performs matrix multiplication on batches of 2D tensors.
    /// The dimensions must be compatible: (B, M, K) × (B, K, N) → (B, M, N).
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand side tensor for multiplication
    ///
    /// # Returns
    ///
    /// A new tensor containing the batched matrix multiplication result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either tensor is not 3D
    /// - The batch sizes don't match
    /// - The matrix dimensions don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let a = Tensor::ones(vec![2, 3, 4]).unwrap(); // 2 batches of 3x4 matrices
    /// let b = Tensor::ones(vec![2, 4, 5]).unwrap(); // 2 batches of 4x5 matrices
    /// let result = a.bmm(&b).unwrap();
    /// assert_eq!(result.shape().dims(), &[2, 3, 5]); // 2 batches of 3x5 matrices
    /// ```
    fn tensordot(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;
    fn bmm(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    fn neg(&self) -> Result<Self>
    where
        Self: Sized;

    fn exp(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise natural logarithm.
    ///
    /// Applies the natural logarithm (ln(x)) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with the natural logarithm applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.718, 7.389], vec![3]).unwrap();
    /// let result = tensor.log().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0]
    /// ```
    fn log(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise square root.
    ///
    /// Applies the square root function to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with the square root applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], vec![4]).unwrap();
    /// let result = tensor.sqrt().unwrap();
    /// assert_eq!(result.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn sqrt(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise power function.
    ///
    /// Raises each element to the specified power.
    ///
    /// # Arguments
    ///
    /// * `power` - The exponent to apply
    ///
    /// # Returns
    ///
    /// A new tensor with each element raised to the specified power.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
    /// let result = tensor.pow(2.0).unwrap();
    /// assert_eq!(result.to_vec().unwrap(), vec![4.0, 9.0, 16.0]);
    /// ```
    fn pow(&self, power: f32) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise sine function.
    ///
    /// Applies the sine function to each element (in radians).
    ///
    /// # Returns
    ///
    /// A new tensor with the sine function applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// use std::f32::consts::PI;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, PI/2.0, PI], vec![3]).unwrap();
    /// let result = tensor.sin().unwrap();
    /// // result ≈ [0.0, 1.0, 0.0]
    /// ```
    fn sin(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise cosine function.
    ///
    /// Applies the cosine function to each element (in radians).
    ///
    /// # Returns
    ///
    /// A new tensor with the cosine function applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    /// use std::f32::consts::PI;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, PI/2.0, PI], vec![3]).unwrap();
    /// let result = tensor.cos().unwrap();
    /// // result ≈ [1.0, 0.0, -1.0]
    /// ```
    fn cos(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise ReLU activation function.
    ///
    /// Applies ReLU (Rectified Linear Unit): max(0, x) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with ReLU applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).unwrap();
    /// let result = tensor.relu().unwrap();
    /// assert_eq!(result.to_vec().unwrap(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    /// ```
    fn relu(&self) -> Result<Self>
    where
        Self: Sized;

    /// max関数のバックプロパゲーション用関数
    ///
    /// 入力値がmaxより大きい場合は1.0を、それ以下は0.0を返す。
    ///
    fn max_mask(&self, max: f32) -> Result<Self>
    where
        Self: Sized;

    /// min関数のバックプロパゲーション用関数
    ///
    /// 入力値がminより小さい場合は1.0を、それ以上は0.0を返す。
    ///
    fn min_mask(&self, min: f32) -> Result<Self>
    where
        Self: Sized;

    /// relu関数のバックプロパゲーション用関数
    ///
    /// 入力値が0.0より大きい場合は1.0を、それ以下は0.0を返す。
    ///
    fn mask_for_grad_relu(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise sigmoid activation function.
    ///
    /// Applies sigmoid: 1 / (1 + e^(-x)) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with sigmoid applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![-2.0, 0.0, 2.0], vec![3]).unwrap();
    /// let result = tensor.sigmoid().unwrap();
    /// // result ≈ [0.119, 0.5, 0.881]
    /// ```
    fn sigmoid(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise hyperbolic tangent activation function.
    ///
    /// 要素ごとにtanh関数を計算します。
    ///
    /// Applies tanh(x) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with tanh applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]).unwrap();
    /// let result = tensor.tanh().unwrap();
    /// // result ≈ [-0.762, 0.0, 0.762]
    /// ```
    fn tanh(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise hyperbolic sine activation function.
    ///
    /// 要素ごとにsinh関数を計算します。
    ///
    /// Applies sinh(x) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with tanh applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]).unwrap();
    /// let result = tensor.tanh().unwrap();
    /// // result ≈ [-0.762, 0.0, 0.762]
    /// ```
    fn sinh(&self) -> Result<Self>
    where
        Self: Sized;

    fn cosh(&self) -> Result<Self>
    where
        Self: Sized;

    /// 入力された行列の最大値を返す   
    ///
    ///
    /// 軸指定にも対応   
    ///
    /// 現在3次元までの行列に対応        
    ///
    fn max(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Max関数のバックプロパゲーション用関数
    ///
    /// Maxのバックプロパゲーションで使用するmaskをMax関数のinputの行列から生成する
    ///
    /// Max関数のinputの行列にこのメソッドを用いる
    fn max_backward(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    /// clamp用関数
    ///    
    /// 入力された値maxよりも大きい場合はmaxを、それ以下はそのまま値を流す。
    ///
    fn clamp_max(&self, max: f32) -> Result<Self>
    where
        Self: Sized;

    /// clamp用関数
    ///
    /// 入力された値minよりも小さい場合はminを、それ以上はそのまま値を流す。
    ///
    fn clamp_min(&self, min: f32) -> Result<Self>
    where
        Self: Sized;

    /// clamp関数のバックプロパゲーション用関数
    ///
    /// 入力の要素が1.0より大きいときは1.0を、それ以下は0.0を返す。
    ///
    fn max_for_clamp_grad(&self) -> Result<Self>
    where
        Self: Sized;

    /// clamp関数のバックプロパゲーション用関数
    ///
    /// 入力の要素が0.0より大きいときは1.0を、それ以下は0.0を返す。
    ///
    fn min_for_clamp_grad(&self) -> Result<Self>
    where
        Self: Sized;

    fn argmax_axis(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;

    fn argmax_axis_2d(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;

    fn one_hot_encode(&self, num_class: usize) -> Result<Self>
    where
        Self: Sized;

    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Self>
    where
        Self: Sized;

    fn col2im(
        &self,
        im_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Self>
    where
        Self: Sized;
}
