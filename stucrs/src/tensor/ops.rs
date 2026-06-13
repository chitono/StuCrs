// Derived from Tensor-Frame
// <https://github.com/TrainPioneers/Tensor-Frame>
// Original project licensed under MIT license option.

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
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    /// let sum = tensor.sum(None).unwrap();
    /// assert_eq!(sum.to_vec().unwrap(), vec![10.0]);
    /// ```
    fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<Self>
    where
        Self: Sized;

    /// Broadcasts a tensor to the specified shape.
    ///
    /// # Arguments
    ///
    /// * `to_shape` - The desired shape
    ///
    /// # Returns
    ///
    /// A tensor broadcasted.
    ///
    ///
    /// # Errors
    ///
    /// Returns an error if the desired shape is not compatible with broadcasting.
    ///
    /// # Examples
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1])?;
    /// let result = tensor.broadcast_to(Shape { dims: vec![4, 20] })?;
    /// assert_eq!(result.shape().dims(), vec![4,20]);
    /// ```
    ///
    ///
    fn broadcast_to(&self, to_shape: Shape) -> Result<Self>
    where
        Self: Sized;

    /// Reduces the tensor to the specified shape by summation.
    ///
    /// # Arguments
    ///
    /// * `to_shape` - The desired shape
    ///
    /// # Returns
    ///
    ///
    /// A tensor with shape `to_shape`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps,Shape};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    /// let sum = tensor.sum_to(&Shape { dims: vec![1, 2] }).unwrap();
    /// assert_eq!(sum.to_vec()?, vec![4.0,6.0]);
    /// ```

    fn sum_to(&self, to_shape: &Shape) -> Result<Self>
    where
        Self: Sized;
    /// Selects elements along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to slice.
    /// * `indices` - The indices to select along the axis.
    ///
    /// # Returns
    ///
    /// A tensor containing the selected elements.
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps,Shape};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
    /// let result = tensor.axis_slice(0, &[0, 2])?;
    /// assert_eq!(result.to_vec()?, vec![1.0, 2.0, 5.0, 6.0]);
    /// ```
    ///
    fn axis_slice(&self, axis: usize, indices: &[usize]) -> Result<Self>
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

    /// Permute the axes.
    ///
    /// # Arguments
    ///
    /// * `axes` The axes indices
    ///
    /// # Returns
    ///
    /// A tensor permuted axes, which is containing the same data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - any of the axes are out of bounds
    /// - an axis is missing
    /// - an axis is repeated more than once
    ///
    /// # Examples
    ///
    /// ```
    ///     
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1,2, 3]).unwrap();
    /// let result = tensor.permute(&vec![1, 2, 0])?;
    /// assert_eq!(result.shape().dims(), &[2, 3, 1]);
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
    fn squeeze(&self, axis: usize) -> Result<Self>
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

    /// Tensor multiplication with automatic rank dispatch.
    ///
    /// Performs matrix multiplication on batches of 2D tensors.
    ///
    /// If one operand is 2D and the other is 3D, the 2D tensor is broadcast
    /// across the batch dimension before mulitplication.
    ///
    /// Supported rank combination:
    /// - 3D × 2D
    ///
    /// The compatible dimensions: (N,k,l) ×　(l,m) -> (N,k,m)
    ///
    /// - 2D × 3D
    ///
    /// The compatible dimensions: (k,l) ×　(N,l,m) -> (N,k,m)
    ///
    /// - 3D × 3D
    ///
    /// The compatible dimensions: (N,k,l) ×　(N,l,m) -> (N,k,m)
    ///
    ///
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
    /// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
    /// let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], vec![2, 2, 2])?;
    ///
    /// let result = a.tensordot(&b)?; // 3D × 3D
    /// // result ≈ [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]
    /// ```
    fn tensordot(&self, other: &Self) -> Result<Self>
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
    fn bmm(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise negation.
    ///
    /// This method is use by negation operation.
    ///
    /// # Returns
    ///
    /// A new tensor with the negation element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
    /// let result = -tensor?;
    /// assert_eq!(result.to_vec()/, vec![-2.0, -3.0, -4.0]);
    /// ```
    fn neg(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise exponential function.
    ///
    /// Applies the exponential function (exp(x)) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with the exponential applied element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3])?;
    /// let result = tensor.exp()?;
    /// //result ≈ [1.0, 2.718, 7.389]
    /// ```
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
    /// assert_eq!(result.to_vec()?, vec![4.0, 9.0, 16.0]);
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
    /// let tensor = Tensor::from_vec(vec![0.0, PI/2.0, PI], vec![3])?;
    /// let result = tensor.sin()?;
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
    /// let tensor = Tensor::from_vec(vec![0.0, PI/2.0, PI], vec![3])?;
    /// let result = tensor.cos()?;
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
    /// let tensor = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5])?;
    /// let result = tensor.relu()?;
    /// assert_eq!(result.to_vec()?, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    /// ```
    fn relu(&self) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise mask for values greater than `max`.
    ///
    /// Elements greater than `max` are set to `1.0`,
    /// while all other elements are set to `0.0`.
    ///
    /// 入力値がmaxより大きい場合は1.0を、それ以下は0.0を返す。
    ///
    ///
    /// # Arguments
    ///
    /// * `max` - The threshold value
    ///
    /// # Return
    ///
    /// A new tensor containing the mask.
    ///
    /// # Example
    /// ```
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
    ///
    /// let result = tensor.max_mask(3.0f32)?;
    /// assert_eq!(result.to_vec()?, vec![0.0, 0.0, 0.0, 1.0, 1.0,1.0]);
    /// ```
    ///
    fn max_mask(&self, max: f32) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise mask for values less than `min`.
    ///
    /// Elements less than `min` are set to `1.0`,
    /// while all other elements are set to `0.0`.
    ///
    /// 入力値がminより小さい場合は1.0を、それ以上は0.0を返す。
    ///
    ///
    /// # Arguments
    ///
    /// * `min` - The threshold value
    ///
    /// # Return
    ///
    /// A new tensor containing the mask.
    ///
    /// # Example
    /// ```
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])?;
    ///
    /// let result = tensor.min_mask(3.0f32)?;
    /// assert_eq!(result.to_vec()?, vec![1.0, 1.0, 0.0, 0.0, 0.0,0.0]);
    /// ```
    ///
    fn min_mask(&self, min: f32) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise ReLU gradient mask.
    ///
    /// relu関数のバックプロパゲーション用関数
    ///
    /// Elements greater than `0.0` are set to `1.0`,
    /// and all other elements are set to `0.0`.   
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

    /// Element-wise hyperbolic sine function.
    ///
    /// Applies sinh(x) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with sinh applied element-wise.
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

    /// Element-wise hyperbolic cosine function.
    ///
    /// Applies cosh(x) to each element.
    ///
    /// # Returns
    ///
    /// A new tensor with cosh applied element-wise.
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
    fn cosh(&self) -> Result<Self>
    where
        Self: Sized;

    /// Computes the maximum of a tensor or maximum along a specified axis.   
    ///
    ///# Arguments
    ///
    /// * `axis` - Optional axis along which the maximum is computed.
    /// If `None`, the maximum of all elements ic computed.
    ///
    /// # Returns
    ///
    /// A tensor containing the maximum value.
    ///
    /// # Examples
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
    /// let result = tensor.max(Some(0))?;
    /// assert_eq!(result.to_vec()?, vec![5.0, 6.0, 7.0, 8.0]);
    /// ```
    ///
    fn max(&self, axis: Option<usize>) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise upper-bound mask for clamp.
    ///
    /// Elements greater than `max` are set to `max`,
    /// while all other elements remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `max` - The threshold value.
    ///    
    /// 入力された値maxよりも大きい場合はmaxを、それ以下はそのまま値を流す。
    ///
    fn clamp_max(&self, max: f32) -> Result<Self>
    where
        Self: Sized;

    /// Element-wise lower-bound mask for clamp.
    ///
    /// Elements less than `min` are set to `min`,
    /// while all other elements remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `min` - The threshold value.
    ///
    /// 入力された値minよりも小さい場合はminを、それ以上はそのまま値を流す。
    ///
    fn clamp_min(&self, min: f32) -> Result<Self>
    where
        Self: Sized;

    /// Gradient mask for clamp upper bound.
    ///
    /// Elements less than `1.0` are set to `1.0`,
    /// while all other elements are set to `0.0`.
    ///
    /// 入力の要素が1.0より小さいときは1.0を、それ以外は0.0を返す。
    ///
    fn max_for_clamp_grad(&self) -> Result<Self>
    where
        Self: Sized;

    /// Gradient mask for clamp lower bound.
    ///
    /// Elements greater than `0.0` are set to `1.0`,
    /// while all other elements are set to `0.0`.
    ///
    /// 入力の要素が0.0より大きいときは1.0を、それ以外は0.0を返す。
    ///
    fn min_for_clamp_grad(&self) -> Result<Self>
    where
        Self: Sized;

    /// Argmax along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to find the indices of the maximum values.
    ///
    /// # Returns
    ///
    /// A tensor containing the indices of the maximum values.
    ///
    /// # Examples
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0], vec![2, 3])?;
    /// let result =  tensor.argmax_axis(1)?;;
    /// assert_eq!(result.to_vec()?, vec![2.0, 0.0]);
    /// ```
    ///
    ///
    ///
    fn argmax_axis(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;

    /// Converts argmax indices to a gradient mask for the `Max`` function.
    ///
    /// Elements at the argmax indices are set to `1.0`,
    /// and all other elements are set to `0.0`.
    ///
    /// # Arguments
    ///
    /// * `to_shape` - The shape of the input tensor originally passed to the `Max` function.
    ///
    /// * `axis` - The axis along which the `Max` function was applied.
    ///
    /// # Examples
    ///
    ///
    /// This is the `Max` function gradient sample.
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])?;
    /// let tensor_argmax = tensor.argmax_axis(1)?;
    /// let result = tensor_argmax.argmax_to_max_backward(&tensor.shape(), 1)?;
    /// assert_eq!(result.to_vec()?,vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    /// ```
    ///
    fn argmax_to_max_backward(&self, to_shape: &Shape, axis: usize) -> Result<Self>
    where
        Self: Sized;

    /// Argmax along the specified axis of a 2D tensor.
    ///
    /// # Note
    /// Since `argmax_axis` supports N-dimentional tensors and covers this functionality,
    ///  It is generally recommend to use `argmax_axis` instead.
    fn argmax_axis_2d(&self, axis: usize) -> Result<Self>
    where
        Self: Sized;

    /// One-hot encoding.
    ///
    /// # Arguments
    ///
    /// * `num_class` - The total number of classes for the one-hot encoding.
    ///
    /// # Performance
    /// This function is not optimized for performance.
    /// It is not recommended to call it frequently.
    fn one_hot_encode(&self, num_class: usize) -> Result<Self>
    where
        Self: Sized;

    /// Compute Im2col function.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - A tuple `(kernel_height, kernel_width)` representing the dimentions of the convolution filter.
    ///
    ///
    ///
    /// * `stride_size` - A tuple `(stride_height, stride_width)` specifying the step size of the moving filter.
    ///
    ///
    /// * `pad_size` - A tuple `(pad_height, pad_width)` - indicating the number of the zero-padding pixels to add each side.
    ///
    ///
    /// # Examples
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let input_tensor = Tensor::from_vec(
    ///     vec![
    ///         1.0f32, 2.0, 3.0, 4.0,
    ///         5.0, 6.0, 7.0, 8.0,
    ///         9.0, 10.0, 11.0, 12.0,
    ///         13.0, 14.0, 15.0, 16.0,],
    ///     vec![1, 1, 4, 4],
    /// )?;
    ///
    ///
    /// let kernel_size = (2, 2);
    /// let stride_size = (1, 1);
    /// let pad_size = (0, 0);
    ///
    /// let output_tensor = input_tensor.im2col(kernel_size, stride_size, pad_size)?;
    ///
    /// assert_eq!(output_tensor.to_vec()?,
    /// vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0,                                      
    ///     2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0,                                      
    ///     5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0,                                  
    ///     6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]);
    /// ```
    ///
    ///
    ///
    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Self>
    where
        Self: Sized;

    /// Compute Col2im function.
    ///
    ///
    /// # Arguments
    ///
    /// * `im_shape` - The shape of the input tensor originally passed to the `Im2col` function.
    ///
    /// * `kernel_size` - A tuple `(kernel_height, kernel_width)` representing the dimentions of the convolution filter.
    ///
    /// * `stride_size` - A tuple `(stride_height, stride_width)` specifying the step size of the moving filter.
    ///
    /// * `pad_size` - A tuple `(pad_height, pad_width)` - indicating the number of the zero-padding pixels to add each side.
    ///
    ///
    /// # Examples
    /// ```
    /// use tensor_frame::{Tensor, TensorOps};
    ///
    /// let input_tensor = Tensor::from_vec(
    ///    vec![
    ///         1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0,
    ///         2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0,
    ///         5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0,
    ///         6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0,
    ///     ],
    ///     Shape::new(vec![1, 4, 9])?,
    /// )?;
    ///
    /// let kernel_size = (2, 2);
    /// let stride_size = (1, 1);
    /// let pad_size = (0, 0);
    ///
    /// let output_tensor = input_tensor.col2im([1, 1, 4, 4], kernel_size, stride_size, pad_size)?;
    /// assert_eq!(output_tensor.to_vec()?,
    ///     vec![
    ///         10.0, 4.0, 6.0, 4.0,
    ///         10.0, 24.0, 28.0, 16.0,
    ///         18.0, 40.0, 44.0, 24.0,
    ///         13.0, 28.0, 30.0, 16.0,]
    /// );
    /// ```
    ///
    ///
    ///
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
