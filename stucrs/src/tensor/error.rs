//! Error types for tensor operations.
//!
//! This module defines the error types that can occur during tensor operations,
//! including shape mismatches, invalid indices, and backend-specific errors.

use std::fmt;

/// The main error type for tensor operations.
///
/// This enum encompasses all possible errors that can occur when working with tensors,
/// from shape mismatches to backend-specific errors.
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Occurs when tensor shapes don't match for an operation.
    ///
    /// # Example
    /// ```
    /// # use tensor_frame::{Tensor, TensorError};
    /// let a = Tensor::ones(vec![2, 3]).unwrap();
    /// let b = Tensor::ones(vec![3, 2]).unwrap();
    /// // This will return a ShapeMismatch error if broadcasting is not possible
    /// let result = a + b;
    /// ```
    ShapeMismatch {
        /// The expected shape for the operation
        expected: Vec<usize>,
        /// The actual shape encountered
        got: Vec<usize>,
    },

    /// Occurs when an invalid shape is provided for tensor creation or reshaping.
    ///
    /// # Example
    /// ```
    /// # use tensor_frame::Tensor;
    /// // This will return an InvalidShape error because the data size doesn't match
    /// let result = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 2]);
    /// ```
    InvalidShape(String),

    /// Occurs when trying to access a tensor element with an out-of-bounds index.
    InvalidIndex {
        /// The index that was attempted
        index: Vec<usize>,
        /// The shape of the tensor
        shape: Vec<usize>,
    },

    /// Errors specific to the compute backend (CPU, WGPU, CUDA).
    ///
    /// These errors can include device initialization failures, memory allocation
    /// errors, or kernel execution failures.
    BackendError(String),

    /// Occurs when broadcasting rules cannot be satisfied.
    ///
    /// Broadcasting follows NumPy-style rules where dimensions are compatible if:
    /// - They are equal, or
    /// - One of them is 1
    BroadcastError(String),

    /// Occurs when an operation expects a specific number of dimensions.
    ///
    /// # Example
    /// ```
    /// # use tensor_frame::{Tensor, TensorOps};
    /// let tensor_1d = Tensor::ones(vec![5]).unwrap();
    /// // Transpose on 1D tensor might return a DimensionMismatch error
    /// let result = tensor_1d.transpose();
    /// ```
    DimensionMismatch {
        /// The expected number of dimensions
        expected: usize,
        /// The actual number of dimensions
        got: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }
            TensorError::InvalidShape(msg) => write!(f, "Invalid shape: {msg}"),
            TensorError::InvalidIndex { index, shape } => {
                write!(f, "Invalid index {index:?} for shape {shape:?}")
            }
            TensorError::BackendError(msg) => write!(f, "Backend error: {msg}"),
            TensorError::BroadcastError(msg) => write!(f, "Broadcast error: {msg}"),
            TensorError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// A type alias for `Result<T, TensorError>`.
///
/// This is the standard result type used throughout the tensor library.
pub type Result<T> = std::result::Result<T, TensorError>;
