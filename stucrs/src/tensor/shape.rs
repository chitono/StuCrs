//! Shape representation and broadcasting functionality for tensors.
//!
//! This module provides the [`Shape`] struct which represents the dimensions
//! of a tensor and implements broadcasting logic compatible with NumPy-style
//! broadcasting rules.

use crate::tensor::error::{Result, TensorError};
use ndarray::{IntoDimension, IxDyn};
/// Represents the shape (dimensions) of a tensor.
///
/// A shape is essentially a list of dimension sizes. For example:
/// - A scalar has shape `[]`
/// - A vector of length 5 has shape `[5]`
/// - A 3x4 matrix has shape `[3, 4]`
/// - A 3D tensor might have shape `[2, 3, 4]`
///
/// The `Shape` struct also provides methods for checking broadcasting compatibility
/// and computing the result shape of broadcasted operations.
///
/// # Examples
///
/// ```
///
/// let shape = Shape::new(vec![2, 3, 4]).unwrap();
/// assert_eq!(shape.ndim(), 3);
/// assert_eq!(shape.numel(), 24);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new shape from a vector of dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - A vector of dimension sizes. Each dimension must be greater than 0.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new shape or an error if any dimension is invalid.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidShape` if:
    /// - The shape would result in more than `usize::MAX` elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3]).unwrap();
    /// assert_eq!(shape.dims(), &[2, 3]);
    ///
    /// // Empty tensors are allowed
    /// let empty_shape = Shape::new(vec![0]).unwrap();
    /// assert_eq!(empty_shape.numel(), 0);
    ///
    /// // Very large shapes will overflow
    /// assert!(Shape::new(vec![usize::MAX, 2]).is_err());
    /// ```
    pub fn new(dims: Vec<usize>) -> Result<Self> {
        // Check for potential overflow when computing total elements
        // Note: We allow zero dimensions as they represent empty tensors
        if !dims.is_empty() {
            let mut total: usize = 1;
            for &dim in &dims {
                if let Some(new_total) = total.checked_mul(dim) {
                    total = new_total;
                } else {
                    return Err(TensorError::InvalidShape(format!(
                        "Shape {dims:?} would overflow (too many elements)"
                    )));
                }
            }
        }

        Ok(Shape { dims })
    }

    /// Creates a scalar shape (empty dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Shape;
    ///
    /// let shape = Shape::scalar();
    /// assert_eq!(shape.dims(), &[] as &[usize]);
    /// assert_eq!(shape.numel(), 1);
    /// ```
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    /// Returns a slice of the dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the number of dimensions (rank) of the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Shape;
    ///
    /// assert_eq!(Shape::scalar().ndim(), 0);
    /// assert_eq!(Shape::new(vec![5]).unwrap().ndim(), 1);
    /// assert_eq!(Shape::new(vec![2, 3]).unwrap().ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements represented by this shape.
    ///
    /// For a scalar (empty shape), this returns 1.
    /// For other shapes, it's the product of all dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Shape;
    ///
    /// assert_eq!(Shape::scalar().numel(), 1);
    /// assert_eq!(Shape::new(vec![2, 3, 4]).unwrap().numel(), 24);
    /// ```
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Checks if this shape can be broadcast to another shape.
    ///
    /// Broadcasting follows NumPy-style rules:
    /// 1. If the shapes have different numbers of dimensions, the smaller shape
    ///    is padded with ones on the left.
    /// 2. Two dimensions are compatible when they are equal, or one of them is 1.
    ///
    /// # Arguments
    ///
    /// * `other` - The target shape to check broadcasting compatibility with
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Shape;
    ///
    /// let shape1 = Shape::new(vec![1, 3]).unwrap();
    /// let shape2 = Shape::new(vec![2, 3]).unwrap();
    /// assert!(shape1.can_broadcast_to(&shape2));
    ///
    /// let shape3 = Shape::new(vec![3]).unwrap();
    /// let shape4 = Shape::new(vec![2, 3]).unwrap();
    /// assert!(shape3.can_broadcast_to(&shape4));
    /// ```
    pub fn can_broadcast_to(&self, other: &Shape) -> bool {
        let self_dims = &self.dims;
        let other_dims = &other.dims;

        if self_dims.len() > other_dims.len() {
            return false;
        }

        let offset = other_dims.len() - self_dims.len();

        for (i, &self_dim) in self_dims.iter().enumerate() {
            let other_dim = other_dims[i + offset];
            if self_dim != 1 && self_dim != other_dim {
                return false;
            }
        }

        true
    }

    /// Computes the shape that would result from broadcasting two shapes together.
    ///
    /// Returns `None` if the shapes are not compatible for broadcasting.
    ///
    /// # Arguments
    ///
    /// * `other` - The other shape to broadcast with
    ///
    /// # Returns
    ///
    /// * `Some(Shape)` - The resulting broadcast shape if compatible
    /// * `None` - If the shapes cannot be broadcast together
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Shape;
    ///
    /// let shape1 = Shape::new(vec![2, 1]).unwrap();
    /// let shape2 = Shape::new(vec![1, 3]).unwrap();
    /// let result = shape1.broadcast_shape(&shape2).unwrap();
    /// assert_eq!(result.dims(), &[2, 3]);
    ///
    /// let shape3 = Shape::new(vec![3]).unwrap();
    /// let shape4 = Shape::new(vec![2, 3]).unwrap();
    /// let result = shape3.broadcast_shape(&shape4).unwrap();
    /// assert_eq!(result.dims(), &[2, 3]);
    /// ```
    pub fn broadcast_shape(&self, other: &Shape) -> Option<Shape> {
        let self_dims = &self.dims;
        let other_dims = &other.dims;

        let max_len = self_dims.len().max(other_dims.len());
        let mut result_dims = vec![1; max_len];

        for i in 0..max_len {
            let self_dim = if i < self_dims.len() {
                self_dims[self_dims.len() - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other_dims.len() {
                other_dims[other_dims.len() - 1 - i]
            } else {
                1
            };

            if self_dim == 1 {
                result_dims[max_len - 1 - i] = other_dim;
            } else if other_dim == 1 || self_dim == other_dim {
                result_dims[max_len - 1 - i] = self_dim;
            } else {
                return None; // Incompatible shapes
            }
        }

        // Use the validated constructor
        Shape::new(result_dims).ok()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims).expect("Invalid shape dimensions")
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec()).expect("Invalid shape dimensions")
    }
}

impl From<&Shape> for IxDyn {
    fn from(shape: &Shape) -> Self {
        IxDyn(&shape.dims)
    }
}
