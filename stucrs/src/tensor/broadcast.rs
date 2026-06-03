// Derived from Tensor-Frame
// <https://github.com/TrainPioneers/Tensor-Frame>
// Original project licensed under MIT license option.

//! Broadcasting functionality for tensor operations.
//!
//! This module implements NumPy-style broadcasting rules for tensors, allowing
//! operations between tensors of different but compatible shapes.
//!
//! # Broadcasting Rules
//!
//! Two tensors are compatible for broadcasting if:
//! 1. They have the same shape, or
//! 2. For each dimension (starting from the rightmost):
//!    - The dimensions are equal, or
//!    - One of the dimensions is 1
//!
//! The smaller tensor is "broadcast" across the larger tensor by repeating
//! its elements along the dimensions where it has size 1.

use crate::tensor::error::{Result, TensorError};
use crate::tensor::shape::Shape;

/// Broadcasts two tensors to a common shape for element-wise operations.
///
/// This function takes two tensors (represented by their data and shapes) and
/// broadcasts them to a common result shape, returning the broadcasted data
/// for both tensors.
///
/// # Arguments
///
/// * `lhs_data` - The data of the left-hand side tensor
/// * `lhs_shape` - The shape of the left-hand side tensor
/// * `rhs_data` - The data of the right-hand side tensor
/// * `rhs_shape` - The shape of the right-hand side tensor
/// * `result_shape` - The target shape to broadcast to
///
/// # Returns
///
/// A tuple containing the broadcasted data for both tensors.
///
/// # Errors
///
/// Returns an error if either tensor cannot be broadcast to the result shape.
///
/// # Examples
///
/// ```ignore
/// // Broadcasting a [2, 1] tensor with a [1, 3] tensor to [2, 3]
/// let lhs = vec![1.0, 2.0];
/// let lhs_shape = Shape::new(vec![2, 1]).unwrap();
/// let rhs = vec![10.0, 20.0, 30.0];
/// let rhs_shape = Shape::new(vec![1, 3]).unwrap();
/// let result_shape = Shape::new(vec![2, 3]).unwrap();
///
/// let (lhs_broadcast, rhs_broadcast) = broadcast_data(
///     &lhs, &lhs_shape, &rhs, &rhs_shape, &result_shape
/// ).unwrap();
/// ```
pub fn broadcast_data(
    lhs_data: &[f32],
    lhs_shape: &Shape,
    rhs_data: &[f32],
    rhs_shape: &Shape,
    result_shape: &Shape,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let lhs_broadcasted = broadcast_single(lhs_data, lhs_shape, result_shape)?;
    let rhs_broadcasted = broadcast_single(rhs_data, rhs_shape, result_shape)?;
    Ok((lhs_broadcasted, rhs_broadcasted))
}

/// Broadcasts a single tensor to a target shape.
///
/// This internal function handles the actual broadcasting logic for a single tensor.
/// It expands the tensor data by repeating elements along dimensions where the
/// original shape has size 1.
fn broadcast_single(data: &[f32], from_shape: &Shape, to_shape: &Shape) -> Result<Vec<f32>> {
    if from_shape == to_shape {
        return Ok(data.to_vec());
    }

    if !from_shape.can_broadcast_to(to_shape) {
        return Err(TensorError::BroadcastError(format!(
            "Cannot broadcast shape {:?} to {:?}",
            from_shape.dims(),
            to_shape.dims()
        )));
    }

    let result_size = to_shape.numel();
    let mut result = vec![0.0; result_size];

    let from_dims = from_shape.dims();
    let to_dims = to_shape.dims();

    // Handle broadcasting by expanding dimensions
    let dim_offset = to_dims.len() - from_dims.len();

    for (i, result_val) in result.iter_mut().enumerate().take(result_size) {
        let mut from_idx = 0;
        let mut temp_i = i;

        for (dim_idx, &to_dim) in to_dims.iter().enumerate().rev() {
            let coord = temp_i % to_dim;
            temp_i /= to_dim;

            if dim_idx >= dim_offset {
                let from_dim_idx = dim_idx - dim_offset;
                let from_dim = from_dims[from_dim_idx];

                if from_dim == 1 {
                    // Broadcasting: use index 0 for this dimension
                } else {
                    let mut stride = 1;
                    for from_dim in from_dims.iter().skip(from_dim_idx + 1) {
                        stride *= from_dim;
                    }
                    from_idx += coord * stride;
                }
            }
        }

        *result_val = data[from_idx];
    }

    Ok(result)
}
