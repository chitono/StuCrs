use super::{Backend, Storage};
use crate::tensor::error::{Result, TensorError};
use crate::tensor::shape::Shape;

#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn zeros(&self, shape: &Shape) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![0.0; size]))
    }

    fn ones(&self, shape: &Shape) -> Result<Storage> {
        let size = shape.numel();
        Ok(Storage::Cpu(vec![1.0; size]))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        Ok(Storage::Cpu(data.to_vec()))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x + y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x - y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| x * y)
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;

        if lhs_data.len() != rhs_data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![lhs_data.len()],
                got: vec![rhs_data.len()],
            });
        }

        let result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(x, y)| {
                if *y == 0.0 {
                    // Division by zero - return appropriate IEEE floating point value
                    if *x == 0.0 {
                        f32::NAN // 0/0 = NaN
                    } else if *x > 0.0 {
                        f32::INFINITY // positive/0 = +inf
                    } else {
                        f32::NEG_INFINITY // negative/0 = -inf
                    }
                } else {
                    x / y
                }
            })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sum(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;

        match axis {
            None => {
                // Sum all elements
                let sum: f32 = data.iter().sum();
                Ok(Storage::Cpu(vec![sum]))
            }
            Some(axis_idx) => {
                // Sum along specific axis
                let dims = shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }

                // Calculate result shape (remove the summed axis)
                let mut result_shape = dims.to_vec();
                result_shape.remove(axis_idx);
                let result_size = if result_shape.is_empty() {
                    1
                } else {
                    result_shape.iter().product()
                };

                // Calculate strides for the original tensor
                let mut strides = vec![1; dims.len()];
                for i in (0..dims.len() - 1).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1];
                }

                let mut result = vec![0.0; result_size];

                // Iterate through all elements and accumulate along the specified axis
                for (linear_idx, &value) in data.iter().enumerate() {
                    // Convert linear index to multi-dimensional coordinates
                    let mut coords = vec![0; dims.len()];
                    let mut temp_idx = linear_idx;
                    for (i, &stride) in strides.iter().enumerate() {
                        coords[i] = temp_idx / stride;
                        temp_idx %= stride;
                    }

                    // Calculate result index by removing the summed axis coordinate
                    let mut result_coords = coords.clone();
                    result_coords.remove(axis_idx);

                    // Convert result coordinates to linear index
                    let mut result_idx = 0;
                    if !result_coords.is_empty() {
                        let mut result_strides = vec![1; result_coords.len()];
                        for i in (0..result_coords.len() - 1).rev() {
                            result_strides[i] = result_strides[i + 1] * result_shape[i + 1];
                        }
                        for (i, &coord) in result_coords.iter().enumerate() {
                            result_idx += coord * result_strides[i];
                        }
                    }

                    result[result_idx] += value;
                }

                Ok(Storage::Cpu(result))
            }
        }
    }

    fn broadcast_to(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
    ) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.powf(2.0)).collect();
        Ok(Storage::Cpu(result))
    }

    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        // Calculate sum first
        let sum_result = self.sum(storage, shape, axis)?;
        let sum_data = self.to_vec_f32(&sum_result)?;

        match axis {
            None => {
                // Mean of all elements
                let total_elements = shape.numel() as f32;
                let mean = sum_data[0] / total_elements;
                Ok(Storage::Cpu(vec![mean]))
            }
            Some(axis_idx) => {
                // Mean along specific axis
                let dims = shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }

                let axis_size = dims[axis_idx] as f32;
                let result: Vec<f32> = sum_data.iter().map(|&sum| sum / axis_size).collect();
                Ok(Storage::Cpu(result))
            }
        }
    }

    fn rows_slice(&self, storage: &Storage, shape: &Shape, indices: &[u32]) -> Result<Storage> {
        println!("cpuでのrow-sliceは未実装");
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.ln()).collect();
        Ok(Storage::Cpu(result))
    }
    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(TensorError::BackendError(
                "Transpose only supports 2D tensors".to_string(),
            ));
        }

        let data = self.to_vec_f32(storage)?;
        let rows = dims[0];
        let cols = dims[1];
        let mut result = vec![0.0; data.len()];

        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = data[i * cols + j];
            }
        }

        Ok(Storage::Cpu(result))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            #[cfg(feature = "cpu")]
            Storage::Cpu(data) => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => Err(TensorError::BackendError(
                "Cannot convert CUDA storage with CPU backend".to_string(),
            )),
        }
    }

    fn matmul(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();

        // Validate that tensors are 2D
        if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
            return Err(TensorError::InvalidShape(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        // Validate dimensions for matrix multiplication: (M, K) x (K, N) -> (M, N)
        let (m, k1) = (lhs_dims[0], lhs_dims[1]);
        let (k2, n) = (rhs_dims[0], rhs_dims[1]);

        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![k1],
                got: vec![k2],
            });
        }

        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;
        let mut result = vec![0.0; m * n];

        // Perform matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j])
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += lhs_data[i * k1 + k] * rhs_data[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Storage::Cpu(result))
    }

    fn bmm(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();

        // Validate that tensors are 3D
        if lhs_dims.len() != 3 || rhs_dims.len() != 3 {
            return Err(TensorError::InvalidShape(
                "Batched matrix multiplication requires 3D tensors".to_string(),
            ));
        }

        // Validate dimensions: (B, M, K) x (B, K, N) -> (B, M, N)
        let (b1, m, k1) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
        let (b2, k2, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

        if b1 != b2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![b1],
                got: vec![b2],
            });
        }

        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![k1],
                got: vec![k2],
            });
        }

        let lhs_data = self.to_vec_f32(lhs)?;
        let rhs_data = self.to_vec_f32(rhs)?;
        let mut result = vec![0.0; b1 * m * n];

        // Perform batched matrix multiplication
        for b in 0..b1 {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k1 {
                        let lhs_idx = b * m * k1 + i * k1 + k;
                        let rhs_idx = b * k1 * n + k * n + j;
                        sum += lhs_data[lhs_idx] * rhs_data[rhs_idx];
                    }
                    result[b * m * n + i * n + j] = sum;
                }
            }
        }

        Ok(Storage::Cpu(result))
    }

    fn neg(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| -x).collect();
        Ok(Storage::Cpu(result))
    }

    fn exp(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.exp()).collect();
        Ok(Storage::Cpu(result))
    }

    fn log(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.ln()).collect();
        Ok(Storage::Cpu(result))
    }

    fn sqrt(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Ok(Storage::Cpu(result))
    }

    fn pow(&self, storage: &Storage, power: f32) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.powf(power)).collect();
        Ok(Storage::Cpu(result))
    }

    fn sin(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.sin()).collect();
        Ok(Storage::Cpu(result))
    }

    fn cos(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.cos()).collect();
        Ok(Storage::Cpu(result))
    }

    fn relu(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        Ok(Storage::Cpu(result))
    }

    fn max_mask(&self, storage: &Storage, max: f32) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > max { 1.0 } else { 0.0 })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn min_mask(&self, storage: &Storage, min: f32) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x < min { 1.0 } else { 0.0 })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn mask_for_grad_relu(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn sigmoid(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(Storage::Cpu(result))
    }

    fn tanh(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
        Ok(Storage::Cpu(result))
    }

    fn sinh(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.sinh()).collect();
        Ok(Storage::Cpu(result))
    }

    fn cosh(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.cosh()).collect();
        Ok(Storage::Cpu(result))
    }

    fn clamp_max(&self, storage: &Storage, max: f32) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > max { max } else { x })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn clamp_min(&self, storage: &Storage, min: f32) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x < min { min } else { x })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn max_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 1.0f32 { 1.0f32 } else { 0.0 })
            .collect();
        Ok(Storage::Cpu(result))
    }

    fn min_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 1.0e-15 { 1.0f32 } else { 0.0 })
            .collect();
        Ok(Storage::Cpu(result))
    }

    // 仮の処理 coshx関数を使っている
    fn argmax_axis_2d(&self, storage: &Storage, shape: &Shape, axis: usize) -> Result<Storage> {
        println!("argmax_axis_2dの関数がcpuで処理されています。この処理は未実装です。");
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.cosh()).collect();
        Ok(Storage::Cpu(result))
    }

    // 仮の処理 coshx関数を使っている
    fn one_hot_encode(
        &self,
        storage: &Storage,
        shape: &Shape,
        num_class: usize,
    ) -> Result<Storage> {
        println!("argmax_axis_2dの関数がcpuで処理されています。この処理は未実装です。");
        let data = self.to_vec_f32(storage)?;
        let result: Vec<f32> = data.iter().map(|&x| x.cosh()).collect();
        Ok(Storage::Cpu(result))
    }
}
