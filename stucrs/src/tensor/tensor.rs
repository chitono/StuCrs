//! The main tensor module containing the core tensor type and its implementations.
//!
//! This module provides the [`Tensor`] struct which is the central data structure
//! in the library, along with its associated operations and traits.

use crate::tensor::backend::{Storage, BACKENDS};
use crate::tensor::error::{Result, TensorError};
//use broadcast::broadcast_data;
use crate::functions_cnn::get_conv_outsize;
use crate::tensor::ops::TensorOps;
use crate::tensor::shape::Shape;

use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};

use std::{fmt, vec};

/// A multi-dimensional array with support for various backends and operations.
///
/// The `Tensor` struct is the core data structure of this library. It consists of:
/// - `storage`: Backend-specific storage for the tensor data
/// - `shape`: The dimensions of the tensor
///
/// Tensors support various operations including element-wise arithmetic, broadcasting,
/// reductions, and shape manipulation. The actual computation is delegated to the
/// available backends (CPU, WGPU, or CUDA).
///
/// # Examples
///
/// ```
/// use tensor_frame::Tensor;
///
/// // Create a 2x3 tensor of ones
/// let tensor = Tensor::ones(vec![2, 3]).unwrap();
///
/// // Create from data
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
}

impl Tensor {
    /// Creates a new tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result` containing the new tensor or an error if no backend could create it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let zeros = Tensor::zeros(vec![3, 4]).unwrap();
    /// assert_eq!(zeros.shape().dims(), &[3, 4]);
    /// ```
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        for backend in &BACKENDS[0..] {
            match backend.zeros(&shape) {
                Ok(storage) => return Ok(Tensor { storage, shape }),
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create zeros tensor".to_string(),
        ))
    }

    /// Creates a new tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create
    ///
    /// # Returns
    ///
    /// A `Result` containing the new tensor or an error if no backend could create it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let ones = Tensor::ones(vec![2, 2]).unwrap();
    /// assert_eq!(ones.to_vec().unwrap(), vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn ones(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        for backend in &BACKENDS[0..] {
            match backend.ones(&shape) {
                Ok(storage) => return Ok(Tensor { storage, shape }),
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create ones tensor".to_string(),
        ))
    }

    /// Creates a new tensor from a vector of data with the specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to fill the tensor with
    /// * `shape` - The shape of the tensor
    ///
    /// # Returns
    ///
    /// A `Result` containing the new tensor or an error if:
    /// - The data length doesn't match the shape
    /// - No backend could create the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();
    /// assert_eq!(tensor.shape().dims(), &[2, 3]);
    /// ```
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        for backend in &BACKENDS[0..] {
            match backend.from_slice(&data, &shape) {
                Ok(storage) => return Ok(Tensor { storage, shape }),
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could create tensor from vector".to_string(),
        ))
    }

    pub fn standard_normal(shape: impl Into<Shape>) -> Result<Self> {
        let mut rng = thread_rng();
        let shape: Shape = shape.into();
        let numel = shape.numel();

        let vec: Vec<f32> = (0..numel)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let tensor = Tensor::from_vec(vec, shape)?;
        Ok(tensor)
    }

    /// Returns a reference to the tensor's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the number of dimensions of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3, 4]).unwrap();
    /// assert_eq!(tensor.ndim(), 3);
    /// ```
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3, 4]).unwrap();
    /// assert_eq!(tensor.numel(), 24);  // 2 * 3 * 4
    /// ```
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Creates a new tensor with explicit shape validation.
    ///
    /// This method validates the shape before creating the tensor, allowing for
    /// better error handling than the `From` trait implementations.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to fill the tensor with
    /// * `dims` - The dimensions of the tensor (will be validated)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(tensor.shape().dims(), &[2, 2]);
    ///
    /// // This will fail with proper error handling
    /// let result = Tensor::from_vec_with_shape(vec![1.0, 2.0], vec![0, 2]);
    /// assert!(result.is_err());
    /// ```
    pub fn from_vec_with_shape(data: Vec<f32>, dims: Vec<usize>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        Self::from_vec(data, shape)
    }

    /// Converts the tensor to a vector of f32 values.
    ///
    /// The data is returned in row-major (C-style) order.
    ///
    /// # Returns
    ///
    /// A `Result` containing the data as a vector or an error if no backend
    /// could perform the conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let data = tensor.to_vec().unwrap();
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        for backend in &BACKENDS[0..] {
            match backend.to_vec_f32(&self.storage) {
                Ok(vec) => return Ok(vec),
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could convert storage to Vec<f32>".to_string(),
        ))
    }

    /// Gets the backend type currently used by this tensor.
    ///
    /// # Returns
    ///
    /// A string identifying the backend type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 2]).unwrap();
    /// let backend_type = tensor.backend_type();
    /// // Returns "CPU", "CUDA", or "WGPU" depending on available backends
    /// ```
    pub fn backend_type(&self) -> &'static str {
        match &self.storage {
            #[cfg(feature = "cpu")]
            Storage::Cpu(_) => "CPU",
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => "CUDA",
        }
    }

    /// Attempts to move this tensor to a specific backend.
    ///
    /// This method tries to convert the tensor to use the specified backend.
    /// If the backend is not available or the conversion fails, returns an error.
    ///
    /// # Arguments
    ///
    /// * `backend_name` - The name of the target backend ("CPU", "CUDA", or "WGPU")
    ///
    /// # Returns
    ///
    /// A new tensor using the specified backend, or an error if conversion fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 2]).unwrap();
    /// // Try to move to CUDA backend (if available)
    /// match tensor.to_backend("CUDA") {
    ///     Ok(cuda_tensor) => println!("Successfully moved to CUDA"),
    ///     Err(e) => println!("Could not move to CUDA: {}", e),
    /// }
    /// ```
    pub fn to_backend(&self, backend_name: &str) -> Result<Self> {
        // If already on the requested backend, return a clone
        if self.backend_type() == backend_name {
            return Ok(self.clone());
        }

        // Get the data from current backend
        let data = self.to_vec()?;

        // Find the requested backend and create tensor with it
        for backend in &BACKENDS[0..] {
            match backend_name {
                "CPU" => {
                    #[cfg(feature = "cpu")]
                    if let Ok(storage) = backend.from_slice(&data, &self.shape) {
                        if matches!(storage, Storage::Cpu(_)) {
                            return Ok(Tensor {
                                storage,
                                shape: self.shape.clone(),
                            });
                        }
                    }
                    continue;
                }
                "CUDA" => {
                    #[cfg(feature = "cuda")]
                    if let Ok(storage) = backend.from_slice(&data, &self.shape) {
                        println!("strage = {:?}", storage);
                        if matches!(storage, Storage::Cuda(_)) {
                            return Ok(Tensor {
                                storage,
                                shape: self.shape.clone(),
                            });
                        }
                    }
                    continue;
                }

                _ => {
                    return Err(TensorError::BackendError(format!(
                        "Unknown backend: {backend_name}. Supported backends: CPU, CUDA"
                    )));
                }
            }
        }

        Err(TensorError::BackendError(format!(
            "Backend {backend_name} is not available or failed to create tensor"
        )))
    }

    /// Lists all available backends on the current system.
    ///
    /// # Returns
    ///
    /// A vector of backend names that are available and working.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_frame::Tensor;
    ///
    /// let available = Tensor::available_backends();
    /// println!("Available backends: {:?}", available);
    /// ```
    pub fn available_backends() -> Vec<String> {
        let mut available = Vec::new();

        for backend in &BACKENDS[0..] {
            if backend.is_available() {
                // Test create a small tensor to verify the backend works
                if let Ok(storage) = backend.zeros(&Shape::new(vec![1]).unwrap()) {
                    match storage {
                        #[cfg(feature = "cpu")]
                        Storage::Cpu(_) => available.push("CPU".to_string()),
                        #[cfg(feature = "cuda")]
                        Storage::Cuda(_) => available.push("CUDA".to_string()),
                    }
                }
            }
        }

        available
    }
}

impl Add for Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: Self) -> Self::Output {
        let self_shape = self.shape;
        let other_shape = other.shape;

        let result_shape = if self_shape == other_shape {
            self_shape.clone()
        } else if let Some(broadcasted_shape) = self_shape.broadcast_shape(&other_shape) {
            broadcasted_shape
        } else if let Some(broadcasted_shape) = other_shape.broadcast_shape(&self_shape) {
            broadcasted_shape
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: self_shape.dims().to_vec(),
                got: other_shape.dims().to_vec(),
            });
        };

        #[cfg(feature = "debug")]
        {
            println!(
                "Adding tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            );
            println!("Backend length: {}", BACKENDS.len());
        }

        // If shapes are the same, try backends directly
        if self_shape == other_shape {
            for backend in &BACKENDS[0..] {
                match backend.add(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self_shape,
                        });
                    }
                    Err(_) => continue,
                }
            }
        }

        // broadcastが生じる場合
        for backend in &BACKENDS[0..] {
            let self_storage = match &self.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &self.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if self_shape != result_shape {
                        &backend
                            .broadcast_to(&self.storage, &self_shape, &result_shape)
                            .unwrap()
                    } else {
                        &self.storage
                    }
                }
            };

            let other_storage = match &other.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &other.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if other_shape != result_shape {
                        &backend
                            .broadcast_to(&other.storage, &other_shape, &result_shape)
                            .unwrap()
                    } else {
                        &other.storage
                    }
                }
            };

            match backend.add(&self_storage, &other_storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform add operation".to_string(),
        ))
    }
}

impl Sub for Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: Self) -> Self::Output {
        let self_shape = self.shape;
        let other_shape = other.shape;

        let result_shape = if self_shape == other_shape {
            self_shape.clone()
        } else if let Some(broadcasted_shape) = self_shape.broadcast_shape(&other_shape) {
            broadcasted_shape
        } else if let Some(broadcasted_shape) = other_shape.broadcast_shape(&self_shape) {
            broadcasted_shape
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: self_shape.dims().to_vec(),
                got: other_shape.dims().to_vec(),
            });
        };

        #[cfg(feature = "debug")]
        {
            println!(
                "Adding tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            );
            println!("Backend length: {}", BACKENDS.len());
        }

        // If shapes are the same, try backends directly
        if self_shape == other_shape {
            for backend in &BACKENDS[0..] {
                match backend.sub(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self_shape,
                        });
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        // broadcastが生じる場合
        for backend in &BACKENDS[0..] {
            let self_storage = match &self.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &self.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if self_shape != result_shape {
                        &backend
                            .broadcast_to(&self.storage, &self_shape, &result_shape)
                            .unwrap()
                    } else {
                        &self.storage
                    }
                }
            };

            let other_storage = match &other.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &other.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if other_shape != result_shape {
                        &backend
                            .broadcast_to(&other.storage, &other_shape, &result_shape)
                            .unwrap()
                    } else {
                        &other.storage
                    }
                }
            };

            match backend.sub(&self_storage, &other_storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform sub operation".to_string(),
        ))
    }
}

impl Mul for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: Self) -> Self::Output {
        let self_shape = self.shape;
        let other_shape = other.shape;

        let result_shape = if self_shape == other_shape {
            self_shape.clone()
        } else if let Some(broadcasted_shape) = self_shape.broadcast_shape(&other_shape) {
            broadcasted_shape
        } else if let Some(broadcasted_shape) = other_shape.broadcast_shape(&self_shape) {
            broadcasted_shape
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: self_shape.dims().to_vec(),
                got: other_shape.dims().to_vec(),
            });
        };

        #[cfg(feature = "debug")]
        {
            println!(
                "Adding tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            );
            println!("Backend length: {}", BACKENDS.len());
        }

        // If shapes are the same, try backends directly
        if self_shape == other_shape {
            for backend in &BACKENDS[0..] {
                match backend.mul(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self_shape,
                        });
                    }
                    Err(_) => continue,
                }
            }
        }

        // broadcastが生じる場合
        for backend in &BACKENDS[0..] {
            let self_storage = match &self.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &self.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if self_shape != result_shape {
                        &backend
                            .broadcast_to(&self.storage, &self_shape, &result_shape)
                            .unwrap()
                    } else {
                        &self.storage
                    }
                }
            };

            let other_storage = match &other.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &other.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if other_shape != result_shape {
                        &backend
                            .broadcast_to(&other.storage, &other_shape, &result_shape)
                            .unwrap()
                    } else {
                        &other.storage
                    }
                }
            };

            match backend.mul(&self_storage, &other_storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform mul operation".to_string(),
        ))
    }
}

impl Div for Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: Self) -> Self::Output {
        let self_shape = self.shape;
        let other_shape = other.shape;

        let result_shape = if self_shape == other_shape {
            self_shape.clone()
        } else if let Some(broadcasted_shape) = self_shape.broadcast_shape(&other_shape) {
            broadcasted_shape
        } else if let Some(broadcasted_shape) = other_shape.broadcast_shape(&self_shape) {
            broadcasted_shape
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: self_shape.dims().to_vec(),
                got: other_shape.dims().to_vec(),
            });
        };

        #[cfg(feature = "debug")]
        {
            println!(
                "Adding tensors with shapes {:?} and {:?}",
                self.shape, other.shape
            );
            println!("Backend length: {}", BACKENDS.len());
        }

        // If shapes are the same, try backends directly
        if self_shape == other_shape {
            for backend in &BACKENDS[0..] {
                match backend.div(&self.storage, &other.storage) {
                    Ok(storage) => {
                        return Ok(Tensor {
                            storage,
                            shape: self_shape,
                        });
                    }
                    Err(_) => continue,
                }
            }
        }

        // broadcastが生じる場合
        for backend in &BACKENDS[0..] {
            let self_storage = match &self.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &self.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if self_shape != result_shape {
                        &backend
                            .broadcast_to(&self.storage, &self_shape, &result_shape)
                            .unwrap()
                    } else {
                        &self.storage
                    }
                }
            };

            let other_storage = match &other.storage {
                #[cfg(feature = "cpu")]
                Storage::Cpu(_) => &other.storage,
                #[cfg(feature = "cuda")]
                Storage::Cuda(_) => {
                    if other_shape != result_shape {
                        &backend
                            .broadcast_to(&other.storage, &other_shape, &result_shape)
                            .unwrap()
                    } else {
                        &other.storage
                    }
                }
            };

            match backend.div(&self_storage, &other_storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform div operation".to_string(),
        ))
    }
}

impl Neg for Tensor {
    type Output = Result<Tensor>;

    fn neg(self) -> Self::Output {
        // broadcastが生じる場合
        for backend in &BACKENDS[0..] {
            match backend.neg(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }

        Err(TensorError::BackendError(
            "No backend could perform neg operation".to_string(),
        ))
    }
}

impl TensorOps for Tensor {
    fn sum(&self, axis: Option<usize>, keepdims: bool) -> Result<Self> {
        let mut result_no_keepdims_shape = Shape::scalar(); // keepdimsを考慮した形状をcudaに渡すと正しい処理が行えないため
        let result_shape = match axis {
            None => {
                if keepdims {
                    Shape::new(vec![1; self.shape.ndim()])?
                } else {
                    Shape::scalar()
                }
            }
            Some(axis_idx) => {
                let dims = self.shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }
                // Remove the summed axis from the shape
                let mut result_dims = dims.to_vec();
                result_dims.remove(axis_idx);
                result_no_keepdims_shape = Shape::new(result_dims.clone())?;
                if keepdims {
                    result_dims.insert(axis_idx, 1);
                }
                Shape::new(result_dims)?
            }
        };

        for backend in &BACKENDS[0..] {
            match backend.sum(
                &self.storage,
                &self.shape,
                &result_no_keepdims_shape,
                axis,
                keepdims,
            ) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sum operation".to_string(),
        ))
    }

    fn broadcast_to(&self, to_shape: Shape) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.broadcast_to(&self.storage, &self.shape, &to_shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: to_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform broadcast_to operation".to_string(),
        ))
    }

    fn sum_to(&self, to_shape: &Shape) -> Result<Self> {
        let x_shape = self.shape().dims();
        let mut axes_to_sum = HashSet::new();

        // 合計する軸を特定する
        for i in 0..x_shape.len() {
            if i >= to_shape.ndim() || x_shape[i] != to_shape.dims()[i] {
                axes_to_sum.insert(i);
            }
        }

        let mut result = self.clone();

        let mut sorted_axes: Vec<_> = axes_to_sum.into_iter().collect();
        sorted_axes.sort_unstable();

        // 特定した軸を合計する
        for &axis in sorted_axes.iter().rev() {
            result = result.sum(Some(axis), true)?;
        }

        Ok(result)
    }

    fn axis_slice(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        let mut current_shape_vec = self.shape().dims.clone();
        current_shape_vec[axis] = indices.len();
        let result_shape = Shape::new(current_shape_vec).unwrap();
        for backend in &BACKENDS[0..] {
            match backend.axis_slice(&self.storage, &self.shape, &result_shape, axis, &indices) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform axis_slice operation".to_string(),
        ))
    }

    fn mean(&self, axis: Option<usize>) -> Result<Self> {
        // Calculate the result shape (same logic as sum)
        let result_shape = match axis {
            None => Shape::scalar(),
            Some(axis_idx) => {
                let dims = self.shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }
                // Remove the averaged axis from the shape
                let mut result_dims = dims.to_vec();
                result_dims.remove(axis_idx);
                Shape::new(result_dims)?
            }
        };

        for backend in &BACKENDS[0..] {
            match backend.mean(&self.storage, &self.shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform mean operation".to_string(),
        ))
    }

    fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_shape = Shape::new(new_shape)?;
        if self.shape.numel() != new_shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self.shape.numel()],
                got: vec![new_shape.numel()],
            });
        }

        for backend in &BACKENDS[0..] {
            match backend.reshape(&self.storage, &new_shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: new_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform transpose operation".to_string(),
        ))
    }

    fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidShape(
                "Transpose only supports 2D tensors".to_string(),
            ));
        }
        for backend in &BACKENDS[0..] {
            match backend.transpose(&self.storage, &self.shape) {
                Ok(storage) => {
                    let dims = self.shape.dims();
                    let new_shape = Shape::new(vec![dims[1], dims[0]])?;
                    return Ok(Tensor {
                        storage,
                        shape: new_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform transpose operation".to_string(),
        ))
    }

    fn permuted_axes(&self, axes: &Vec<usize>) -> Result<Self> {
        if self.ndim() != axes.len() {
            return Err(TensorError::InvalidShape(
                "行列の次元数と入力された配置の長さが一致しません".to_string(),
            ));
        }
        for backend in &BACKENDS[0..] {
            match backend.permuted_axes(&self.storage, &self.shape, axes) {
                Ok(storage) => {
                    let dims = self.shape.dims();
                    let new_shape_vec: Vec<usize> = axes.iter().map(|&i| dims[i]).collect();
                    let new_shape = Shape::new(new_shape_vec)?;
                    return Ok(Tensor {
                        storage,
                        shape: new_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform permuted_axes operation".to_string(),
        ))
    }

    fn squeeze(&self, axis: Option<usize>) -> Result<Self> {
        let dims = self.shape.dims();
        let new_dims = if let Some(axis) = axis {
            if axis >= self.ndim() || dims[axis] != 1 {
                return Err(TensorError::InvalidShape(format!(
                    "Cannot squeeze axis {} with size {}",
                    axis, dims[axis]
                )));
            }
            dims.iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &d)| d)
                .collect()
        } else {
            dims.iter().filter(|&&d| d != 1).copied().collect()
        };

        let new_shape = Shape::new(new_dims)?;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
        })
    }

    fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(TensorError::InvalidShape(format!(
                "Axis {} out of range for {}D tensor",
                axis,
                self.ndim()
            )));
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(axis, 1);
        let new_shape = Shape::new(new_dims)?;
        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
        })
    }

    fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(TensorError::InvalidShape(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();

        // Validate dimensions: (M, K) × (K, N) → (M, N)
        if self_dims[1] != other_dims[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_dims[1]],
                got: vec![other_dims[0]],
            });
        }

        let result_shape = Shape::new(vec![self_dims[0], other_dims[1]])?;

        for backend in &BACKENDS[0..] {
            match backend.matmul(&self.storage, &other.storage, &self.shape, &other.shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform matrix multiplication".to_string(),
        ))
    }

    fn tensordot(&self, other: &Self) -> Result<Self> {
        let x_shape = self.shape().dims();
        let w_shape = other.shape().dims();
        let result_shape = match (self.ndim(), other.ndim()) {
            (3, 2) => {
                if x_shape[2] != w_shape[0] {
                    return Err(TensorError::InvalidShape(
                        "3次元のxと2次元のwの次元が適合しません。".to_string(),
                    ));
                } else {
                    Shape::new(vec![x_shape[0], x_shape[1], w_shape[1]])?
                }
            }
            (2, 3) => {
                if x_shape[1] != w_shape[1] {
                    return Err(TensorError::InvalidShape(
                        "2次元のxと3次元のwの次元が適合しません。".to_string(),
                    ));
                } else {
                    Shape::new(vec![w_shape[0], x_shape[0], w_shape[2]])?
                }
            }
            (3, 3) => {
                return Err(TensorError::UnimplementedTensor(
                    "3次元同士ののテンソル積は今後対応".to_string(),
                ));
            }
            _ => {
                return Err(TensorError::UnimplementedTensor(
                    "4次元以上のテンソル積は非対応".to_string(),
                ));
            }
        };

        for backend in &BACKENDS[0..] {
            match backend.tensordot(&self.storage, &other.storage, &self.shape, &other.shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform TensorDot".to_string(),
        ))
    }

    fn bmm(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(TensorError::InvalidShape(
                "Batched matrix multiplication requires 3D tensors".to_string(),
            ));
        }

        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();

        // Validate dimensions: (B, M, K) × (B, K, N) → (B, M, N)
        if self_dims[0] != other_dims[0] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_dims[0]],
                got: vec![other_dims[0]],
            });
        }

        if self_dims[2] != other_dims[1] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![self_dims[2]],
                got: vec![other_dims[1]],
            });
        }

        let result_shape = Shape::new(vec![self_dims[0], self_dims[1], other_dims[2]])?;

        for backend in &BACKENDS[0..] {
            match backend.bmm(&self.storage, &other.storage, &self.shape, &other.shape) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform batched matrix multiplication".to_string(),
        ))
    }

    fn neg(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.neg(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform neg operation".to_string(),
        ))
    }

    fn exp(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.exp(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform exp operation".to_string(),
        ))
    }

    fn log(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.log(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform log operation".to_string(),
        ))
    }

    fn sqrt(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sqrt(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sqrt operation".to_string(),
        ))
    }

    fn pow(&self, power: f32) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.pow(&self.storage, power) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform pow operation".to_string(),
        ))
    }

    fn sin(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sin(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sin operation".to_string(),
        ))
    }

    fn cos(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.cos(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform cos operation".to_string(),
        ))
    }

    fn relu(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.relu(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform relu operation".to_string(),
        ))
    }

    fn max_mask(&self, max: f32) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.max_mask(&self.storage, max) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform max_mask operation".to_string(),
        ))
    }

    fn min_mask(&self, min: f32) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.min_mask(&self.storage, min) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform min_mask operation".to_string(),
        ))
    }

    fn mask_for_grad_relu(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.mask_for_grad_relu(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform mask_for_grad_relu operation".to_string(),
        ))
    }

    fn sigmoid(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sigmoid(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sigmoid operation".to_string(),
        ))
    }

    fn tanh(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.tanh(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform tanh operation".to_string(),
        ))
    }

    fn sinh(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.sinh(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sinh operation".to_string(),
        ))
    }

    fn cosh(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.cosh(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform cosh operation".to_string(),
        ))
    }

    fn max(&self, axis: Option<usize>) -> Result<Self> {
        let result_shape = match axis {
            None => Shape::new(vec![1; self.shape.ndim()])?,

            Some(axis_idx) => {
                let dims = self.shape.dims();
                if axis_idx >= dims.len() {
                    return Err(TensorError::InvalidShape(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis_idx,
                        dims.len()
                    )));
                }
                // Remove the summed axis from the shape
                let mut result_dims = dims.to_vec();
                result_dims.remove(axis_idx);

                Shape::new(result_dims)?
            }
        };
        for backend in &BACKENDS[0..] {
            match backend.max(&self.storage, &self.shape, &result_shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform max operation".to_string(),
        ))
    }

    fn max_backward(&self, axis: Option<usize>) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.max_backward(&self.storage, &self.shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform max_backward operation".to_string(),
        ))
    }

    fn clamp_max(&self, max: f32) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.clamp_max(&self.storage, max) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform max operation".to_string(),
        ))
    }

    fn clamp_min(&self, min: f32) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.clamp_min(&self.storage, min) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform min operation".to_string(),
        ))
    }

    fn max_for_clamp_grad(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.max_for_clamp_grad(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform mask_for_grad_relu operation".to_string(),
        ))
    }

    fn min_for_clamp_grad(&self) -> Result<Self> {
        for backend in &BACKENDS[0..] {
            match backend.min_for_clamp_grad(&self.storage) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: self.shape.clone(),
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform mask_for_grad_relu operation".to_string(),
        ))
    }

    fn argmax_axis(&self, axis: usize) -> Result<Self> {
        let dims = self.shape.dims();
        let new_dims = if axis >= self.ndim() {
            return Err(TensorError::InvalidShape(format!(
                "Cannot argmax axis {} with size {}",
                axis, dims[axis]
            )));
        } else {
            dims.iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &d)| d)
                .collect()
        };

        let result_shape = Shape::new(new_dims)?;
        for backend in &BACKENDS[0..] {
            match backend.argmax_axis(&self.storage, &self.shape, &result_shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform argmax_axis operation".to_string(),
        ))
    }

    fn argmax_axis_2d(&self, axis: usize) -> Result<Self> {
        // Calculate the result shape
        let dims = self.shape.dims();
        if dims.len() > 2 {
            panic!("このメソッドは2")
        }
        let result_shape = match axis {
            0 => Shape::new(vec![1, dims[1]])?,
            1 => Shape::new(vec![dims[0], 1])?,
            _ => {
                panic!(
                    "Axis {} is out of bounds for tensor with 0 or 1 dimensions",
                    axis,
                );
            }
        };

        for backend in &BACKENDS[0..] {
            match backend.argmax_axis_2d(&self.storage, &self.shape, axis) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform sum operation".to_string(),
        ))
    }

    fn one_hot_encode(&self, num_class: usize) -> Result<Self> {
        let result_shape = Shape::new(vec![self.shape.dims()[0], num_class]).unwrap();
        for backend in &BACKENDS[0..] {
            match backend.one_hot_encode(&self.storage, &self.shape, num_class) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(e) => println!("{:?}", e),
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform one_hot_encode operation".to_string(),
        ))
    }

    fn im2col(
        &self,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Self> {
        let n = self.shape().dims()[0]; //バッチ数
        let c = self.shape().dims()[1]; //チャンネル数
        let h = self.shape().dims()[2]; //縦
        let w = self.shape().dims()[3]; //横

        let (kh, kw) = kernel_size;
        let (stride_h, stride_w) = stride_size;
        let (pad_h, pad_w) = pad_size;

        let (oh, ow) = get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));
        let result_shape = Shape::new(vec![n, c * kh * kw, oh * ow])?;
        for backend in &BACKENDS[0..] {
            match backend.im2col(
                &self.storage,
                &self.shape,
                kernel_size,
                stride_size,
                pad_size,
            ) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform im2col operation".to_string(),
        ))
    }

    fn col2im(
        &self,
        im_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Self> {
        let result_shape = Shape::new(im_shape.to_vec())?;
        for backend in &BACKENDS[0..] {
            match backend.col2im(
                &self.storage,
                &self.shape,
                im_shape,
                kernel_size,
                stride_size,
                pad_size,
            ) {
                Ok(storage) => {
                    return Ok(Tensor {
                        storage,
                        shape: result_shape,
                    });
                }
                Err(_) => continue,
            }
        }
        Err(TensorError::BackendError(
            "No backend could perform col2im operation".to_string(),
        ))
    }
}
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.to_vec().map_err(|_| fmt::Error)?;
        let shape = self.shape().dims();

        write!(f, "Tensor(")?;

        if shape.is_empty() {
            write!(f, "{:.4}", data[0])?;
        } else if shape.len() == 1 {
            write!(f, "[")?;
            for (i, &val) in data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{val:.4}")?;
            }
            write!(f, "]")?;
        } else if shape.len() == 2 {
            write!(f, "[")?;
            for row in 0..shape[0] {
                if row > 0 {
                    write!(f, ",\n       ")?;
                }
                write!(f, "[")?;
                for col in 0..shape[1] {
                    if col > 0 {
                        write!(f, ", ")?;
                    }
                    let idx = row * shape[1] + col;
                    write!(f, "{val:.4}", val = data[idx])?;
                }
                write!(f, "]")?;
            }
            write!(f, "]")?;
        } else {
            write!(f, "shape={shape:?}, data=[")?;
            let max_display = 8.min(data.len());
            for (i, &val) in data.iter().take(max_display).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{val:.4}")?;
            }
            if data.len() > max_display {
                write!(f, ", ...")?;
            }
            write!(f, "]")?;
        }

        write!(f, ", dtype=f32)")
    }
}
