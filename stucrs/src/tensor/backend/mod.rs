//! Backend abstraction for tensor operations.
//!
//! This module provides the backend trait and storage types that allow the tensor
//! library to support multiple compute backends (CPU, WGPU, CUDA). The backend
//! system is designed to be extensible and allows seamless switching between
//! different hardware accelerators.
//!
//! # Architecture
//!
//! - [`Backend`]: The trait that all backends must implement
//! - [`Storage`]: An enum that holds backend-specific storage implementations
//! - [`BACKENDS`]: A global list of available backends, initialized lazily
//!
//! Backends are selected automatically based on availability, with preference
//! given to GPU backends (CUDA, then WGPU) before falling back to CPU.

use crate::tensor::error::Result;
use crate::tensor::shape::Shape;
use ndarray::ArrayD;
use once_cell::sync::Lazy;
use std::fmt::Debug;
use std::sync::Arc;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

/// The main backend trait that all compute backends must implement.
///
/// This trait defines the interface for tensor operations across different
/// hardware backends. Each backend is responsible for:
/// - Creating tensors (zeros, ones, from data)
/// - Performing arithmetic operations
/// - Performing reductions and transformations
/// - Converting between storage formats
///
/// # Implementation Notes
///
/// Backends should be thread-safe (`Send + Sync`) and implement `Debug` for
/// diagnostic purposes. The `is_available` method allows backends to report
/// whether they can be used on the current system.
pub trait Backend: Debug + Send + Sync {
    /// Checks if this backend is available on the current system.
    ///
    /// Default implementation returns `true`. Backends should override this
    /// to perform actual availability checks (e.g., CUDA device presence).
    fn is_available(&self) -> bool {
        true
    }

    /// Creates a tensor filled with zeros.
    fn zeros(&self, shape: &Shape) -> Result<Storage>;

    /// Creates a tensor filled with ones.
    fn ones(&self, shape: &Shape) -> Result<Storage>;

    fn rand_uniform(&self, new_shape: &Shape) -> Result<Storage>;

    /// Creates a tensor from a slice of f32 values.
    #[allow(clippy::wrong_self_convention)]
    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage>;

    /// Performs element-wise addition.
    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;

    /// Performs element-wise subtraction.
    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;

    /// Performs element-wise multiplication.
    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;

    /// Performs element-wise division.
    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage>;

    fn reshape(&self, storage: &Storage, new_shape: &Shape) -> Result<Storage>;

    fn squeeze(&self, storage: &Storage, axis: usize) -> Result<Storage>;

    fn unsqueeze(&self, storage: &Storage, axis: usize) -> Result<Storage>;

    /// Computes the sum of elements along an optional axis.
    fn sum(
        &self,
        storage: &Storage,
        shape: &Shape,
        result_shape: &Shape,
        axis: Option<usize>,
        keepdims: bool,
    ) -> Result<Storage>;

    fn sum_to(&self, storage: &Storage, from_shape: &Shape, to_shape: &Shape) -> Result<Storage>;

    fn broadcast_to(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
    ) -> Result<Storage>;

    fn axis_slice(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
        axis: usize,
        indices: &[usize],
    ) -> Result<Storage>;

    /// Computes the mean of elements along an optional axis.
    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage>;

    /// Transposes a 2D tensor.
    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage>;

    fn permute(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
        axes: &Vec<usize>,
    ) -> Result<Storage>;

    /// Converts the storage to a vector of f32 values.
    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>>;

    /// Matrix multiplication for 2D tensors.
    fn matmul(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage>;

    /// Matrix multiplication for 3D tensors.
    fn tensordot(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage>;

    /// Batched matrix multiplication for 3D tensors.
    fn bmm(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage>;

    fn neg(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise exponential function.
    fn exp(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise natural logarithm.
    fn log(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise square root.
    fn sqrt(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise power function.
    fn pow(&self, storage: &Storage, power: f32) -> Result<Storage>;

    /// Element-wise sine function.
    fn sin(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise cosine function.
    fn cos(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise ReLU activation function.
    fn relu(&self, storage: &Storage) -> Result<Storage>;

    fn max_mask(&self, storage: &Storage, max: f32) -> Result<Storage>;

    fn min_mask(&self, storage: &Storage, min: f32) -> Result<Storage>;

    fn mask_for_grad_relu(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise sigmoid activation function.
    fn sigmoid(&self, storage: &Storage) -> Result<Storage>;

    /// Element-wise tanh activation function.
    fn tanh(&self, storage: &Storage) -> Result<Storage>;

    fn sinh(&self, storage: &Storage) -> Result<Storage>;

    fn cosh(&self, storage: &Storage) -> Result<Storage>;

    fn max(
        &self,
        storage: &Storage,
        shape: &Shape,
        result_shape: &Shape,
        axis: Option<usize>,
    ) -> Result<Storage>;

    fn argmax_to_max_backward(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
        axis: usize,
    ) -> Result<Storage>;

    fn clamp_max(&self, storage: &Storage, max: f32) -> Result<Storage>;

    fn clamp_min(&self, storage: &Storage, min: f32) -> Result<Storage>;

    fn max_for_clamp_grad(&self, storage: &Storage) -> Result<Storage>;

    fn min_for_clamp_grad(&self, storage: &Storage) -> Result<Storage>;

    fn argmax_axis(
        &self,
        storage: &Storage,
        shape: &Shape,
        result_shape: &Shape,
        axis: usize,
    ) -> Result<Storage>;

    fn argmax_axis_2d(&self, storage: &Storage, shape: &Shape, axis: usize) -> Result<Storage>;

    fn one_hot_encode(&self, storage: &Storage, shape: &Shape, num_class: usize)
        -> Result<Storage>;

    fn im2col(
        &self,
        storage: &Storage,
        shape: &Shape,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Storage>;

    fn col2im(
        &self,
        storage: &Storage,
        im_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Storage>;
}

/// Backend-specific storage for tensor data.
///
/// This enum wraps the different storage implementations for each backend.
/// The active variants depend on which features are enabled at compile time.
///
/// # Variants
///
/// - `Cpu`: CPU backend using a simple `Vec<f32>`
/// - `Cuda`: CUDA backend using GPU memory
/// - `Wgpu`: WebGPU backend for cross-platform GPU support
#[derive(Debug, Clone)]
pub enum Storage {
    #[cfg(feature = "cpu")]
    Cpu(ArrayD<f32>),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
/// CUDA-specific storage wrapper.
///
/// Holds a reference-counted CUDA buffer for GPU memory management.
pub struct CudaStorage {
    /// The underlying CUDA buffer containing f32 data.
    pub buffer: std::sync::Arc<cudarc::driver::CudaSlice<f32>>,
}

/// Global list of available backends, initialized lazily.
///
/// This static variable holds all available backends in order of preference:
/// 1. CUDA (if available and feature enabled)
/// 2. WGPU (if available and feature enabled)
/// 3. CPU (always available when feature enabled)
///
/// The tensor operations will try backends in this order until one succeeds.
/// This allows automatic fallback from GPU to CPU when GPU operations fail.
///
/// # Example
///
/// ```ignore
/// // Backends are automatically selected when creating tensors
/// let tensor = Tensor::zeros(vec![2, 3]).unwrap();
/// // This will use CUDA if available, then WGPU, then CPU
/// ```
pub static BACKENDS: Lazy<Vec<Arc<dyn Backend>>> = Lazy::new(|| {
    let mut backends: Vec<Arc<dyn Backend>> = Vec::new();
    #[cfg(feature = "cuda")]
    if cuda::is_available() {
        if let Ok(backend) = cuda::CudaBackend::new() {
            backends.push(Arc::new(backend) as Arc<dyn Backend>);
        }
    }

    #[cfg(feature = "cpu")]
    {
        backends.push(Arc::new(cpu::CpuBackend::new()) as Arc<dyn Backend>);
    }

    backends
});
