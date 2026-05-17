use super::{Backend, CudaStorage, Storage};
use crate::tensor::error::{Result, TensorError};
use crate::tensor::shape::Shape;

use crate::functions_cnn::get_conv_outsize;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
pub struct CudaBackend {
    //context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: HashMap<String, CudaFunction>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        {
            let context = CudaContext::new(0).map_err(|e| {
                TensorError::BackendError(format!("Failed to initialize CUDA: {}", e))
            })?;

            let stream = context.default_stream();

            let kernels = Self::load_kernels(&context)?;

            Ok(CudaBackend { stream, kernels })
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn load_kernels(
        context: &std::sync::Arc<CudaContext>,
    ) -> Result<HashMap<String, CudaFunction>> {
        let mut kernels = HashMap::new();

        // Define kernel files and their respective kernels
        let kernel_files = [
            (
                "fill",
                include_str!("../kernels/fill.cu"),
                vec!["fill_ones_kernel"],
            ),
            (
                "arithmetic",
                include_str!("../kernels/arithmetic.cu"),
                vec!["add_kernel", "sub_kernel", "mul_kernel", "div_kernel"],
            ),
            (
                "reduction",
                include_str!("../kernels/reduction.cu"),
                vec![
                    "sum_kernel",
                    "sum_axis_kernel",
                    "mean_kernel",
                    "broadcast_to_kernel",
                    "rows_slice_kernel",
                    "axis_slice_kernel",
                    "axis0_slice_kernel",
                    "argmax_axis_kernel",
                    "argmax_axis0_2d_kernel",
                    "argmax_axis1_2d_kernel",
                    "max_axis_kernel",
                    "argmax_to_max_backward_kernel",
                    "one_hot_encode_kernel",
                ],
            ),
            (
                "transform",
                include_str!("../kernels/transform.cu"),
                vec!["transpose_2d_kernel", "permute_kernel"],
            ),
            (
                "matmul",
                include_str!("../kernels/matmul.cu"),
                vec![
                    "matmul_kernel",
                    "matmul_tiled_kernel",
                    "bmm_kernel",
                    "tensordot_32_kernel",
                    "tensordot_23_kernel",
                ],
            ),
            (
                "math",
                include_str!("../kernels/math.cu"),
                vec![
                    "neg_kernel",
                    "exp_kernel",
                    "log_kernel",
                    "sqrt_kernel",
                    "pow_kernel",
                    "sin_kernel",
                    "cos_kernel",
                    "relu_kernel",
                    "max_mask_kernel",
                    "min_mask_kernel",
                    "mask_for_grad_relu_kernel",
                    "sigmoid_kernel",
                    "tanh_kernel",
                    "sinh_kernel",
                    "cosh_kernel",
                    "clamp_max_kernel",
                    "clamp_min_kernel",
                    "max_for_clamp_grad_kernel",
                    "min_for_clamp_grad_kernel",
                ],
            ),
            (
                "cnn",
                include_str!("../kernels/cnn.cu"),
                vec!["im2col_kernel", "col2im_kernel"],
            ),
        ];

        for (module_name, kernel_source, kernel_names) in &kernel_files {
            // Compile kernels using nvrtc
            let ptx = cudarc::nvrtc::compile_ptx(kernel_source).map_err(|e| {
                eprintln!("Failed to compile CUDA kernels in {}: {}", module_name, e);
                TensorError::BackendError(format!(
                    "Failed to compile CUDA kernels in {}: {}",
                    module_name, e
                ))
            })?;

            // Load the module using the correct API
            let module = context.load_module(ptx).map_err(|e| {
                eprintln!("Failed to load PTX module {}: {}", module_name, e);
                TensorError::BackendError(format!(
                    "Failed to load PTX module {}: {}",
                    module_name, e
                ))
            })?;

            for &name in kernel_names {
                let func = module.load_function(name).map_err(|e| {
                    eprintln!("Failed to get kernel {} from {}: {}", name, module_name, e);
                    TensorError::BackendError(format!(
                        "Failed to get kernel {} from {}: {}",
                        name, module_name, e
                    ))
                })?;
                kernels.insert(name.to_string(), func);
            }
        }

        println!(
            "Successfully loaded {} CUDA kernels from {} modules",
            kernels.len(),
            kernel_files.len()
        );
        Ok(kernels)
    }

    fn launch_binary_kernel(
        &self,
        kernel_name: &str,
        a: &CudaStorage,
        b: &CudaStorage,
    ) -> Result<Storage> {
        if a.buffer.len() != b.buffer.len() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![a.buffer.len()],
                got: vec![b.buffer.len()],
            });
        }

        let stream = self.stream.clone();
        let mut result_buf = stream.alloc_zeros::<f32>(a.buffer.len()).map_err(|e| {
            TensorError::BackendError(format!("Failed to allocate CUDA result buffer: {}", e))
        })?;

        let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
            TensorError::BackendError(format!("Kernel {} not found", kernel_name))
        })?;

        let size = a.buffer.len();
        let cfg = LaunchConfig::for_num_elems(size as u32);

        let mut builder = stream.launch_builder(kernel);
        builder.arg(a.buffer.as_ref());
        builder.arg(b.buffer.as_ref());
        builder.arg(&mut result_buf);
        let size_arg = size as i32;
        builder.arg(&size_arg);

        unsafe { builder.launch(cfg) }.map_err(|e| {
            TensorError::BackendError(format!("Failed to launch {} kernel: {}", kernel_name, e))
        })?;

        Ok(Storage::Cuda(CudaStorage {
            buffer: std::sync::Arc::new(result_buf),
        }))
    }

    fn launch_unary_math_kernel(&self, kernel_name: &str, storage: &Storage) -> Result<Storage> {
        match storage {
            Storage::Cuda(cuda_storage) => {
                let stream = self.stream.clone();
                let mut result_buf = stream
                    .alloc_zeros::<f32>(cuda_storage.buffer.len())
                    .map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
                    TensorError::BackendError(format!("{} not found", kernel_name))
                })?;

                let size = cuda_storage.buffer.len();
                let cfg = LaunchConfig::for_num_elems(size as u32);

                let mut builder = stream.launch_builder(kernel);
                builder.arg(cuda_storage.buffer.as_ref());
                builder.arg(&mut result_buf);
                let size_arg = size as i32;
                builder.arg(&size_arg);

                unsafe { builder.launch(cfg) }.map_err(|e| {
                    TensorError::BackendError(format!(
                        "Failed to launch {} kernel: {}",
                        kernel_name, e
                    ))
                })?;

                Ok(Storage::Cuda(CudaStorage {
                    buffer: std::sync::Arc::new(result_buf),
                }))
            }
            _ => {
                // Convert to CUDA and try again
                let data = self.to_vec_f32(storage)?;
                let shape = Shape::new(vec![data.len()])?;
                let cuda_storage = self.from_slice(&data, &shape)?;
                self.launch_unary_math_kernel(kernel_name, &cuda_storage)
            }
        }
    }
}

pub fn is_available() -> bool {
    match CudaContext::new(0) {
        Ok(_) => println!("CUDA context created!"),
        Err(e) => eprintln!("Failed to create CUDA context: {:?}", e),
    }
    CudaContext::new(0).is_ok()
}

impl Backend for CudaBackend {
    fn is_available(&self) -> bool {
        is_available()
    }

    fn zeros(&self, shape: &Shape) -> Result<Storage> {
        {
            let size = shape.numel();
            let stream = self.stream.clone();
            let buf = stream.alloc_zeros::<f32>(size).map_err(|e| {
                TensorError::BackendError(format!("Failed to allocate CUDA memory: {}", e))
            })?;
            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn ones(&self, shape: &Shape) -> Result<Storage> {
        {
            let size = shape.numel();
            let stream = self.stream.clone();
            let mut buf = stream.alloc_zeros::<f32>(size).map_err(|e| {
                TensorError::BackendError(format!("Failed to allocate CUDA memory: {}", e))
            })?;

            let kernel = self.kernels.get("fill_ones_kernel").ok_or_else(|| {
                TensorError::BackendError("fill_ones_kernel not found".to_string())
            })?;

            let cfg = LaunchConfig::for_num_elems(size as u32);

            let mut builder = stream.launch_builder(kernel);
            builder.arg(&mut buf);
            let size_arg = size as i32;
            builder.arg(&size_arg);

            unsafe { builder.launch(cfg) }.map_err(|e| {
                TensorError::BackendError(format!("Failed to launch fill_ones kernel: {}", e))
            })?;

            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        {
            if data.len() != shape.numel() {
                return Err(TensorError::ShapeMismatch {
                    expected: vec![shape.numel()],
                    got: vec![data.len()],
                });
            }

            let stream = self.stream.clone();
            let buf = stream.memcpy_stod(data).map_err(|e| {
                TensorError::BackendError(format!("Failed to copy data to CUDA: {}", e))
            })?;

            Ok(Storage::Cuda(CudaStorage {
                buffer: std::sync::Arc::new(buf),
            }))
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Now perform the operation
            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("add_kernel", a, b)
                }
                _ => {
                    panic!("Unsupported combination of storages");
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Now perform the operation
            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("sub_kernel", a, b)
                }
                _ => {
                    panic!("Unsupported combination of storages");
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Now perform the operation
            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("mul_kernel", a, b)
                }
                _ => {
                    panic!("Unsupported combination of storages");
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        {
            // Now perform the operation
            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    self.launch_binary_kernel("div_kernel", a, b)
                }
                _ => {
                    panic!("Unsupported combination of storages");
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn reshape(&self, storage: &Storage, _new_shape: &Shape) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        Ok(storage.clone())
    }

    fn squeeze(&self, storage: &Storage, _axis: usize) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        Ok(storage.clone())
    }

    fn unsqueeze(&self, storage: &Storage, _axis: usize) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        Ok(storage.clone())
    }

    fn sum(
        &self,
        storage: &Storage,
        shape: &Shape,
        result_shape: &Shape,
        axis: Option<usize>,
        _keepdims: bool,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match axis {
                None => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        let stream = self.stream.clone();
                        let mut result_buf = stream.alloc_zeros::<f32>(1).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel = self.kernels.get("sum_kernel").ok_or_else(|| {
                            TensorError::BackendError("sum_kernel not found".to_string())
                        })?;

                        let size = cuda_storage.buffer.len();
                        let block_size = 256;
                        let grid_size = (size + block_size - 1) / block_size;

                        let cfg = LaunchConfig {
                            grid_dim: (grid_size as u32, 1, 1),
                            block_dim: (block_size as u32, 1, 1),
                            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
                        };

                        let mut builder = stream.launch_builder(kernel);
                        builder.arg(cuda_storage.buffer.as_ref());
                        builder.arg(&mut result_buf);
                        let size_arg = size as i32;
                        builder.arg(&size_arg);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            TensorError::BackendError(format!("Failed to launch sum kernel: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }

                Some(axis) => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        let in_shape = &shape.dims;
                        let out_shape = &result_shape.dims;

                        let in_strides = shape.strides();
                        let out_strides = result_shape.strides();

                        let in_ndim = in_shape.len();
                        let out_ndim = out_shape.len();

                        let in_n = shape.numel();
                        let out_n = result_shape.numel();

                        let stream = self.stream.clone();

                        let mut result_buf = stream.alloc_zeros::<f32>(out_n).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel = self.kernels.get("sum_axis_kernel").ok_or_else(|| {
                            TensorError::BackendError("sum_axis_kernel not found".to_string())
                        })?;

                        let size = cuda_storage.buffer.len();
                        let block_size = 256;
                        let grid_size = (size + block_size - 1) / block_size;

                        let cfg = LaunchConfig {
                            grid_dim: (grid_size as u32, 1, 1),
                            block_dim: (block_size as u32, 1, 1),
                            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
                        };

                        let in_shape_i32: Vec<i32> = in_shape.iter().map(|&x| x as i32).collect();
                        let out_shape_i32: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();
                        let in_strides_i32: Vec<i32> =
                            in_strides.iter().map(|&x| x as i32).collect();
                        let out_strides_i32: Vec<i32> =
                            out_strides.iter().map(|&x| x as i32).collect();

                        let in_shape_buffer = stream.memcpy_stod(&in_shape_i32).unwrap();
                        let out_shape_buffer = stream.memcpy_stod(&out_shape_i32).unwrap();
                        let in_strides_buffer = stream.memcpy_stod(&in_strides_i32).unwrap();
                        let out_strides_buffer = stream.memcpy_stod(&out_strides_i32).unwrap();

                        let mut builder = stream.launch_builder(kernel);
                        builder.arg(cuda_storage.buffer.as_ref());
                        builder.arg(&mut result_buf);
                        builder.arg(&in_shape_buffer);
                        builder.arg(&out_shape_buffer);
                        builder.arg(&in_strides_buffer);
                        builder.arg(&out_strides_buffer);
                        builder.arg(&in_ndim);
                        builder.arg(&out_ndim);
                        builder.arg(&in_n);
                        builder.arg(&out_n);
                        builder.arg(&axis);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            TensorError::BackendError(format!("Failed to launch sum kernel: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sum_to(
        &self,
        _storage: &Storage,
        _from_shape: &Shape,
        _to_shape: &Shape,
    ) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA has not support sum_to yet".to_string(),
        ))
    }

    fn broadcast_to(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let mut in_shape = from_shape.dims.clone();
                    let out_shape = &to_shape.dims;

                    let in_strides = from_shape.strides();
                    let out_strides = to_shape.strides();

                    if to_shape.ndim() != from_shape.ndim() {
                        let offset = out_shape.len() - in_shape.len();
                        for _ in 0..offset {
                            in_shape.insert(0, 1);
                        }
                    }

                    let in_ndim = in_shape.len();
                    let out_ndim = out_shape.len();

                    let in_n = from_shape.numel();
                    let out_n = to_shape.numel();

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(out_n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("broadcast_to_kernel").ok_or_else(|| {
                        TensorError::BackendError("broadcast_to_kernel not found".to_string())
                    })?;

                    //let in_rows = from_shape.dims()[0];
                    //let in_cols = from_shape.dims()[1];

                    let size = cuda_storage.buffer.len();
                    //let block_x = 16;
                    //let block_y = 16;

                    let grid_x = (out_n + 256 - 1) / 256;
                    //let grid_y = (out_rows + block_y - 1) / block_y;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_shape_i32: Vec<i32> = in_shape.iter().map(|&x| x as i32).collect();
                    let out_shape_i32: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();
                    let in_strides_i32: Vec<i32> = in_strides.iter().map(|&x| x as i32).collect();
                    let out_strides_i32: Vec<i32> = out_strides.iter().map(|&x| x as i32).collect();

                    let in_shape_buffer = stream.memcpy_stod(&in_shape_i32).unwrap();
                    let out_shape_buffer = stream.memcpy_stod(&out_shape_i32).unwrap();
                    let in_strides_buffer = stream.memcpy_stod(&in_strides_i32).unwrap();
                    let out_strides_buffer = stream.memcpy_stod(&out_strides_i32).unwrap();

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&in_shape_buffer);
                    builder.arg(&out_shape_buffer);
                    builder.arg(&in_strides_buffer);
                    builder.arg(&out_strides_buffer);
                    builder.arg(&in_ndim);
                    builder.arg(&out_ndim);
                    builder.arg(&in_n);
                    builder.arg(&out_n);
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch broadcast_to kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match axis {
                None => {
                    // Mean of all elements
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        // sumの引数あとで直す
                        let sum_result = self.sum(storage, shape, &shape.clone(), axis, false)?;

                        let Storage::Cuda(sum_storage) = sum_result else {
                            panic!("想定外のバックエンド: この関数はCUDA専用です");
                        };
                        let sum_data = self.to_vec_f32(&Storage::Cuda(sum_storage))?;
                        let mean_val = sum_data[0] / cuda_storage.buffer.len() as f32;

                        let stream = self.stream.clone();
                        let result_buf = stream.memcpy_stod(&[mean_val]).map_err(|e| {
                            TensorError::BackendError(format!("Failed to copy mean to CUDA: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
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

                    // First calculate sum, then divide by axis size
                    // TODO:sumの引数を修正する
                    let sum_result =
                        self.sum(storage, shape, &shape.clone(), Some(axis_idx), false)?;
                    let sum_data = self.to_vec_f32(&sum_result)?;
                    let axis_size = dims[axis_idx] as f32;
                    let result: Vec<f32> = sum_data.iter().map(|&sum| sum / axis_size).collect();

                    // Convert result back to CUDA storage
                    let stream = self.stream.clone();
                    let result_buf = stream.memcpy_stod(&result).map_err(|e| {
                        TensorError::BackendError(format!("Failed to copy mean to CUDA: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        {
            let Storage::Cuda(cuda_storage) = storage else {
                panic!("想定外のバックエンド: この関数はCUDA専用です");
            };
            {
                let dims = shape.dims();
                if dims.len() != 2 {
                    return Err(TensorError::BackendError(
                        "Transpose only supports 2D tensors".to_string(),
                    ));
                }

                let rows = dims[0];
                let cols = dims[1];
                let stream = self.stream.clone();
                let mut result_buf = stream
                    .alloc_zeros::<f32>(cuda_storage.buffer.len())
                    .map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                let kernel = self.kernels.get("transpose_2d_kernel").ok_or_else(|| {
                    TensorError::BackendError("transpose_2d_kernel not found".to_string())
                })?;

                let block_dim_x = 16;
                let block_dim_y = 16;
                let grid_dim_x = (cols + block_dim_x - 1) / block_dim_x;
                let grid_dim_y = (rows + block_dim_y - 1) / block_dim_y;

                let cfg = LaunchConfig {
                    grid_dim: (grid_dim_x as u32, grid_dim_y as u32, 1),
                    block_dim: (block_dim_x as u32, block_dim_y as u32, 1),
                    shared_mem_bytes: 0,
                };

                let mut builder = stream.launch_builder(kernel);
                builder.arg(cuda_storage.buffer.as_ref());
                builder.arg(&mut result_buf);
                let rows_arg = rows as i32;
                let cols_arg = cols as i32;
                builder.arg(&rows_arg);
                builder.arg(&cols_arg);

                unsafe { builder.launch(cfg) }.map_err(|e| {
                    TensorError::BackendError(format!("Failed to launch transpose kernel: {}", e))
                })?;

                Ok(Storage::Cuda(CudaStorage {
                    buffer: std::sync::Arc::new(result_buf),
                }))
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    // TODO:permuted_axes cuda未対応
    fn permute(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
        axes: &Vec<usize>,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let numel = to_shape.numel();

                    let in_strides = from_shape.strides();
                    let out_strides = to_shape.strides();

                    let ndim = axes.len() as i32;

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(numel).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("permute_kernel").ok_or_else(|| {
                        TensorError::BackendError("permute_kernel not found".to_string())
                    })?;

                    let numel_i32 = numel as i32;

                    let in_strides_i32: Vec<i32> = in_strides.iter().map(|&x| x as i32).collect();
                    let out_strides_i32: Vec<i32> = out_strides.iter().map(|&x| x as i32).collect();
                    let axes_i32: Vec<i32> = axes.iter().map(|&x| x as i32).collect();

                    let in_strides_buffer = stream.memcpy_stod(&in_strides_i32).unwrap();
                    let out_strides_buffer = stream.memcpy_stod(&out_strides_i32).unwrap();
                    let axes_buffer = stream.memcpy_stod(&axes_i32).unwrap();

                    let block_x = 256;

                    let grid_x = (numel + block_x - 1) / block_x;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, 1, 1),
                        block_dim: (block_x as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&in_strides_buffer);
                    builder.arg(&out_strides_buffer);
                    builder.arg(&axes_buffer);
                    builder.arg(&ndim);
                    builder.arg(&numel_i32);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch permute kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again

                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }
    // TODO:axis_slice axis = 0の時のみcuda対応
    fn axis_slice(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
        _axis: usize,
        indices: &[usize],
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let num_indices = indices.len();
                    let in_n = from_shape.numel();
                    let out_n = to_shape.numel();

                    let inner_size = (in_n / from_shape.dims()[0]) as i32;

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(out_n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("axis0_slice_kernel").ok_or_else(|| {
                        TensorError::BackendError("axis_slice_kernel not found".to_string())
                    })?;

                    let indices_i32: Vec<i32> = indices.iter().map(|&x| x as i32).collect();

                    let indices_buffer = stream.memcpy_stod(&indices_i32).unwrap();

                    let block_x = 256;

                    let grid_x = (inner_size + block_x - 1) / block_x;
                    let grid_y = num_indices;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, 1),
                        block_dim: (block_x as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&inner_size);
                    builder.arg(&indices_buffer);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch axis_slice kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again

                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let mut result = vec![0.0f32; cuda_storage.buffer.len()];
                    stream
                        .memcpy_dtoh(cuda_storage.buffer.as_ref(), &mut result)
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to copy data from CUDA device: {}",
                                e
                            ))
                        })?;
                    Ok(result)
                }
                _ => {
                    panic!("Unsupported combination of storages");
                }
                #[cfg(feature = "cpu")]
                Storage::Cpu(data) => Ok(data.iter().cloned().collect()),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn matmul(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
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

            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    let stream = self.stream.clone();

                    let mut result_buf = stream.alloc_zeros::<f32>(m * n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    // Use tiled kernel for better performance
                    let kernel = self.kernels.get("matmul_tiled_kernel").ok_or_else(|| {
                        TensorError::BackendError("matmul_tiled_kernel not found".to_string())
                    })?;

                    let block_size = 16;
                    let grid_x = (n + block_size - 1) / block_size;
                    let grid_y = (m + block_size - 1) / block_size;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, 1),
                        block_dim: (block_size as u32, block_size as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(a.buffer.as_ref());
                    builder.arg(b.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    let m_arg = m as i32;
                    let k_arg = k1 as i32;
                    let n_arg = n as i32;
                    builder.arg(&m_arg);
                    builder.arg(&k_arg);
                    builder.arg(&n_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch matmul kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    panic!("Unsupported combination of storages");
                }
                _ => Err(TensorError::BackendError(
                    "Invalid storage types for CUDA matmul".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    // TODO:tensordot cuda未対応
    fn tensordot(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            let lhs_dims = lhs_shape.dims();
            let rhs_dims = rhs_shape.dims();

            let lhs_ndim = lhs_dims.len();
            let rhs_ndim = rhs_dims.len();

            /*

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

            */

            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    match (lhs_ndim, rhs_ndim) {
                        (3, 2) => {
                            // 3D × 2D
                            //(N,k,l) ×　(l,m) -> (N,k,m)
                            let (n, k, l1) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
                            let (l2, m) = (rhs_dims[0], rhs_dims[1]);

                            let numel = n * k * m;

                            let stream = self.stream.clone();
                            let mut result_buf = stream.alloc_zeros::<f32>(numel).map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;

                            let kernel =
                                self.kernels.get("tensordot_32_kernel").ok_or_else(|| {
                                    TensorError::BackendError(
                                        "tensordot_32_kernel not found".to_string(),
                                    )
                                })?;

                            let block_size = 16;
                            let grid_x = (m + block_size - 1) / block_size;
                            let grid_y = (k + block_size - 1) / block_size;
                            let grid_z = n;

                            let cfg = LaunchConfig {
                                grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
                                block_dim: (block_size as u32, block_size as u32, 1),
                                shared_mem_bytes: 0,
                            };

                            let (n_i32, k_i32, l_i32, m_i32) =
                                (n as i32, k as i32, l1 as i32, m as i32);

                            let mut builder = stream.launch_builder(kernel);
                            builder.arg(a.buffer.as_ref());
                            builder.arg(b.buffer.as_ref());
                            builder.arg(&mut result_buf);
                            builder.arg(&n_i32);
                            builder.arg(&k_i32);
                            builder.arg(&l_i32);
                            builder.arg(&m_i32);

                            unsafe { builder.launch(cfg) }.map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to launch tensordot32 kernel: {}",
                                    e
                                ))
                            })?;

                            Ok(Storage::Cuda(CudaStorage {
                                buffer: std::sync::Arc::new(result_buf),
                            }))
                        }

                        (2, 3) => {
                            // 2D × 3D
                            //(k,l) ×　(N,l,m) -> (N,k,m)

                            let (k, l1) = (lhs_dims[0], lhs_dims[1]);
                            let (n, l2, m) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

                            let numel = n * k * m;

                            let stream = self.stream.clone();

                            let mut result_buf = stream.alloc_zeros::<f32>(numel).map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;

                            let kernel =
                                self.kernels.get("tensordot_23_kernel").ok_or_else(|| {
                                    TensorError::BackendError(
                                        "sum_axis_kernel not found".to_string(),
                                    )
                                })?;

                            let block_size = 16;
                            let grid_x = (m + block_size - 1) / block_size;
                            let grid_y = (k + block_size - 1) / block_size;
                            let grid_z = n;

                            let cfg = LaunchConfig {
                                grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
                                block_dim: (block_size as u32, block_size as u32, 1),
                                shared_mem_bytes: 0,
                            };

                            let (n_i32, k_i32, l_i32, m_i32) =
                                (n as i32, k as i32, l1 as i32, m as i32);

                            let mut builder = stream.launch_builder(kernel);
                            builder.arg(a.buffer.as_ref());
                            builder.arg(b.buffer.as_ref());
                            builder.arg(&mut result_buf);
                            builder.arg(&n_i32);
                            builder.arg(&k_i32);
                            builder.arg(&l_i32);
                            builder.arg(&m_i32);

                            unsafe { builder.launch(cfg) }.map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to launch tensordot23 kernel: {}",
                                    e
                                ))
                            })?;

                            Ok(Storage::Cuda(CudaStorage {
                                buffer: std::sync::Arc::new(result_buf),
                            }))
                        }

                        (3, 3) => {
                            // 3D × 3D
                            //(N,k,l) ×　(N,l,m) -> (N,k,m)

                            let (n1, k, l1) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
                            let m = rhs_dims[2];

                            let numel = n1 * k * m;

                            let stream = self.stream.clone();

                            let mut result_buf = stream.alloc_zeros::<f32>(numel).map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;

                            let kernel = self.kernels.get("bmm_kernel").ok_or_else(|| {
                                TensorError::BackendError("bmm_kernel not found".to_string())
                            })?;

                            let block_size = 16;
                            let grid_x = (m + block_size - 1) / block_size;
                            let grid_y = (k + block_size - 1) / block_size;
                            let grid_z = n1;

                            let cfg = LaunchConfig {
                                grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
                                block_dim: (block_size as u32, block_size as u32, 1),
                                shared_mem_bytes: 0,
                            };

                            let (n_i32, k_i32, l_i32, m_i32) =
                                (n1 as i32, k as i32, l1 as i32, m as i32);

                            let mut builder = stream.launch_builder(kernel);
                            builder.arg(a.buffer.as_ref());
                            builder.arg(b.buffer.as_ref());
                            builder.arg(&mut result_buf);
                            builder.arg(&n_i32);
                            builder.arg(&k_i32);
                            builder.arg(&l_i32);
                            builder.arg(&m_i32);

                            unsafe { builder.launch(cfg) }.map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to launch sum kernel: {}",
                                    e
                                ))
                            })?;

                            Ok(Storage::Cuda(CudaStorage {
                                buffer: std::sync::Arc::new(result_buf),
                            }))
                        }

                        _ => {
                            return Err(TensorError::DimensionMismatch {
                                expected: 3,
                                got: 4,
                            });
                        }
                    }
                }
                _ => Err(TensorError::BackendError(
                    "Invalid storage types for CUDA bmm".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn bmm(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
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

            match (&lhs, &rhs) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(b1 * m * n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("bmm_kernel").ok_or_else(|| {
                        TensorError::BackendError("bmm_kernel not found".to_string())
                    })?;

                    let block_size = 16;
                    let grid_x = (n + block_size - 1) / block_size;
                    let grid_y = (m + block_size - 1) / block_size;
                    let grid_z = b1;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
                        block_dim: (block_size as u32, block_size as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(a.buffer.as_ref());
                    builder.arg(b.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    let batch_arg = b1 as i32;
                    let m_arg = m as i32;
                    let k_arg = k1 as i32;
                    let n_arg = n as i32;
                    builder.arg(&batch_arg);
                    builder.arg(&m_arg);
                    builder.arg(&k_arg);
                    builder.arg(&n_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch bmm kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => Err(TensorError::BackendError(
                    "Invalid storage types for CUDA bmm".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn neg(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("neg_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn exp(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("exp_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn log(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("log_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sqrt(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sqrt_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn pow(&self, storage: &Storage, power: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(cuda_storage.buffer.len())
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                    let kernel = self.kernels.get("pow_kernel").ok_or_else(|| {
                        TensorError::BackendError("pow_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&power);
                    let size_arg = size as i32;
                    builder.arg(&size_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!("Failed to launch pow kernel: {}", e))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, power)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sin(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sin_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn cos(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("cos_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn relu(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("relu_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn max_mask(&self, storage: &Storage, max: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(cuda_storage.buffer.len())
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                    let kernel = self.kernels.get("max_mask_kernel").ok_or_else(|| {
                        TensorError::BackendError("max_mask_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&max);
                    let size_arg = size as i32;
                    builder.arg(&size_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch max_mask kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.max_mask(&cuda_storage, max)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn min_mask(&self, storage: &Storage, min: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(cuda_storage.buffer.len())
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                    let kernel = self.kernels.get("min_mask_kernel").ok_or_else(|| {
                        TensorError::BackendError("min_mask_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&min);
                    let size_arg = size as i32;
                    builder.arg(&size_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch min_mask kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.min_mask(&cuda_storage, min)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn mask_for_grad_relu(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("mask_for_grad_relu", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sigmoid(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sigmoid_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn tanh(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("tanh_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn sinh(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("sinh_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn cosh(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("cosh_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    // TODO:max axis=Noneの対処未対応
    fn max(
        &self,
        storage: &Storage,
        shape: &Shape,
        result_shape: &Shape,
        axis: Option<usize>,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let in_shape = &shape.dims;
                    let out_shape = &result_shape.dims;

                    let in_strides = shape.strides();
                    let out_strides = result_shape.strides();

                    let in_ndim = in_shape.len();
                    let out_ndim = out_shape.len();

                    let in_n = shape.numel();
                    let out_n = result_shape.numel();
                    let axis = axis.unwrap_or(0) as i32;

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(out_n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("max_axis_kernel").ok_or_else(|| {
                        TensorError::BackendError("max_axis_kernel not found".to_string())
                    })?;

                    println!("sdfsdf");

                    //let in_rows = from_shape.dims()[0];
                    //let in_cols = from_shape.dims()[1];

                    //let block_x = 16;
                    //let block_y = 16;

                    let grid_x = (out_n + 256 - 1) / 256;
                    //let grid_y = (out_rows + block_y - 1) / block_y;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_shape_i32: Vec<i32> = in_shape.iter().map(|&x| x as i32).collect();
                    let out_shape_i32: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();
                    let in_strides_i32: Vec<i32> = in_strides.iter().map(|&x| x as i32).collect();
                    let out_strides_i32: Vec<i32> = out_strides.iter().map(|&x| x as i32).collect();

                    let in_shape_buffer = stream.memcpy_stod(&in_shape_i32).unwrap();
                    let out_shape_buffer = stream.memcpy_stod(&out_shape_i32).unwrap();
                    let in_strides_buffer = stream.memcpy_stod(&in_strides_i32).unwrap();
                    let out_strides_buffer = stream.memcpy_stod(&out_strides_i32).unwrap();

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&in_shape_buffer);
                    builder.arg(&out_shape_buffer);
                    builder.arg(&in_strides_buffer);
                    builder.arg(&out_strides_buffer);
                    builder.arg(&in_ndim);
                    builder.arg(&out_ndim);
                    builder.arg(&in_n);
                    builder.arg(&out_n);
                    builder.arg(&axis);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch argmax_axis kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn argmax_to_max_backward(
        &self,
        storage: &Storage,
        from_shape: &Shape,
        to_shape: &Shape,
        axis: usize,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let in_strides = from_shape.strides();
                    let out_strides = to_shape.strides();

                    println!("out_strides = {:?}", out_strides);

                    let in_ndim = from_shape.ndim();
                    let out_ndim = to_shape.ndim();

                    let out_n = to_shape.numel();
                    let axis = axis as i32;

                    let in_numel = from_shape.numel() as i32;

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(out_n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self
                        .kernels
                        .get("argmax_to_max_backward_kernel")
                        .ok_or_else(|| {
                            TensorError::BackendError(
                                "argmax_to_max_backward_kernel not found".to_string(),
                            )
                        })?;

                    //let in_rows = from_shape.dims()[0];
                    //let in_cols = from_shape.dims()[1];

                    //let block_x = 16;
                    //let block_y = 16;

                    let grid_x = (out_n + 256 - 1) / 256;
                    //let grid_y = (out_rows + block_y - 1) / block_y;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_strides_i32: Vec<i32> = in_strides.iter().map(|&x| x as i32).collect();
                    let out_strides_i32: Vec<i32> = out_strides.iter().map(|&x| x as i32).collect();

                    let in_strides_buffer = stream.memcpy_stod(&in_strides_i32).unwrap();
                    let out_strides_buffer = stream.memcpy_stod(&out_strides_i32).unwrap();

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&in_strides_buffer);
                    builder.arg(&out_strides_buffer);
                    builder.arg(&in_ndim);
                    builder.arg(&out_ndim);
                    builder.arg(&in_numel);
                    builder.arg(&axis);
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch argmax_axis kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn clamp_max(&self, storage: &Storage, max: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(cuda_storage.buffer.len())
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                    let kernel = self.kernels.get("clamp_max_kernel").ok_or_else(|| {
                        TensorError::BackendError("clamp_max_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&max);
                    let size_arg = size as i32;
                    builder.arg(&size_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch clamp_max kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.clamp_max(&cuda_storage, max)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn clamp_min(&self, storage: &Storage, min: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(cuda_storage.buffer.len())
                        .map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                    let kernel = self.kernels.get("clamp_min_kernel").ok_or_else(|| {
                        TensorError::BackendError("clamp_min_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&min);
                    let size_arg = size as i32;
                    builder.arg(&size_arg);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch clamp_min kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.clamp_min(&cuda_storage, min)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn max_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("max_for_clamp_grad_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn min_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            self.launch_unary_math_kernel("min_for_clamp_grad_kernel", storage)
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn argmax_axis(
        &self,
        storage: &Storage,
        shape: &Shape,
        result_shape: &Shape,
        axis: usize,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let in_shape = &shape.dims;
                    let out_shape = &result_shape.dims;

                    let in_strides = shape.strides();
                    let out_strides = result_shape.strides();

                    let in_ndim = in_shape.len();
                    let out_ndim = out_shape.len();

                    let in_n = shape.numel();
                    let out_n = result_shape.numel();
                    let axis = axis as i32;

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(out_n).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("argmax_axis_kernel").ok_or_else(|| {
                        TensorError::BackendError("argmax_axis_kernel not found".to_string())
                    })?;

                    //let in_rows = from_shape.dims()[0];
                    //let in_cols = from_shape.dims()[1];

                    //let block_x = 16;
                    //let block_y = 16;

                    let grid_x = (out_n + 256 - 1) / 256;
                    //let grid_y = (out_rows + block_y - 1) / block_y;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    let in_shape_i32: Vec<i32> = in_shape.iter().map(|&x| x as i32).collect();
                    let out_shape_i32: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();
                    let in_strides_i32: Vec<i32> = in_strides.iter().map(|&x| x as i32).collect();
                    let out_strides_i32: Vec<i32> = out_strides.iter().map(|&x| x as i32).collect();

                    let in_shape_buffer = stream.memcpy_stod(&in_shape_i32).unwrap();
                    let out_shape_buffer = stream.memcpy_stod(&out_shape_i32).unwrap();
                    let in_strides_buffer = stream.memcpy_stod(&in_strides_i32).unwrap();
                    let out_strides_buffer = stream.memcpy_stod(&out_strides_i32).unwrap();

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&in_shape_buffer);
                    builder.arg(&out_shape_buffer);
                    builder.arg(&in_strides_buffer);
                    builder.arg(&out_strides_buffer);
                    builder.arg(&in_ndim);
                    builder.arg(&out_ndim);
                    builder.arg(&in_n);
                    builder.arg(&out_n);
                    builder.arg(&axis);
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch argmax_axis kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn argmax_axis_2d(&self, storage: &Storage, shape: &Shape, axis: usize) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match axis {
                0 => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        let rows = shape.dims()[0];
                        let cols = shape.dims()[1];

                        let stream = self.stream.clone();
                        let mut result_buf = stream.alloc_zeros::<f32>(cols).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel =
                            self.kernels.get("argmax_axis0_2d_kernel").ok_or_else(|| {
                                TensorError::BackendError(
                                    "argmax_axis0_2d_kernel not found".to_string(),
                                )
                            })?;

                        let size = cuda_storage.buffer.len();
                        let block_size = 256;
                        let grid_size = (size + block_size - 1) / block_size;

                        let cfg = LaunchConfig {
                            grid_dim: (grid_size as u32, 1, 1),
                            block_dim: (block_size as u32, 1, 1),
                            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
                        };

                        let mut builder = stream.launch_builder(kernel);
                        builder.arg(cuda_storage.buffer.as_ref());
                        builder.arg(&mut result_buf);
                        let in_rows = rows as i32;
                        let in_cols = cols as i32;
                        builder.arg(&in_rows);
                        builder.arg(&in_cols);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to launch argmax_axis0_2d kernel: {}",
                                e
                            ))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }

                1 => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        let rows = shape.dims()[0];
                        let cols = shape.dims()[1];

                        let stream = self.stream.clone();

                        let mut result_buf = stream.alloc_zeros::<f32>(rows).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel =
                            self.kernels.get("argmax_axis1_2d_kernel").ok_or_else(|| {
                                TensorError::BackendError(
                                    "argmax_axis1_2d_kernel not found".to_string(),
                                )
                            })?;

                        let size = cuda_storage.buffer.len();
                        let block_size = 256;
                        let grid_size = (size + block_size - 1) / block_size;

                        let cfg = LaunchConfig {
                            grid_dim: (grid_size as u32, 1, 1),
                            block_dim: (block_size as u32, 1, 1),
                            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
                        };

                        let mut builder = stream.launch_builder(kernel);
                        builder.arg(cuda_storage.buffer.as_ref());
                        builder.arg(&mut result_buf);
                        let in_rows = rows as i32;
                        let in_cols = cols as i32;
                        builder.arg(&in_rows);
                        builder.arg(&in_cols);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to launch argmax_axis1_2d kernel: {}",
                                e
                            ))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }
                _ => Err(TensorError::BackendError(
                    "CUDA support not compiled in".to_string(),
                )),
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    fn one_hot_encode(
        &self,
        storage: &Storage,
        shape: &Shape,
        num_class: usize,
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.stream.clone();
                    let n = shape.dims()[0];
                    let mut result_buf = stream.alloc_zeros::<f32>(n * num_class).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("one_hot_encode_kernel").ok_or_else(|| {
                        TensorError::BackendError("one_hot_encode_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let cfg = LaunchConfig::for_num_elems(size as u32);

                    let mut builder = stream.launch_builder(kernel);
                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&n);
                    let num_class = num_class as i32;
                    builder.arg(&num_class);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch one_hot_encode kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again
                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.one_hot_encode(&cuda_storage, &shape, num_class)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    // TODO:im2col cuda未対応
    fn im2col(
        &self,
        storage: &Storage,
        shape: &Shape,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let n = shape.dims()[0]; //バッチ数
                    let c = shape.dims()[1]; //チャンネル数
                    let h = shape.dims()[2]; //縦
                    let w = shape.dims()[3]; //横

                    let (kh, kw) = kernel_size;
                    let (stride_h, stride_w) = stride_size;
                    let (pad_h, pad_w) = pad_size;

                    let (oh, ow) =
                        get_conv_outsize((h, w), (kh, kw), (stride_h, stride_w), (pad_h, pad_w));

                    let numel = n * c * kh * kw * oh * ow;

                    let stream = self.stream.clone();
                    let mut result_buf = stream.alloc_zeros::<f32>(numel).map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to allocate CUDA result buffer: {}",
                            e
                        ))
                    })?;

                    let kernel = self.kernels.get("im2col_kernel").ok_or_else(|| {
                        TensorError::BackendError("im2col_kernel not found".to_string())
                    })?;

                    let block_size = 16;

                    let grid_x = (oh * ow + block_size - 1) / block_size;
                    let grid_y = (c * kh * kw + block_size - 1) / block_size;
                    let grid_z = n;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, grid_z as u32),
                        block_dim: (block_size as u32, block_size as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let (n, c, h, w, kh, kw, oh, ow) = (
                        n as i32, c as i32, h as i32, w as i32, kh as i32, kw as i32, oh as i32,
                        ow as i32,
                    );

                    let (stride_h, stride_w, pad_h, pad_w) =
                        (stride_h as i32, stride_w as i32, pad_h as i32, pad_w as i32);

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&n);
                    builder.arg(&c);
                    builder.arg(&h);
                    builder.arg(&w);
                    builder.arg(&kh);
                    builder.arg(&kw);
                    builder.arg(&oh);
                    builder.arg(&ow);
                    builder.arg(&stride_h);
                    builder.arg(&stride_w);
                    builder.arg(&pad_h);
                    builder.arg(&pad_w);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch axis_slice kernel: {}",
                            e
                        ))
                    })?;

                    Ok(Storage::Cuda(CudaStorage {
                        buffer: std::sync::Arc::new(result_buf),
                    }))
                }
                _ => {
                    // Convert to CUDA and try again

                    let data = self.to_vec_f32(storage)?;
                    let shape = Shape::new(vec![data.len()])?;
                    let cuda_storage = self.from_slice(&data, &shape)?;
                    self.pow(&cuda_storage, 2.0)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    // TODO:col2im cuda未対応
    fn col2im(
        &self,
        storage: &Storage,
        _shape: &Shape,
        im_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Storage> {
        Err(TensorError::BackendError(
            "CUDA has not support col2im".to_string(),
        ))
    }
}
