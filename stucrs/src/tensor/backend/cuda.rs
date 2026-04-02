use super::{Backend, CudaStorage, Storage};
use crate::error::{Result, TensorError};
use crate::tensor::shape::Shape;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};

use std::collections::HashMap;

#[derive(Debug)]
pub struct CudaBackend {
    context: std::sync::Arc<CudaContext>,

    kernels: HashMap<String, CudaFunction>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        {
            let context = CudaContext::new(0).map_err(|e| {
                TensorError::BackendError(format!("Failed to initialize CUDA: {}", e))
            })?;

            let kernels = Self::load_kernels(&context)?;

            Ok(CudaBackend { context, kernels })
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
                    "sum_axis0_kernel",
                    "sum_axis1_kernel",
                    "mean_kernel",
                    "broadcast_to_kernel",
                    "rows_slice_kernel",
                    "argmax_axis0_2d_kernel",
                    "argmax_axis1_2d_kernel",
                    "one_hot_encode_kernel",
                ],
            ),
            (
                "transform",
                include_str!("../kernels/transform.cu"),
                vec!["transpose_2d_kernel"],
            ),
            (
                "matmul",
                include_str!("../kernels/matmul.cu"),
                vec!["matmul_kernel", "matmul_tiled_kernel", "bmm_kernel"],
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

        let stream = self.context.default_stream();
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
                let stream = self.context.default_stream();
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
            let stream = self.context.default_stream();
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
            let stream = self.context.default_stream();
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

            let stream = self.context.default_stream();
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

    fn sum(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match axis {
                None => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        let stream = self.context.default_stream();
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

                Some(axis_idx) => {
                    // Sum all elements using CUDA kernel
                    let Storage::Cuda(cuda_storage) = storage else {
                        panic!("想定外のバックエンド: この関数はCUDA専用です");
                    };
                    {
                        let rows = shape.dims()[0];
                        let cols = shape.dims()[1];

                        let stream = self.context.default_stream();

                        let mut result_buf;

                        let kernel;
                        let grid_size;

                        if axis_idx == 0 {
                            result_buf = stream.alloc_zeros::<f32>(cols).map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;

                            kernel = self.kernels.get("sum_axis0_kernel").ok_or_else(|| {
                                TensorError::BackendError("sum_axis0_kernel not found".to_string())
                            })?;
                            grid_size = cols;
                        } else if axis_idx == 1 {
                            result_buf = stream.alloc_zeros::<f32>(rows).map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;
                            kernel = self.kernels.get("sum_axis1_kernel").ok_or_else(|| {
                                TensorError::BackendError("sum_axis1_kernel not found".to_string())
                            })?;

                            grid_size = rows;
                        } else {
                            panic!("axisは0か1のみ指定できます。");
                        }

                        let size = cuda_storage.buffer.len();
                        let block_size = 256;

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
                    let out_rows = to_shape.dims()[0];
                    let out_cols = to_shape.dims()[1];

                    let stream = self.context.default_stream();
                    let mut result_buf =
                        stream
                            .alloc_zeros::<f32>(out_rows * out_cols)
                            .map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;

                    let kernel = self.kernels.get("broadcast_to_kernel").ok_or_else(|| {
                        TensorError::BackendError("broadcast_to_kernel not found".to_string())
                    })?;

                    let size = cuda_storage.buffer.len();
                    let block_x = 16;
                    let block_y = 16;

                    let grid_x = (out_cols + block_x - 1) / block_x;
                    let grid_y = (out_rows + block_y - 1) / block_y;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, 1),
                        block_dim: (block_x as u32, block_y as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);
                    let in_rows = from_shape.dims()[0];
                    let in_cols = from_shape.dims()[1];

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&in_rows);
                    builder.arg(&in_cols);
                    builder.arg(&out_rows);
                    builder.arg(&out_cols);

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
                        let sum_result = self.sum(storage, shape, axis)?;

                        let Storage::Cuda(sum_storage) = sum_result else {
                            panic!("想定外のバックエンド: この関数はCUDA専用です");
                        };
                        let sum_data = self.to_vec_f32(&Storage::Cuda(sum_storage))?;
                        let mean_val = sum_data[0] / cuda_storage.buffer.len() as f32;

                        let stream = self.context.default_stream();
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
                    let sum_result = self.sum(storage, shape, Some(axis_idx))?;
                    let sum_data = self.to_vec_f32(&sum_result)?;
                    let axis_size = dims[axis_idx] as f32;
                    let result: Vec<f32> = sum_data.iter().map(|&sum| sum / axis_size).collect();

                    // Convert result back to CUDA storage
                    let stream = self.context.default_stream();
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
                let stream = self.context.default_stream();
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

    fn rows_slice(&self, storage: &Storage, shape: &Shape, indices: &[u32]) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let num_indices = indices.len();
                    let in_cols = shape.dims()[1];

                    let stream = self.context.default_stream();
                    let mut result_buf =
                        stream
                            .alloc_zeros::<f32>(num_indices * in_cols)
                            .map_err(|e| {
                                TensorError::BackendError(format!(
                                    "Failed to allocate CUDA result buffer: {}",
                                    e
                                ))
                            })?;

                    let kernel = self.kernels.get("rows_slice_kernel").ok_or_else(|| {
                        TensorError::BackendError("rows_slice_kernel not found".to_string())
                    })?;

                    let indices_gpu = stream.memcpy_stod(indices).unwrap();

                    let size = cuda_storage.buffer.len();
                    let block_x = 16;
                    let block_y = 16;

                    let grid_x = (in_cols + block_x - 1) / block_x;
                    let grid_y = (num_indices + block_y - 1) / block_y;

                    let cfg = LaunchConfig {
                        grid_dim: (grid_x as u32, grid_y as u32, 1),
                        block_dim: (block_x as u32, block_y as u32, 1),
                        shared_mem_bytes: 0,
                    };

                    let mut builder = stream.launch_builder(kernel);

                    builder.arg(cuda_storage.buffer.as_ref());
                    builder.arg(&mut result_buf);
                    builder.arg(&indices_gpu);
                    builder.arg(&num_indices);
                    builder.arg(&in_cols);

                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        TensorError::BackendError(format!(
                            "Failed to launch rows_slice kernel: {}",
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
                    let stream = self.context.default_stream();
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
                Storage::Cpu(data) => Ok(data.clone()),
                #[cfg(feature = "wgpu")]
                Storage::Wgpu(_) => Err(TensorError::BackendError(
                    "Cannot convert WGPU storage with CUDA backend".to_string(),
                )),
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

            // Convert inputs to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;
            let lhs_shape_single = Shape::new(vec![lhs_data.len()])?;
            let rhs_shape_single = Shape::new(vec![rhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &lhs_shape_single)?;
            let rhs_storage = self.from_slice(&rhs_data, &rhs_shape_single)?;

            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    let stream = self.context.default_stream();
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

            // Convert inputs to CUDA storage if needed
            let lhs_data = self.to_vec_f32(lhs)?;
            let rhs_data = self.to_vec_f32(rhs)?;
            let lhs_shape_single = Shape::new(vec![lhs_data.len()])?;
            let rhs_shape_single = Shape::new(vec![rhs_data.len()])?;
            let lhs_storage = self.from_slice(&lhs_data, &lhs_shape_single)?;
            let rhs_storage = self.from_slice(&rhs_data, &rhs_shape_single)?;

            match (&lhs_storage, &rhs_storage) {
                (Storage::Cuda(a), Storage::Cuda(b)) => {
                    let stream = self.context.default_stream();
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
                    let stream = self.context.default_stream();
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
                    let stream = self.context.default_stream();
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
                        TensorError::BackendError(format!("Failed to launch max_mask kernel: {}", e))
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
                    let stream = self.context.default_stream();
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
                        TensorError::BackendError(format!("Failed to launch min_mask kernel: {}", e))
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

    fn clamp_max(&self, storage: &Storage, max: f32) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.context.default_stream();
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
                    let stream = self.context.default_stream();
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

                        let stream = self.context.default_stream();
                        let mut result_buf = stream.alloc_zeros::<f32>(cols).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel = self.kernels.get("argmax_axis0_2d_kernel").ok_or_else(|| {
                            TensorError::BackendError("argmax_axis0_2d_kernel not found".to_string())
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
                            TensorError::BackendError(format!("Failed to launch argmax_axis0_2d kernel: {}", e))
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

                        let stream = self.context.default_stream();
                        
                        let mut result_buf = stream.alloc_zeros::<f32>(rows).map_err(|e| {
                            TensorError::BackendError(format!(
                                "Failed to allocate CUDA result buffer: {}",
                                e
                            ))
                        })?;

                        let kernel = self.kernels.get("argmax_axis1_2d_kernel").ok_or_else(|| {
                                TensorError::BackendError("argmax_axis1_2d_kernel not found".to_string())
                            })?;
            
                        let size = cuda_storage.buffer.len();
                        let block_size = 256;
                        let grid_size = (size+block_size-1)/block_size;

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
                            TensorError::BackendError(format!("Failed to launch argmax_axis1_2d kernel: {}", e))
                        })?;

                        Ok(Storage::Cuda(CudaStorage {
                            buffer: std::sync::Arc::new(result_buf),
                        }))
                    }
                }
                _ => {Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))}
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }





    fn one_hot_encode(&self, storage: &Storage, shape:&Shape,num_class: usize) -> Result<Storage> {
        #[cfg(feature = "cuda")]
        {
            match storage {
                Storage::Cuda(cuda_storage) => {
                    let stream = self.context.default_stream();
                    let n = shape.dims()[0];
                    let mut result_buf = stream
                        .alloc_zeros::<f32>(n*num_class)
                        .map_err(|e| {
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
                        TensorError::BackendError(format!("Failed to launch one_hot_encode kernel: {}", e))
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
                    self.one_hot_encode(&cuda_storage,&shape, num_class)
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        Err(TensorError::BackendError(
            "CUDA support not compiled in".to_string(),
        ))
    }

    
}
