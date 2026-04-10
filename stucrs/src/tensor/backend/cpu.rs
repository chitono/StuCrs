use std::result;

use super::{Backend, Storage};

#[cfg(feature = "cpu")]
use crate::tensor::error::{Result, TensorError};
use crate::tensor::ndarray_nn::ndarray_cnn::*;
use crate::tensor::shape::Shape;
use ndarray::{array, Array2, ArrayD, ArrayViewD, Axis, Ix1, Ix2, IxDyn};

#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Storage {
    fn to_ndarray(&self) -> Result<ArrayD<f32>> {
        match self {
            #[cfg(feature = "cpu")]
            Storage::Cpu(array) => Ok(array.clone()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => Err(TensorError::BackendError(
                "Cannot convert CUDA storage with cpu ndarray backend".to_string(),
            )),
        }
    }
}

impl Backend for CpuBackend {
    fn zeros(&self, shape: &Shape) -> Result<Storage> {
        Ok(Storage::Cpu(ArrayD::<f32>::zeros(IxDyn::from(shape))))
    }

    fn ones(&self, shape: &Shape) -> Result<Storage> {
        Ok(Storage::Cpu(ArrayD::<f32>::ones(IxDyn::from(shape))))
    }

    fn from_slice(&self, data: &[f32], shape: &Shape) -> Result<Storage> {
        let result = if let Ok(array) = ArrayD::from_shape_vec(IxDyn::from(shape), data.to_vec()) {
            array
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        };

        Ok(Storage::Cpu(result))
    }

    fn add(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?;
        let rhs_data = rhs.to_ndarray()?;

        let result = lhs_data + rhs_data;

        Ok(Storage::Cpu(result))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?;
        let rhs_data = rhs.to_ndarray()?;

        let result: ArrayD<f32> = lhs_data - rhs_data;

        Ok(Storage::Cpu(result))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?;
        let rhs_data = rhs.to_ndarray()?;

        let result: ArrayD<f32> = lhs_data * rhs_data;

        Ok(Storage::Cpu(result))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?;
        let rhs_data = rhs.to_ndarray()?;

        let result = lhs_data / rhs_data;
        Ok(Storage::Cpu(result))
    }

    fn sum(
        &self,
        storage: &Storage,
        shape: &Shape,
        _result_shape: &Shape,
        axis: Option<usize>,
        keepdims: bool,
    ) -> Result<Storage> {
        let data = storage.to_ndarray()?;

        let ndim = shape.ndim();

        let result = if let Some(axis) = axis {
            if ndim <= axis {
                todo!("指定されたaxisは入力された行列の次元を超えています error対応予定")
            }

            let y = data.sum_axis(Axis(axis));

            if keepdims {
                y.insert_axis(Axis(axis))
            } else {
                y
            }
        } else {
            let scalar = data.sum();
            if keepdims {
                let shape = vec![1; ndim];
                ArrayD::from_elem(shape, scalar)
            } else {
                array![scalar].into_dyn()
            }
        };

        println!("ndarrayのshape = {:?}", result.shape());

        Ok(Storage::Cpu(result))
    }

    fn broadcast_to(
        &self,
        storage: &Storage,
        _from_storage: &Shape,
        to_shape: &Shape,
    ) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = if let Some(array) = data.broadcast(IxDyn::from(to_shape)) {
            array.to_owned()
        } else {
            panic!("broadcast変換不可:error対応予定")
        };
        Ok(Storage::Cpu(result))
    }

    fn mean(&self, storage: &Storage, shape: &Shape, axis: Option<usize>) -> Result<Storage> {
        // Calculate sum first
        panic!("cpuでのmeanは未実装");
        let sum_result = self.sum(storage, shape, &Shape { dims: vec![1] }, axis, false)?;

        let sum_data = self.to_vec_f32(&sum_result)?;

        match axis {
            None => {
                // Mean of all elements
                let total_elements = shape.numel() as f32;
                let mean = sum_data[0] / total_elements;
                //Ok(Storage::Cpu(vec![mean]))
                Err(TensorError::BackendError("cpuでのmeanは未実装".to_string()))
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
                //Ok(Storage::Cpu(result))
                Err(TensorError::BackendError("cpuでのmeanは未実装".to_string()))
            }
        }
    }

    fn rows_slice(&self, storage: &Storage, shape: &Shape, indices: &[u32]) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.ln();
        //Ok(Storage::Cpu(result));
        Err(TensorError::BackendError(
            "cpuでのrow-sliceは未実装".to_string(),
        ))
    }
    fn transpose(&self, storage: &Storage, shape: &Shape) -> Result<Storage> {
        let dims = shape.dims();
        if dims.len() != 2 {
            return Err(TensorError::BackendError(
                "Transpose only supports 2D tensors".to_string(),
            ));
        }

        let data = storage.to_ndarray()?;
        let result = data.t().to_owned();

        Ok(Storage::Cpu(result))
    }

    fn to_vec_f32(&self, storage: &Storage) -> Result<Vec<f32>> {
        match storage {
            #[cfg(feature = "cpu")]
            Storage::Cpu(data) => Ok(data.iter().cloned().collect()),
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
        _lhs_shape: &Shape,
        _rhs_shape: &Shape,
    ) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?;
        let rhs_data = rhs.to_ndarray()?;

        let lhs_dims = lhs_data.shape();
        let rhs_dims = rhs_data.shape();

        // Validate that tensors are 2D
        if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
            return Err(TensorError::InvalidShape(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        // Validate dimensions for matrix multiplication: (M, K) x (K, N) -> (M, N)
        let (_m, k1) = (lhs_dims[0], lhs_dims[1]);
        let (k2, _n) = (rhs_dims[0], rhs_dims[1]);

        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![k1],
                got: vec![k2],
            });
        }

        let result = array_matmul(&lhs_data.view(), &rhs_data.view());

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

        //Ok(Storage::Cpu(result))
        Err(TensorError::BackendError("cpuでのbmmは未実装".to_string()))
    }

    fn neg(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = -data;
        Ok(Storage::Cpu(result))
    }

    fn exp(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.exp();
        Ok(Storage::Cpu(result))
    }

    fn log(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.ln();
        Ok(Storage::Cpu(result))
    }

    fn sqrt(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.sqrt();
        Ok(Storage::Cpu(result))
    }

    fn pow(&self, storage: &Storage, power: f32) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.powf(power);
        Ok(Storage::Cpu(result))
    }

    fn sin(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.sin();
        Ok(Storage::Cpu(result))
    }

    fn cos(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.cos();
        Ok(Storage::Cpu(result))
    }

    fn relu(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x > 0.0 { x } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn max_mask(&self, storage: &Storage, max: f32) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x > max { 1.0 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn min_mask(&self, storage: &Storage, min: f32) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x < min { 1.0 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn mask_for_grad_relu(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn sigmoid(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Ok(Storage::Cpu(result))
    }

    fn tanh(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| x.tanh());
        Ok(Storage::Cpu(result))
    }

    fn sinh(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| x.sinh());
        Ok(Storage::Cpu(result))
    }

    fn cosh(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| x.cosh());
        Ok(Storage::Cpu(result))
    }

    fn clamp_max(&self, storage: &Storage, max: f32) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x > max { max } else { x });
        Ok(Storage::Cpu(result))
    }

    fn clamp_min(&self, storage: &Storage, min: f32) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x < min { min } else { x });
        Ok(Storage::Cpu(result))
    }

    fn max_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x > 1.0f32 { 1.0f32 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn min_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x > 1.0e-15 { 1.0f32 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    // 仮の処理 coshx関数を使っている
    fn argmax_axis_2d(&self, storage: &Storage, shape: &Shape, axis: usize) -> Result<Storage> {
        println!("argmax_axis_2dの関数がcpuで処理されています。この処理は未実装です。");
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| x.cosh());
        Ok(Storage::Cpu(result))
    }

    fn one_hot_encode(
        &self,
        storage: &Storage,
        shape: &Shape,
        num_class: usize,
    ) -> Result<Storage> {
        let data = storage.to_ndarray()?;

        let mut result = Array2::zeros((data.shape()[0], num_class));
        match shape.ndim() {
            1 => {
                for i in 0..data.len() {
                let data_t = data[i];
                result[[i, data_t as usize]] = 1.0;
                }
            },
            2=> {
                for i in 0..data.shape()[0] {
                    let data_t = data[[i, 0]];
                    result[[i, data_t as usize]] = 1.0;
                }
            },
            _ => panic!("one_hot_encodeは2次元以下の行列に対応しています。1,2以外の次元の行列が入力されました。")
        }
        Ok(Storage::Cpu(result.into_dyn()))
    }
}

// ndarray用の処理をまとめた関数

fn array_matmul(x_array: &ArrayViewD<f32>, w_array: &ArrayViewD<f32>) -> ArrayD<f32> {
    let y = match (x_array.ndim(), w_array.ndim()) {
        // 1D × 1D → スカラー出力
        (1, 1) => {
            let x = x_array.clone().into_dimensionality::<Ix1>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix1>().unwrap();

            let y = x.dot(&w);
            ArrayD::from_elem(ndarray::IxDyn(&[]), y) // スカラーとして返す
        }

        // 2D × 1D
        (2, 1) => {
            let x = x_array.clone().into_dimensionality::<Ix2>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix1>().unwrap();
            let y = x.dot(&w);
            y.into_dyn()
        }

        // 1D × 2D
        (1, 2) => {
            let x = x_array.clone().into_dimensionality::<Ix1>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix2>().unwrap();
            let y = x.dot(&w);
            y.into_dyn()
        }

        // 2D × 2D
        (2, 2) => {
            let x = x_array.clone().into_dimensionality::<Ix2>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix2>().unwrap();
            let y = x.dot(&w);
            y.into_dyn()
        }

        _ => {
            panic!("3次元以上の行列積は未実装");
        }
    };

    y
}
