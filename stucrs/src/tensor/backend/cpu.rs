use super::{Backend, Storage};
#[cfg(feature = "cpu")]
use crate::tensor::error::{Result, TensorError};
use crate::tensor::ndarray_nn::ndarray_cnn::{col2im_array, im2col_array};
use crate::tensor::shape::Shape;

use ndarray::{
    array, s, Array2, Array3, ArrayD, ArrayViewD, Axis, Dimension, Ix1, Ix2, Ix3, Ix4, IxDyn,
};
use ndarray_stats::QuantileExt;
use std::collections::HashSet;

#[derive(Debug)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Storage {
    fn to_ndarray(&self) -> Result<ArrayViewD<f32>> {
        match self {
            #[cfg(feature = "cpu")]
            Storage::Cpu(array) => Ok(array.view()),
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
        let lhs_data = lhs.to_ndarray()?.to_owned();
        let rhs_data = rhs.to_ndarray()?.to_owned();

        let result = lhs_data + rhs_data;

        Ok(Storage::Cpu(result))
    }

    fn sub(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?.to_owned();
        let rhs_data = rhs.to_ndarray()?.to_owned();

        let result: ArrayD<f32> = lhs_data - rhs_data;

        Ok(Storage::Cpu(result))
    }

    fn mul(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?.to_owned();
        let rhs_data = rhs.to_ndarray()?.to_owned();

        let result: ArrayD<f32> = lhs_data * rhs_data;

        Ok(Storage::Cpu(result))
    }

    fn div(&self, lhs: &Storage, rhs: &Storage) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?.to_owned();
        let rhs_data = rhs.to_ndarray()?.to_owned();

        let result = lhs_data / rhs_data;
        Ok(Storage::Cpu(result))
    }

    fn reshape(&self, storage: &Storage, new_shape: &Shape) -> Result<Storage> {
        let data = storage.to_ndarray()?;

        let result = data.to_shape(IxDyn::from(new_shape)).unwrap().to_owned();

        Ok(Storage::Cpu(result))
    }

    fn squeeze(&self, storage: &Storage, axis: usize) -> Result<Storage> {
        let data = storage.to_ndarray()?;

        let result = data.remove_axis(Axis(axis)).to_owned();
        println!("result_shape = {:?}", result.shape());
        Ok(Storage::Cpu(result))
    }

    fn unsqueeze(&self, storage: &Storage, axis: usize) -> Result<Storage> {
        let data = storage.to_ndarray()?;

        let result = data.insert_axis(Axis(axis)).to_owned();

        println!("result_shape = {:?}", result.shape());

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
            panic!(
                "broadcast変換不可:error対応予定,from_shape = {:?}, to_shape = {:?}",
                data.shape(),
                to_shape
            );
        };

        Ok(Storage::Cpu(result))
    }

    fn sum_to(&self, storage: &Storage, _from_shape: &Shape, to_shape: &Shape) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let x_shape = data.shape();
        let mut axes_to_sum = HashSet::new();

        // 合計する軸を特定する
        for i in 0..x_shape.len() {
            if i >= to_shape.ndim() || x_shape[i] != to_shape.dims()[i] {
                axes_to_sum.insert(i);
            }
        }

        let mut result = data.to_owned();

        let mut sorted_axes: Vec<_> = axes_to_sum.into_iter().collect();
        sorted_axes.sort_unstable();

        // 特定した軸を合計する
        for &axis in sorted_axes.iter().rev() {
            result = result.sum_axis(Axis(axis)).insert_axis(Axis(axis));
        }

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

    fn axis_slice(
        &self,
        storage: &Storage,
        _from_shape: &Shape,
        _to_shape: &Shape,
        axis: usize,
        indices: &[usize],
    ) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.select(Axis(axis), indices);

        Ok(Storage::Cpu(result))
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

    fn permute(
        &self,
        storage: &Storage,
        _from_shape: &Shape,
        _to_shape: &Shape,
        axes: &Vec<usize>,
    ) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.permuted_axes(axes.clone()).to_owned();

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

    fn tensordot(
        &self,
        lhs: &Storage,
        rhs: &Storage,
        _lhs_shape: &Shape,
        _rhs_shape: &Shape,
    ) -> Result<Storage> {
        let lhs_data = lhs.to_ndarray()?;
        let rhs_data = rhs.to_ndarray()?;

        let result = array_tensordot(&lhs_data.view(), &rhs_data.view());

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
        let result = -data.to_owned();
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

    fn max(
        &self,
        storage: &Storage,
        _shape: &Shape,
        _result_shape: &Shape,
        axis: Option<usize>,
    ) -> Result<Storage> {
        //TODO:ndarrayのerrorにも対応予定
        let data = storage.to_ndarray()?;

        let y_data = match axis {
            None => array![data.max().unwrap().clone()].into_dyn(),
            Some(axis) => data.map_axis(Axis(axis), |view| {
                view.iter().fold(f32::MIN, |acc, &x| acc.max(x))
            }),
        };

        Ok(Storage::Cpu(y_data))
    }

    fn argmax_to_max_backward(
        &self,
        storage: &Storage,
        _from_shape: &Shape,
        to_shape: &Shape,
        axis: usize,
    ) -> Result<Storage> {
        // TODO:処理要改善 map_axis()などを用いる
        let x_argmax_data = storage.to_ndarray()?;

        let shape = IxDyn(to_shape.dims());

        let mut result = ArrayD::<f32>::zeros(shape);

        for (idx, max_idx) in x_argmax_data.indexed_iter() {
            let mut full_idx = idx.as_array_view().to_vec();
            let max_usize = *max_idx as usize;
            full_idx.insert(axis, max_usize);

            result[IxDyn(&full_idx)] = 1.0f32;
        }

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
        let result = data.mapv(|x| if x <= 1.0f32 { 1.0f32 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn min_for_clamp_grad(&self, storage: &Storage) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.mapv(|x| if x >= 1.0e-4 { 1.0f32 } else { 0.0 });
        Ok(Storage::Cpu(result))
    }

    fn argmax_axis(
        &self,
        storage: &Storage,
        _shape: &Shape,
        _result_shape: &Shape,
        axis: usize,
    ) -> Result<Storage> {
        let data = storage.to_ndarray()?;
        let result = data.map_axis(Axis(axis), |view| {
            view.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        });

        let result = result.mapv(|x| x as f32);

        Ok(Storage::Cpu(result))
    }

    fn argmax_axis_2d(&self, storage: &Storage, _shape: &Shape, _axis: usize) -> Result<Storage> {
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

    fn im2col(
        &self,
        storage: &Storage,
        _shape: &Shape,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Storage> {
        // TODO:ndarrayのエラーにも対応予定
        let input = storage.to_ndarray()?.into_dimensionality::<Ix4>().unwrap();

        let result = im2col_array(input.view(), kernel_size, stride_size, pad_size);

        Ok(Storage::Cpu(result.into_dyn()))
    }

    fn col2im(
        &self,
        storage: &Storage,
        im_shape: [usize; 4],
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Result<Storage> {
        let input = storage.to_ndarray()?.into_dimensionality::<Ix3>().unwrap();

        let result = col2im_array(input.view(), im_shape, kernel_size, stride_size, pad_size);

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

fn array_tensordot(x_array: &ArrayViewD<f32>, w_array: &ArrayViewD<f32>) -> ArrayD<f32> {
    let y = match (x_array.ndim(), w_array.ndim()) {
        // 3D × 2D
        //(N,k,l) ×　(l,m) -> (N,k,m)
        (3, 2) => {
            let x = x_array.clone().into_dimensionality::<Ix3>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix2>().unwrap();
            if x.shape()[2] != w.shape()[0] {
                panic!("array_tensorの(3,2)での計算でxとwの次元が適合しません。")
            }
            let n = x.shape()[0];
            let k = x.shape()[1];
            let m = w.shape()[1];

            let mut y = Array3::<f32>::zeros((n, k, m));
            // xからバッチのように2次元の行列を取り出し、2次元の行列積
            for b in 0..n {
                let x_matrix = x.slice(s![b, .., ..]);
                let result = x_matrix.dot(&w);
                y.slice_mut(s![b, .., ..]).assign(&result);
            }
            y.into_dyn()
        }

        // 2D × 3D
        //(k,l) ×　(N,l,m) -> (N,k,m)
        (2, 3) => {
            let x = x_array.clone().into_dimensionality::<Ix2>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix3>().unwrap();

            if x.shape()[1] != w.shape()[1] {
                panic!("array_tensorの(2,3)での計算でxとwの次元が適合しません。")
            }
            let n = w.shape()[0];
            let k = x.shape()[0];
            let m = w.shape()[2];

            let mut y = Array3::<f32>::zeros((n, k, m));
            // xからバッチのように2次元の行列を取り出し、2次元の行列積
            for b in 0..n {
                let w_matrix = w.slice(s![b, .., ..]);
                let result = x.dot(&w_matrix);
                y.slice_mut(s![b, .., ..]).assign(&result);
            }
            y.into_dyn()
        }

        // 3D × 3D
        //(N,k,l) ×　(N,l,m) -> (N,k,m)
        (3, 3) => {
            let x = x_array.clone().into_dimensionality::<Ix3>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix3>().unwrap();

            // TODO: Error対応予定
            if x.shape()[2] != w.shape()[1] {
                panic!("array_tensorの(2,3)での計算でxとwの次元が適合しません。")
            }

            if x.shape()[0] != w.shape()[0] {
                panic!("array_tensorの(2,3)での計算でxとwの次元が適合しません。")
            }
            let n = x.shape()[0];
            let k = x.shape()[1];
            let m = w.shape()[2];

            let mut y = Array3::<f32>::zeros((n, k, m));
            // xからバッチのように2次元の行列を取り出し、2次元の行列積
            for b in 0..n {
                let x_matrix = x.slice(s![b, .., ..]);
                let w_matrix = w.slice(s![b, .., ..]);
                let result = x_matrix.dot(&w_matrix);
                y.slice_mut(s![b, .., ..]).assign(&result);
            }
            y.into_dyn()
        }

        _ => {
            panic!("4次元以上または2次元以下の行列積は未実装。");
        }
    };

    y
}

/// 行列の最大値のインデックスを返す。
/// 軸指定可能。
/// 1次元から3次元まで対応。
/// まだ一部の軸しか対応していない。
pub fn argmax_array(x_array: ArrayViewD<f32>, axis: usize) -> ArrayD<usize> {
    x_array.map_axis(Axis(axis), |view| {
        view.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    })
}
