use crate::error::{FrameError, FrameResult};
use crate::tensor::error::TensorError;
use crate::tensor::tensor::Tensor;
use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use rand_distr::StandardNormal;
use thiserror::Error;

use mnist::*;

pub trait Dataset {
    /*
    fn get_item(&self, index: i32) -> ArrayViewD<f32>;
    */
    fn len(&self) -> usize;
    fn data_setup(&mut self);
}

pub struct Spiral {
    pub data: Tensor,  //2次元
    pub label: Tensor, //1次元
}

impl Dataset for Spiral {
    fn data_setup(&mut self) {}

    /*
    fn get_item(&self, index: i32) -> ArrayViewD<f32> {
        self.data.view().into_dyn()
    }
    */
    fn len(&self) -> usize {
        self.data.shape().dims()[0]
    }
}

impl Spiral {
    pub fn new() -> FrameResult<Self> {
        let data_label = get_spiral_data()?;
        let data = data_label.0;
        let label = data_label.1;
        let spiral = Self {
            data: data,
            label: label,
        };
        Ok(spiral)
    }
}
/// Array型でデータを生成してから、Tensorに変換
fn get_spiral_data() -> FrameResult<(Tensor, Tensor)> {
    let data_len = 100;
    let num_class = 3;
    let input_dim = 2;

    let data_size = data_len * num_class;

    let mut x = Array2::zeros((data_size, input_dim));

    let mut t = Array1::zeros(data_size);
    //let normal_dist = Normal::new(0.0f32, 1.0).unwrap();

    for j in 0..num_class {
        for i in 0..data_len {
            let rate = i as f32 / data_len as f32;
            let radius = 1.0 * rate as f32;

            let mut rng = rand::thread_rng();
            let normal: f32 = rng.sample(StandardNormal);

            let theta = j as f32 * 4.0 + 4.0 * rate as f32 + normal * 0.2;

            let ix = data_len * j + i;
            let mut x_row_view = x.row_mut(ix);

            let row_array = array![radius as f32 * theta.sin(), radius as f32 * theta.cos()];

            x_row_view.assign(&row_array);
            t[ix] = j as u32;
        }
    }

    //(x, t)
    //println!("x = {:?}", t.clone());
    let (x_array, t_array) = double_matrix_shuffle_rows_immutable(x.view(), t.view());
    let x_shape = x_array.shape().to_vec();
    let t_shape = t_array.shape().to_vec();

    let t_f32_vec = t_array.iter().map(|value| *value as f32).collect();

    let x_tensor = Tensor::from_vec(x_array.iter().copied().collect(), x_shape)?;
    let t_tensor = Tensor::from_vec(t_f32_vec, t_shape)?;

    Ok((x_tensor, t_tensor))
}

#[derive(Clone)]
pub struct MNIST {
    pub train_img: Tensor,   //4次元
    pub train_label: Tensor, //2次元
    pub test_img: Tensor,    //4次元
    pub test_label: Tensor,  //2次元
}

impl Dataset for MNIST {
    fn data_setup(&mut self) {}

    /*
    fn get_item(&self, index: i32) -> ArrayViewD<f32> {
        self.train_img.slice(s![index, .., ..]).into_dyn()
    }
    */
    fn len(&self) -> usize {
        self.train_img.shape().dims()[0]
    }
}

impl MNIST {
    pub fn new() -> FrameResult<Self> {
        let (train_img, train_label, test_img, test_label) = get_mnist_data()?;
        let mnist = Self {
            train_img: train_img,
            train_label: train_label,
            test_img: test_img,
            test_label: test_label,
        };
        Ok(mnist)
    }
}

fn get_mnist_data() -> FrameResult<(Tensor, Tensor, Tensor, Tensor)> {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Tensor::from_vec(
        trn_img.iter().map(|x| *x as f32 / 256.0).collect(),
        vec![50_000, 1, 28, 28],
    )
    .map_err(|e| {
        FrameError::DatasetError(DatasetError::ConvertError {
            dataset: "MNIST",
            source: e,
        })
    })?;

    //println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels =
        Tensor::from_vec(trn_lbl.iter().map(|x| *x as f32).collect(), vec![50_000, 1]).map_err(
            |e| {
                FrameError::DatasetError(DatasetError::ConvertError {
                    dataset: "MNIST",
                    source: e,
                })
            },
        )?;
    //println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let test_data = Tensor::from_vec(
        tst_img.iter().map(|x| *x as f32 / 256.0).collect(),
        vec![10_000, 1, 28, 28],
    )
    .map_err(|e| {
        FrameError::DatasetError(DatasetError::ConvertError {
            dataset: "MNIST",
            source: e,
        })
    })?;

    let test_labels =
        Tensor::from_vec(tst_lbl.iter().map(|x| *x as f32).collect(), vec![10_000, 1]).map_err(
            |e| {
                FrameError::DatasetError(DatasetError::ConvertError {
                    dataset: "MNIST",
                    source: e,
                })
            },
        )?;
    Ok((train_data, train_labels, test_data, test_labels))
}

//一つ2次元の行列と一次元の行列の対となる行が同じ位置に来るようにシャッフルして新しい二つの行列を返す
pub fn double_matrix_shuffle_rows_immutable(
    arr1: ArrayView2<f32>,
    arr2: ArrayView1<u32>,
) -> (Array2<f32>, Array1<u32>) {
    if arr1.nrows() != arr2.len() {
        panic!("arr1とarr2の行列の行数が異なります")
    }
    // 1. 行のインデックスを作成 (0, 1, 2, ...arr1.nrows())

    let mut indices: Vec<usize> = (0..arr1.nrows()).collect();

    // 2. インデックスをシャッフル
    indices.shuffle(&mut thread_rng());
    //carprintln!("indeces{:?}", indices);

    /*
    for i in indices.iter() {
        //println!("i = {}", i);
        if &0 <= i && i <= &99 {
            println!("0");
        } else if &100 <= i && i <= &199 {
            println!("1");
        } else {
            println!("2");
        }
    } */

    let new_arr1 = arr1.select(Axis(0), &indices);
    let new_arr2 = arr2.select(Axis(0), &indices);

    (new_arr1.to_owned(), new_arr2.to_owned())
}

pub fn arr1d_to_one_hot(data: ArrayView1<u32>, num_class: usize) -> Array2<f32> {
    let mut init_matrix = Array2::zeros((data.len(), num_class));
    for i in 0..data.len() {
        let data_t = data[i];
        init_matrix[[i, data_t as usize]] = 1.0;
    }
    init_matrix
}

pub fn arr2d_to_one_hot(data: ArrayView2<u32>, num_class: usize) -> Array2<f32> {
    if data.shape()[1] != 1 {
        panic!("one_hotベクトルにしたい教師データの列数が1ではありません");
    }
    let mut init_matrix = Array2::zeros((data.shape()[0], num_class));
    for i in 0..data.shape()[0] {
        let data_t = data[[i, 0]];
        init_matrix[[i, data_t as usize]] = 1.0;
    }
    init_matrix
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("データセット:{dataset}でデータをTensorに変換できませんでした")]
    ConvertError {
        dataset: &'static str,
        #[source]
        source: TensorError,
    },
}
