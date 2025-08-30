use ndarray::{
    array, s, Array, Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1,
    Axis, IxDyn,
};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rand_distr::Normal;
use rand_distr::{Distribution, StandardNormal};

pub trait Dataset {
    fn get_item(&self, index: u32) -> ArrayViewD<f32>;
    fn len(&self) -> usize;
    fn data_setup(&mut self);
}

pub struct Spiral {
    train: bool,
    pub data: Array2<f32>,
    pub label: Array1<u32>,
}

impl Dataset for Spiral {
    fn data_setup(&mut self) {}
    fn get_item(&self, index: u32) -> ArrayViewD<f32> {
        self.data.view().into_dyn()
    }
    fn len(&self) -> usize {
        self.data.shape()[0]
    }
}

impl Spiral {
    pub fn new(train: bool) -> Self {
        let data_label = get_spiral_data(train);
        let data = data_label.0;
        let label = data_label.1;
        let spiral = Self {
            train: train,
            data: data,
            label: label,
        };
        spiral
    }
}

fn get_spiral_data(train: bool) -> (Array2<f32>, Array1<u32>) {
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
    double_matrix_shuffle_rows_immutable(x.view(), t.view())
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

pub fn to_one_hot(data: ArrayViewMut1<u32>, num_class: usize) -> Array2<f32> {
    let mut init_matrix = Array2::zeros((data.len(), num_class));
    for i in 0..data.len() {
        let data_t = data[i];
        init_matrix[[i, data_t as usize]] = 1.0;
    }
    init_matrix
}

pub struct MnistData {}
