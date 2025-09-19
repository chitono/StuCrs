use ndarray::{
    array, s, Array, Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1,
    Axis, IxDyn,
};
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rand_distr::Normal;
use rand_distr::{Distribution, StandardNormal};

use mnist::*;

use ndarray::prelude::*;



pub trait Dataset {
    fn get_item(&self, index: i32) -> ArrayViewD<f32>;
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
    fn get_item(&self, index: i32) -> ArrayViewD<f32> {
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



#[derive(Clone)]
pub struct MNIST {
    pub train_img: Array3<f32>,
    pub train_label: Array2<f32>,
    pub test_img: Array3<f32>,
    pub test_label: Array2<f32>,

}

impl Dataset for MNIST {
    fn data_setup(&mut self) {}
    fn get_item(&self, index: i32) -> ArrayViewD<f32> {
        self.train_img.slice(s![index,..,..]).into_dyn()
    }
    fn len(&self) -> usize {
        self.train_img.shape()[0]
    }
}

impl MNIST {
    pub fn new() -> Self {
        let (train_img,train_label,test_img,test_label) = get_mnist_data();
        let mnist = Self {
            train_img: train_img,
            train_label:train_label,
            test_img:test_img,
            test_label:test_label
        };
        mnist
    }
}

fn get_mnist_data() -> (Array3<f32>, Array2<f32>,Array3<f32>,Array2<f32>) {

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

   let image_num = 0;
   // Can use an Array2 or Array3 here (Array3 for visualization)
   let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
       .expect("Error converting images to Array3 struct")
       .map(|x| *x as f32 / 256.0);
   //println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

   // Convert the returned Mnist struct to Array2 format
   let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
       .expect("Error converting training labels to Array2 struct")
       .map(|x| *x as f32);
   //println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

   let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
       .expect("Error converting images to Array3 struct")
       .map(|x| *x as f32 / 256.);

   let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
       .expect("Error converting testing labels to Array2 struct")
       .map(|x| *x as f32);
   (train_data,train_labels,test_data,test_labels)
    
    
    
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

pub fn to_one_hot(data: ArrayView1<u32>, num_class: usize) -> Array2<f32> {
    let mut init_matrix = Array2::zeros((data.len(), num_class));
    for i in 0..data.len() {
        let data_t = data[i];
        init_matrix[[i, data_t as usize]] = 1.0;
    }
    init_matrix
}

