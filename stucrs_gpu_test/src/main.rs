use ndarray::*;
//use ndarray_stats::QuantileExt;
//use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::*;
use stucrs_gpu::core_new::TensorToRcVariable;
use tensor_frame::*;

//use std::array;
//use std::cell::RefCell;
//use std::f32::consts::PI;
//use std::rc::Rc;
use std::time::Instant;
use stucrs_gpu::config;

//use stucrs::core_new::{F32ToRcVariable, RcVariable};
//use stucrs::dataloaders::DataLoader;
use stucrs::datasets::*;
use stucrs_gpu::functions_new::{self as F, accuracy};
use stucrs_gpu::layers::{self as L, Activation};
use stucrs_gpu::models::{BaseModel, Model};
use stucrs_gpu::optimizers::{Optimizer, SGD};

fn main() {
    let mnist = MNIST::new();
    let x_train = mnist.train_img.view();
    let y_train = mnist.train_label.view();
    let x_test = mnist.test_img.view();
    let y_test = mnist.test_label.view();

    //let image_num = 0;

    //println!("{:#.1?}\n", mnist.get_item(image_num));

    //println!("{:?}", x_train.shape());

    //println!("{:?}", y_train.shape());

    let x_train = x_train.to_shape((50000, 28 * 28)).unwrap();
    //let x_test = x_test.to_shape((10000, 28 * 28)).unwrap();

    let y_train = arr2d_to_one_hot(y_train.mapv(|x| x as u32).view(), 10);
    //let y_test = arr2d_to_one_hot(y_test.mapv(|x| x as u32).view(), 10);

    let x_train = Tensor::from_vec(x_train.clone().into_iter().collect(), x_train.shape()).unwrap();
    //let x_test = Tensor::from_vec(x_test.clone().into_iter().collect(), x_test.shape()).unwrap();
    let y_train = Tensor::from_vec(y_train.clone().into_iter().collect(), y_train.shape()).unwrap();
    //let y_test = Tensor::from_vec(y_test.clone().into_iter().collect(), y_test.shape()).unwrap();

    println!("{:?}", x_train.backend_type());

    //println!("faltten = {:?}", x_train.shape());
    //println!("faltten = {:?}", x_test.shape());

    let max_epoch = 5;
    let lr = 0.01;
    let batch_size = 1000;

    // 3x4の2次元行列を作成
    // `mut`キーワードで可変にする
    /*
    let train_spiral = Spiral::new(true);
    let test_spiral = Spiral::new(true);

    let x_train = train_spiral.data;
    let y_train = train_spiral.label.view();
    let y_train = to_one_hot(y_train, 3);

    let x_test = test_spiral.data;
    let y_test = test_spiral.label.view();
    let y_test = to_one_hot(y_test, 3);
    */

    let data_size = x_train.shape().dims()[0];
    println!("data_size={}", data_size);

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(1000, true, None, Activation::Relu));
    model.stack(L::Dense::new(1000, true, None, Activation::Relu));
    model.stack(L::Linear::new(10, true, None));

    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();
    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = Tensor::ones(vec![1]);
        let mut sum_acc = 0.0f32;

        /*
        let train_loader = DataLoader::new(
            x_train.clone().into_dyn(),
            y_train.clone().into_dyn(),
            batch_size,
            true,
        );
        let test_loader = DataLoader::new(
            x_test.clone().into_dyn(),
            y_test.clone().into_dyn(),
            batch_size,
            true,
        ); */

        //for (x_batch, y_batch) in train_loader {
        //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

        for chunk_indices in indices.chunks(batch_size) {
            let batch_indices: Vec<u32> = chunk_indices.iter().map(|&x| x as u32).collect();

            let x_batch = x_train.rows_slice(&batch_indices).unwrap().rv();
            let y_batch = y_train.rows_slice(&batch_indices).unwrap().rv();

            let y = model.call(&x_batch);

            //println!("y ={:?}", y.data().backend_type());

            let mut loss = F::softmax_cross_entropy_simple(&y, &y_batch);

            //println!("loss ={}", loss.data());

            /*
            let acc = accuracy(
                y.data().into_dimensionality().unwrap().view(),
                y_batch.data().into_dimensionality().unwrap().view(),
            ); */

            model.cleargrad();

            loss.backward(false);

            optimizer.update();

            //ここでt_batch.lenはu32からf32に変換、さらに暗黙的にndarray型に変換されて、計算される。
            //また、sum_lossは静的次元なので、epoch_lossを動的次元から静的次元に変換して足せるようにする。

            //sum_acc = sum_acc + acc * (y_batch.len() as f32);

            //let average_acc = sum_acc / (data_size as f32);

            //let epoch_loss = (loss.data() * (y_batch.data().shape().dims()[0])).unwrap();

            //sum_loss =&sum_loss + println!("epoch = {:?}, train_loss = {}", epoch + 1, loss.data());

            //config::set_grad_true();
        }
        //let average_loss = &sum_loss / (data_size as f32);
    }
    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}
