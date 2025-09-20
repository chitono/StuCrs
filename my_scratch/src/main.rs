use image::{GrayImage, Luma};
use ndarray::*;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::*;
use std::array;
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use std::time::Instant;
use stucrs::config;
use stucrs::core_new::ArrayDToRcVariable;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::dataloaders::DataLoader;
use stucrs::datasets::*;
use stucrs::functions_new::{self as F, accuracy, sum};
use stucrs::layers::{self as L, Activation, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

fn main() {
    let mnist = MNIST::new();
    let x_train = mnist.train_img.view();
    let y_train = mnist.train_label.view();
    let x_test = mnist.test_img.view();
    let y_test = mnist.test_label.view();

    let image_num = 0;

    //println!("{:#.1?}\n", mnist.get_item(image_num));

    //println!("{:?}", x_train.shape());

    //println!("{:?}", y_train.shape());

    let x_train = x_train.to_shape((50000, 28 * 28)).unwrap();
    let x_test = x_test.to_shape((10000, 28 * 28)).unwrap();

    let y_train = arr2d_to_one_hot(y_train.mapv(|x| x as u32).view(), 10);
    let y_test = arr2d_to_one_hot(y_test.mapv(|x| x as u32).view(), 10);

    //println!("faltten = {:?}", x_train.shape());
    //println!("faltten = {:?}", x_test.shape());

    //println!("y_train = {:?}", y_train);

    let max_epoch = 5;
    let lr = 0.01;
    let batch_size = 100;

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

    let data_size = x_train.shape()[0];
    println!("data_size={}", data_size);

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(1000, true, None, Activation::Sigmoid));
    model.stack(L::Linear::new(10, true, None));

    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();
    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = array![0.0f32];
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

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_train.select(Axis(0), chunk_indices).to_owned().rv();
            let y_batch = y_train.select(Axis(0), chunk_indices).to_owned().rv();

            //for (x_batch, y_batch) in train_loader {
            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch);

            let mut loss = F::softmax_cross_entropy_simple(&y, &y_batch);
            let acc = accuracy(
                y.data().into_dimensionality().unwrap().view(),
                y_batch.data().into_dimensionality().unwrap().view(),
            );
            model.cleargrad();
            loss.backward(false);
            optimizer.update();

            //ここでt_batch.lenはu32からf32に変換、さらに暗黙的にndarray型に変換されて、計算される。
            //また、sum_lossは静的次元なので、epoch_lossを動的次元から静的次元に変換して足せるようにする。

            let epoch_loss: Array1<f32> = (&loss.data() * (y_batch.len() as f32))
                .into_dimensionality()
                .unwrap();

            sum_loss = &sum_loss + &epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = &sum_loss / (data_size as f32);
        let average_acc = sum_acc / (data_size as f32);

        println!(
            "epoch = {:?}, train_loss = {:?}, accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        //推論
        config::set_grad_false();
        let test_data_size = x_test.shape()[0];
        let mut indices: Vec<usize> = (0..test_data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = array![0.0f32];
        let mut sum_acc = array![0.0f32];

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_test.select(Axis(0), chunk_indices).to_owned().rv();
            let y_batch = y_test.select(Axis(0), chunk_indices).to_owned().rv();

            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch);
            let loss = F::softmax_cross_entropy_simple(&y, &y_batch);
            let acc = accuracy(
                y.data().into_dimensionality().unwrap().view(),
                y_batch.data().into_dimensionality().unwrap().view(),
            );

            let epoch_loss: Array1<f32> = (&loss.data() * (y_batch.len() as f32))
                .into_dimensionality()
                .unwrap();

            sum_loss = &sum_loss + &epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = &sum_loss / (test_data_size as f32);
        let average_acc = sum_acc / (test_data_size as f32);

        println!(
            "epoch = {:?}, test_loss = {:?}, test_accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        config::set_grad_true();
    }
    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}

/*
let root = BitMapBackend::new("plot.png", (640, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("spiral", ("sans-serif", 40))
        .build_cartesian_2d(-1.0f32..1.0, -1.0f32..1.0)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(
        spiral
            .data
            .rows()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                if t[index] == 0 {
                    Circle::new((row[0], row[1]), 5, RED.filled())
                } else if t[index] == 1 {
                    Circle::new((row[0], row[1]), 5, BLUE.filled())
                } else {
                    Circle::new((row[0], row[1]), 5, GREEN.filled())
                }
            }),
    )?;

    Ok(())

    for (x_batch,t_batch) in x.axis_chunks_iter(Axis(0), 10),t.axis_iter(Axis()) {
        println!("i = {:?} batch = {:?}", i, batch);

*/
