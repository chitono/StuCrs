//use image::{GrayImage, Luma};
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
use std::vec;
use stucrs_gpu::config;

//use stucrs::core_new::{F32ToRcVariable, RcVariable};
//use stucrs::dataloaders::DataLoader;
use stucrs_gpu::datasets::*;
use stucrs_gpu::functions_new::{self as F, accuracy};
use stucrs_gpu::layers::{self as L, Activation};
use stucrs_gpu::models::{BaseModel, Model};
use stucrs_gpu::optimizers::{Optimizer, SGD};
use tensor_frame::{Shape, Tensor, TensorOps};

fn main() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let b = a.sum(Some(1)).unwrap();

    println!(
        "b = {:?}, shape = {:?}",
        b.to_vec().to_owned(),
        b.shape().dims()
    );
}

/*

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

            let epoch_loss: Array1<f32> = (&loss.data() * (y_batch.len() as f32))
                .into_dimensionality()
                .unwrap();

            sum_loss = &sum_loss + &epoch_loss;
            //sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = &sum_loss / (data_size as f32);
        //let average_acc = sum_acc / (data_size as f32);

        println!("epoch = {:?}, train_loss = {:?}", epoch + 1, average_loss,);

        config::set_grad_true();
    }










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
