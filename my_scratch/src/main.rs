use ndarray::*;
use std::array;
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::datasets::*;
use stucrs::functions_new::{self as F, sum};
use stucrs::layers::{self as L, Activation, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

use stucrs::core_new::ArrayDToRcVariable;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::*;




fn main() {
    let max_epoch = 300;
    let lr = 1.0;
    let batch_size = 30;
    // 3x4の2次元行列を作成
    // `mut`キーワードで可変にする
    let mut spiral = Spiral::new(true);

    let x = spiral.data.view();
    let t = spiral.label.view_mut();
    let t = to_one_hot(t, 3);

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(10, true, None, Activation::Sigmoid));
    model.stack(L::Linear::new(3, true, None));

    let data_size = x.shape()[0];
    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();
    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..x.shape()[0]).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
        let mut sum_loss = array![0.0f32];

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x.select(Axis(0), chunk_indices).rv();
            let t_batch = t.select(Axis(0), chunk_indices).rv();

            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch);
            let mut loss = F::softmax_cross_entropy_simple(&y, &t_batch);
            model.cleargrad();
            loss.backward(false);
            optimizer.update();

            //ここでt_batch.lenはu32からf32に変換、さらに暗黙的にndarray型に変換されて、計算される。
            //また、sum_lossは静的次元なので、epoch_lossを動的次元から静的次元に変換して足せるようにする。

            let epoch_loss: Array1<f32> = (&loss.data() * (t_batch.len() as f32))
                .into_dimensionality()
                .unwrap();

            sum_loss = &sum_loss + &epoch_loss;
        }

        let average_loss = &sum_loss / (data_size as f32);

        println!("epoch = {:?}, loss = {:?}", epoch + 1, average_loss);
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
