use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::*;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use stucrs::config;
use stucrs::core::{F32ToRcVariable, TensorToRcVariable};
use stucrs::datasets::SinCurve;
use stucrs::error::FrameResult;
use stucrs::functions::loss::mean_squared_error;
use stucrs::functions::neural_funcs::tensor_accuracy;
use stucrs::layers::{
    Activation, ActivationLayer, Conv2d, Dense, Dropout, Flatten, Linear, Maxpool2d,
};
use stucrs::models::{BaseModel, Model, SimpleRNN};
use stucrs::optimizers::{Optimizer, SGD};
use stucrs::tensor::lib::Tensor;
use stucrs::tensor::ops::TensorOps;

fn main() -> FrameResult<()> {
    let train_set = SinCurve::new(true).unwrap();
    let x_train = train_set.data;
    let t_train = train_set.label;

    println!("x_tain_shape = {:?}", x_train.shape());
    println!("t_tain_shape = {:?}", t_train.shape());

    let max_epoch = 300;
    let lr = 0.0005;
    //let batch_size = 128;

    let hidden_size = 100;
    let bptt_length = 30;

    let data_size = x_train.shape().dims()[0];
    println!("data_size={}", data_size);

    let mut model = SimpleRNN::new(hidden_size, 1)?;

    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();

    for epoch in 0..max_epoch {
        model.reset_state();
        let mut loss = 0.0f32.rv();
        let mut count = 0;

        for index in 0..data_size {
            let x_batch = x_train.axis_slice(0, &[index])?.rv();
            let t_batch = t_train.axis_slice(0, &[index])?.rv();

            //println!("x = {:?}", x_batch.data().shape());
            //println!("t = {:?}", t_batch.data().shape());

            let y = model.call(&x_batch)?;

            //println!("y = {}", y.data());

            //thread::sleep(Duration::from_secs_f32(0.1));

            loss = loss + mean_squared_error(&y, &t_batch)?;

            count += 1;

            //println!("count  = {}", count);

            if count % bptt_length == 0 || count == data_size {
                model.cleargrad();
                loss.backward(false)?;
                loss.unchain_backward()?;
                optimizer.update()?;
            }
        }

        let average_loss = loss.data().to_vec()?[0] / (count as f32);

        println!("epoch = {:?}, train_loss = {:?}", epoch + 1, average_loss,);

        /*

        //推論
        config::set_grad_false();
        config::set_test_flag_true();
        let test_data_size = x_test.shape().dims()[0];
        let mut indices: Vec<usize> = (0..test_data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_test.axis_slice(0, chunk_indices)?.rv();
            let y_batch = y_test.axis_slice(0, chunk_indices)?.rv();

            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch)?;
            let loss = softmax_cross_entropy_simple(&y, &y_batch)?;
            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;

            let epoch_loss = loss.data().to_vec()?[0] * (x_batch.len() as f32);

            sum_loss = &sum_loss + &epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (test_data_size as f32);
        let average_acc = sum_acc / (test_data_size as f32);

        println!(
            "epoch = {:?}, test_loss = {:?}, test_accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        config::set_grad_true();
        config::set_test_flag_false();
        */
    }

    use plotters::prelude::*;

    config::set_grad_false();
    config::set_test_flag_true();

    let xs_array: Array1<f32> = Array1::linspace(0.0, 4.0 * std::f32::consts::PI, 1000).cos();

    let xs_tensor = Tensor::from_vec(xs_array.iter().copied().collect(), vec![1000, 1])?;

    model.reset_state();
    let mut pred_list: Vec<f32> = Vec::new();

    for index in 0..1000 {
        let x = xs_tensor.axis_slice(0, &[index])?.rv();
        let y = model.call(&x)?;
        let y_f32 = y.data().to_vec()?[0];
        pred_list.push(y_f32);
    }

    println!("pred_list = {:?}", pred_list);

    let root = BitMapBackend::new("cos_pre_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("cos_pre", ("sans-serif", 40))
        .build_cartesian_2d(0.0f32..1000.0, -1.0f32..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            pred_list.iter().enumerate().map(|(i, &y)| (i as f32, y)),
            &RED,
        ))
        .unwrap();

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);

    Ok(())
}

/*
use rand::seq::SliceRandom;
use rand::*;
use std::time::Instant;
use stucrs::config;
use stucrs::core::TensorToRcVariable;
use stucrs::datasets::CifarTen;
use stucrs::error::FrameResult;
use stucrs::functions::loss::softmax_cross_entropy_simple;
use stucrs::functions::neural_funcs::tensor_accuracy;
use stucrs::layers::{
    Activation, ActivationLayer, Conv2d, Dense, Dropout, Flatten, Linear, Maxpool2d,
};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{MomentumSGD, Optimizer};
use stucrs::tensor::ops::TensorOps;

fn main() -> FrameResult<()> {
    let cifar10 = CifarTen::new(true).unwrap();
    let x_train = cifar10.train_img;
    let y_train = cifar10.train_label;
    let x_test = cifar10.test_img;
    let y_test = cifar10.test_label;

    println!("x_tain_shape = {:?}", x_train.shape());
    println!("y_tain_shape = {:?}", y_train.shape());

    let max_epoch = 20;
    let lr = 0.01;
    let batch_size = 128;

    let data_size = x_train.shape().dims()[0];
    println!("data_size={}", data_size);

    let mut model = BaseModel::new();

    model.stack(Conv2d::new(32, (5, 5), (1, 1), (0, 0), true)?);
    model.stack(ActivationLayer::new(Activation::Relu));
    model.stack(Maxpool2d::new((2, 2), (2, 2), (0, 0)));
    model.stack(Dropout::new(0.2f32));

    model.stack(Conv2d::new(64, (5, 5), (1, 1), (0, 0), true)?);
    model.stack(ActivationLayer::new(Activation::Relu));
    model.stack(Maxpool2d::new((2, 2), (2, 2), (0, 0)));
    model.stack(Dropout::new(0.2f32));

    model.stack(Flatten::new());
    model.stack(Dense::new(64, true, None, Activation::Relu)?);
    model.stack(Dropout::new(0.2f32));
    model.stack(Dense::new(32, true, None, Activation::Relu)?);
    model.stack(Linear::new(10, true, None)?);

    let mut optimizer = MomentumSGD::new(lr, 0.9);
    optimizer.setup(&model);
    let start = Instant::now();

    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_train.axis_slice(0, chunk_indices)?.rv();
            let y_batch = y_train.axis_slice(0, chunk_indices)?.rv();

            let y = model.call(&x_batch)?;

            let mut loss = softmax_cross_entropy_simple(&y, &y_batch)?;

            //println!("loss = {}", loss.data());

            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;
            model.cleargrad();
            loss.backward(false)?;

            optimizer.update()?;

            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);

            sum_loss = sum_loss + epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (data_size as f32);
        let average_acc = sum_acc / (data_size as f32);

        println!(
            "epoch = {:?}, train_loss = {:?}, accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        //推論
        config::set_grad_false();
        config::set_test_flag_true();
        let test_data_size = x_test.shape().dims()[0];
        let mut indices: Vec<usize> = (0..test_data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_test.axis_slice(0, chunk_indices)?.rv();
            let y_batch = y_test.axis_slice(0, chunk_indices)?.rv();

            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch)?;
            let loss = softmax_cross_entropy_simple(&y, &y_batch)?;
            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;

            let epoch_loss = loss.data().to_vec()?[0] * (x_batch.len() as f32);

            sum_loss = &sum_loss + &epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (test_data_size as f32);
        let average_acc = sum_acc / (test_data_size as f32);

        println!(
            "epoch = {:?}, test_loss = {:?}, test_accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        config::set_grad_true();
        config::set_test_flag_false();
    }
    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);

    Ok(())
}

    */

/*
use rand::seq::SliceRandom;
use rand::*;
use std::time::Instant;
use stucrs::config;
use stucrs::core::TensorToRcVariable;
use stucrs::datasets::{tensor2d_to_one_hot, MNIST};
use stucrs::error::FrameResult;
use stucrs::functions::loss::softmax_cross_entropy_simple;
use stucrs::functions::neural_funcs::tensor_accuracy;
use stucrs::layers::{Activation, Dense, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};
use stucrs::tensor::ops::TensorOps;

fn main() -> FrameResult<()> {
    let mnist = MNIST::new()?;
    let x_train = mnist.train_img;
    let y_train = mnist.train_label;
    let x_test = mnist.test_img;
    let y_test = mnist.test_label;

    println!("backend = {:?}", x_train.backend_type());

    let _image_num = 0;

    //println!("{:#.1?}\n", mnist.get_item(image_num));

    //println!("{:?}", x_train.shape());

    //println!("{:?}", y_train.shape());

    let x_train = x_train.reshape(vec![50000, 28 * 28])?;
    let x_test = x_test.reshape(vec![10000, 28 * 28])?;

    let y_train = tensor2d_to_one_hot(y_train, 10)?;
    let y_test = tensor2d_to_one_hot(y_test, 10)?;

    //println!("faltten = {:?}", x_train.shape());
    //println!("faltten = {:?}", x_test.shape());

    //println!("y_train = {:?}", y_train);

    let max_epoch = 5;
    let lr = 0.01;
    let batch_size = 100;

    let data_size = x_train.shape().dims()[0];
    println!("data_size={}", data_size);

    let mut model = BaseModel::new();
    model.stack(Dense::new(1000, false, None, Activation::Relu)?);
    model.stack(Dense::new(1000, true, None, Activation::Relu)?);
    //model.stack(Dense::new(1000, true, None, Activation::Relu)?);
    model.stack(Linear::new(10, false, None)?);

    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();
    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_train.axis_slice(0, chunk_indices)?.rv();
            //println!("x_batch_shape = {:?}", x_batch.data().shape());

            let y_batch = y_train.axis_slice(0, chunk_indices)?.rv();

            let y = model.call(&x_batch)?;

            //println!("y = {}", y.data());

            let mut loss = softmax_cross_entropy_simple(&y, &y_batch)?;

            //println!("loss = {}", loss.data());
            //println!("loss = {}", loss.data());
            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;
            model.cleargrad();

            loss.backward(false)?;

            //println!("loss2 = {}", loss.data());

            optimizer.update()?;

            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);

            sum_loss = sum_loss + epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (data_size as f32);
        let average_acc = sum_acc / (data_size as f32);

        println!(
            "epoch = {:?}, train_loss = {:?}, accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        //推論
        config::set_grad_false();
        let test_data_size = x_test.shape().dims()[0];
        let mut indices: Vec<usize> = (0..test_data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_test.axis_slice(0, chunk_indices)?.rv();
            let y_batch = y_test.axis_slice(0, chunk_indices)?.rv();

            let y = model.call(&x_batch)?;
            let loss = softmax_cross_entropy_simple(&y, &y_batch)?;
            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;

            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);

            sum_loss = &sum_loss + &epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (test_data_size as f32);
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

    Ok(())
}

*/

/*

use rand::seq::SliceRandom;
use rand::*;
use std::time::Instant;
use stucrs::config;
use stucrs::core::TensorToRcVariable;
use stucrs::datasets::{tensor2d_to_one_hot, MNIST};
use stucrs::error::FrameResult;
use stucrs::functions::loss::softmax_cross_entropy_simple;
use stucrs::functions::neural_funcs::tensor_accuracy;
use stucrs::layers::{Activation, ActivationLayer, Conv2d, Dense, Flatten, Linear, Maxpool2d};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};
use stucrs::tensor::ops::TensorOps;

fn main() -> FrameResult<()> {
    let mnist = MNIST::new()?;
    let x_train = mnist.train_img;
    let y_train = mnist.train_label;
    let x_test = mnist.test_img;
    let y_test = mnist.test_label;

    let _image_num = 0;

    //println!("{:#.1?}\n", mnist.get_item(image_num));

    //println!("{:?}", x_train.shape());

    //println!("{:?}", y_train.shape());

    let y_train = tensor2d_to_one_hot(y_train, 10)?;
    let y_test = tensor2d_to_one_hot(y_test, 10)?;

    let max_epoch = 1;
    let lr = 0.01;
    let batch_size = 100;

    let data_size = x_train.shape().dims()[0];
    println!("data_size={}", data_size);

    let mut model = BaseModel::new();
    model.stack(Conv2d::new(32, (5, 5), (1, 1), (0, 0), false)?);
    model.stack(ActivationLayer::new(Activation::Relu));
    model.stack(Conv2d::new(32, (5, 5), (1, 1), (0, 0), false)?);
    model.stack(ActivationLayer::new(Activation::Relu));
    model.stack(Maxpool2d::new((2, 2), (1, 1), (0, 0)));

    model.stack(Conv2d::new(64, (3, 3), (1, 1), (0, 0), false)?);
    model.stack(ActivationLayer::new(Activation::Relu));
    model.stack(Conv2d::new(64, (3, 3), (1, 1), (0, 0), false)?);
    model.stack(ActivationLayer::new(Activation::Relu));
    model.stack(Maxpool2d::new((2, 2), (2, 2), (0, 0)));

    model.stack(Flatten::new());

    model.stack(Dense::new(256, true, None, Activation::Relu)?);
    model.stack(Linear::new(10, true, None)?);

    let mut batch = 0;

    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);
    let start = Instant::now();

    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_train.axis_slice(0, chunk_indices)?.rv();
            let y_batch = y_train.axis_slice(0, chunk_indices)?.rv();

            let y = model.call(&x_batch)?;

            let mut loss = softmax_cross_entropy_simple(&y, &y_batch)?;
            /*
            println!("batch = {}", batch);
            batch = batch + 1;

            */

            println!("loss = {}", loss.data());

            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;
            model.cleargrad();
            loss.backward(false)?;

            optimizer.update()?;

            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);

            sum_loss = sum_loss + epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (data_size as f32);
        let average_acc = sum_acc / (data_size as f32);

        println!(
            "epoch = {:?}, train_loss = {:?}, accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        //推論
        config::set_grad_false();
        let test_data_size = x_test.shape().dims()[0];
        let mut indices: Vec<usize> = (0..test_data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let mut sum_loss = 0.0f32;
        let mut sum_acc = 0.0f32;

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_test.axis_slice(0, chunk_indices)?.rv();
            let y_batch = y_test.axis_slice(0, chunk_indices)?.rv();

            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch)?;
            let loss = softmax_cross_entropy_simple(&y, &y_batch)?;
            let acc = tensor_accuracy(&y.data(), &y_batch.data())?;

            let epoch_loss = loss.data().to_vec()?[0] * (y_batch.len() as f32);

            sum_loss = &sum_loss + &epoch_loss;
            sum_acc = sum_acc + acc * (y_batch.len() as f32);
        }

        let average_loss = sum_loss / (test_data_size as f32);
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

    Ok(())
}

*/

/*
use stucrs::datasets::SinCurve;

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3x4の2次元行列を作成
    // `mut`キーワードで可変にする

    let sincurve = SinCurve::new(true).unwrap();

    //let t = sincurve.label.to_vec().unwrap();
    let y = sincurve.data.to_vec().unwrap();

    let root = BitMapBackend::new("sin_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("sin", ("sans-serif", 40))
        .build_cartesian_2d(-0.0f32..1000.0, -1.0f32..1.0)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(
        y.iter().enumerate().map(|(i, &y)| (i as f32, y)),
        &RED,
    ))?;

    Ok(())
}
*/
