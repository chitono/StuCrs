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
    model.stack(Dense::new(1000, true, None, Activation::Relu)?);
    model.stack(Dense::new(1000, true, None, Activation::Relu)?);
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
