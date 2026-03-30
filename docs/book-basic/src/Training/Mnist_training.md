# MNISTの学習

ではいよいよ最後のMNISTの学習です。MNISTはよくニューラルネットワークの学習で試験的に用いられるデータセットで、これを学習できれば基本的な深層学習のフレームワークとして確立することができます。

## データの用意
MNISTのデータセットはwebからダウンロードし、解凍して利用する形です。その際、ライブラリクレートである **mnistライブラリ** を使用します。このデータの用意に関しては補足の[MNISTのデータの用意](../Supplement/Mnist_data.md)で説明します。内容としては、先ほどの説明した **Dataset構造体** として **Mnist構造体** を実装します。 使用例としてはこのようになります。

```rust
// ライブラリの呼び出しは省略しています

fn main() {
    let mnist = MNIST::new();
    let x_train = mnist.train_img;
    let y_train = mnist.train_label;
    let x_test =mnist.test_img;
    let y_test = mnist.test_label;

    let image_num = 0;

    println!("{:#.1?}\n",x_train.slice(s![image_num, .., ..]));
    println!("{:?}", x_train.shape());
}
```

xは画像データ、yは正解ラベルを表していて、trainは学習データ、testはテストデータで分割しています。最後の`println!("{:#.1?}\n",x_train.slice(s![image_num, .., ..]));` で0番目の画像の行列を表示します。MNISTの画像データは28×28のデータです。枚数は学習データが6万枚、テストデータが1万枚なので、学習データの行列の形状は(60000,28,28)となります。MNIST構造体の保持するデータは全て**ndarray型** なので、そのままモデルに渡すことができます。(正確に言うと、MNIST構造体の中の処理でndarray型に変換します。)     
<br>

```rust
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

    let x_train = x_train.to_shape((50000, 28 * 28)).unwrap();
    let x_test = x_test.to_shape((10000, 28 * 28)).unwrap();

    // one_hotベクトル化
    let y_train = arr2d_to_one_hot(y_train.mapv(|x| x as u32).view(), 10);
    let y_test = arr2d_to_one_hot(y_test.mapv(|x| x as u32).view(), 10);

    let max_epoch = 5;
    let lr = 0.01;
    let batch_size = 100;

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

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_train.select(Axis(0), chunk_indices).to_owned().rv();
            let y_batch = y_train.select(Axis(0), chunk_indices).to_owned().rv();

            let y = model.call(&x_batch);

            let mut loss = F::softmax_cross_entropy_simple(&y, &y_batch);
            let acc = accuracy(
                y.data().into_dimensionality().unwrap().view(),
                y_batch.data().into_dimensionality().unwrap().view(),
            );
            model.cleargrad();
            loss.backward(false);
            optimizer.update();

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
}
```
MNIST構造体を用いてデータを読み込みます。この際、`y_train`、`y_test` のデータは1次元のラベルなので、`arr2d_to_one_hot()`関数で **one_hotベクトル** に変換します。また、MNISTデータは今までのデータ数と桁が異なるので、バッチサイズを100とします。あとは基本的に先ほどの**Spiral学習** の時と同じです。`model` のレイヤーの構造を自由に変更して学習を色々試してみましょう。おおよそ80～90%くらいの精度になると思います。補足でも説明しましたが、他の活性化関数、例えば **ReLU関数** はMNIST学習でよく使われるので、それに変更して学習するのも面白いです。



<br>   

以上で**StuCrsフレームワーク:基礎編** は終了です。フレームワークの基礎をこのドキュメントでは実装してきました。続いてはフレームワークのさらなる機能的な拡張、 **CNN編** です。   

TODO:CNN編ドキュメント、url貼り付け