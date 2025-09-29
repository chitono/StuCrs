
## 研究概要

本研究では Rust言語を用いて「StuCrs」というディープラーニングのフレームワークを一から実装、開発しました。StuCrsというフレームワークの特徴はフルRust実装で、直感的に原理を理解できるシンプルな構造となっており、
ユーザーが一から実装し、深層学習の原理の理解を深めてもらう教材としての役割を果たすフレームワークです。また、Rust言語を学びたい方にとっても良いサンプルコードです。

## 背景
・TensorFlowやPyTorchといった既存のフレームワークのほとんどがドキュメントやコミュニティが英語だったりと、日本語によるフレームワークの開発にとって障壁となっている。

・機械学習の開発がpythonやC系言語に比べてRustは遅れている。


## 研究のコンセプト
・日本語によってコードの説明をすることでユーザー自らが一からフレームワークを実装してもらい、深層学習の原理を探究してもらうこと。

・日本語のコミュニティを構築しやすい国産のフレームワークを、機械学習で開発途上のRustで実装することで、さらなる日本でのRustにおける深層学習のコミュニティを活発にし、開発を促すこと。

## 研究にあたって
本研究は下の著書『ゼロから作るDeep Learning③フレームワーク編』をもとにして実装しています。著者である斎藤康毅氏に著書の考えや表現の使用を許可していただいたことに感謝を申し上げるとともに、この著書オリジナルのフレームワークDeZeroも研究の参考として利用させていただいています。
<p><img width="280" height="134" alt="Image" src="https://github.com/user-attachments/assets/6c0ddf88-3371-40aa-a131-075947068e1b" /> &emsp;
  <img width="100" height="142" alt="Image" src="https://github.com/user-attachments/assets/d5d1ca74-79cb-4de3-b55c-537c705788f7" />


## ドキュメント


開発した深層学習のフレームワーク「StuCrs」の実装までのコードの説明をこちらのドキュメントで見ることができます。これを読んでぜひ一からRustでフレームワークを実装してみましょう！
<https://docs.google.com/document/d/1jJL_ijYnqIFADSTfTqLcnNre754g24bE963L_r3hwus/edit?usp=sharing>


## ファイル構成

|フォルダ名 |説明         |
|:--        |:--                  |
|[stucrs](/stucrs)       |StuCrs(CPU用)のソースコード|
|[stucrs-gpu](/stucrs)    |StuCrs(GPU用)のソースコード|
|[assets](/assets)     |StuCrsを用いて様々な実験した際のデータや画像|



## 使用した外部のクレート

本研究で必要とする外部クレートとバージョンは下記の通りです。

- [ndarray-0.16.0](https://docs.rs/ndarray/0.16.0/ndarray/index.html)
- [ndarray_stats-0.6.0](https://docs.rs/ndarray-stats/0.6.0/ndarray_stats/index.html)
- [ndarray-rand-0.15.0](https://docs.rs/ndarray-rand/latest/ndarray_rand/index.html)
- [mnist-0.6.0](https://docs.rs/mnist/latest/mnist/index.html)
- [rand-0.8](https://docs.rs/rand/latest/rand/index.html)
- [rand_distr-0.4.0](https://docs.rs/rand_distr/0.4.0/rand_distr/index.html)
- [fxhash-0.2.1](https://docs.rs/fxhash/latest/fxhash/index.html)


NVIDIAのGPUで実行できる機能も提供しています。その場合はstucrs-gpuをダウンロードし、また下記のtensor_frameクレートを使用します。

- [tensor_frame](https://docs.rs/tensor_frame/latest/tensor_frame/index.html) （オプション）


## 実行方法
はじめにDockerfileとcompose.yamlファイルを用いてdockerでコンテナを立ち上げてください。
フォルダーのstucrsをダウンロードしていただき、外部クレートとしてご利用ください。また、こちらのクレートはバグといった不具合の対応が不十分だと判断し、ライブラリクレートとしては公開しておりません。またオプションとして、NVIDIAのGPUで実行できる機能も提供しています。その場合はstucrs-gpuをダウンロードしてください。(現在GPU版は一部不具合が発生しており、完全に対応している状態ではないため、使用は今しばらくお待ちください。)

## MNISTの学習の実装例
```

use ndarray::*;

use rand::seq::SliceRandom;
use rand::*;

use std::time::Instant;
use stucrs::config;
use stucrs::core_new::ArrayDToRcVariable;

use stucrs::datasets::*;
use stucrs::functions_new::{self as F, accuracy};
use stucrs::layers::{self as L, Activation};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

fn main() {
    let mnist = MNIST::new();
    let x_train = mnist.train_img.view();
    let y_train = mnist.train_label.view();
    let x_test = mnist.test_img.view();
    let y_test = mnist.test_label.view();

    

    let x_train = x_train.to_shape((50000, 28 * 28)).unwrap();
    let x_test = x_test.to_shape((10000, 28 * 28)).unwrap();

    let y_train = arr2d_to_one_hot(y_train.mapv(|x| x as u32).view(), 10);
    let y_test = arr2d_to_one_hot(y_test.mapv(|x| x as u32).view(), 10);

    

    let max_epoch = 5;
    let lr = 0.01;
    let batch_size = 100;

    

    let data_size = x_train.shape()[0];
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
```


## 試験データ

StuCrsを実際に実行して処理速度などを計測した試験データをこちらに公開しております。　<https://docs.google.com/spreadsheets/d/1Fkxn7yqLILJlHYeADVa_jJljFBYD5ZvIH0I7-EVTuFU/edit?usp=sharing>


## 最後に
はじめに、私たちの研究に目を通していただきありがとうございます。
本研究は素人である高校生が独自に研究したものであり、Rustのパフォーマンス的に、もしくは習わし的にふさわしくないコード、また深層学習の知識における間違いが多くあるかと思います。もし気になる点や改善した方がいいというご意見がございましたら、是非ともお手柔らかにお知らせください。たくさんのご意見、ご感想をお待ちしております。
