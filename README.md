>本研究では、国内コミュニティ及びグローバルな知見の両方を重視する一環として、日本語と英語による`README`の公開を行っています。

[日本語](README.md) / [English](readme_en.md)

## 研究概要

本研究では Rust言語を用いて「StuCrs」というディープラーニングのフレームワークを一から実装、開発しました。StuCrsというフレームワークの特徴はフルRust実装で、直感的に原理を理解できるシンプルな構造となっており、
ユーザーが一から実装し、深層学習の原理の理解を深めてもらう教材としての役割を果たすフレームワークです。また、Rust言語を学びたい方に対して良いサンプルコードです。

## 背景
- TensorFlowやPyTorchといった既存のフレームワークのほとんどがドキュメントやコミュニティが英語だったりと、国内のフレームワークの開発にとって障壁となっている。
  
- 機械学習の開発がpythonやC系言語に比べてRustは遅れている。

- Rustという新しい考えが導入された言語で実装することにより、深層学習の価値を違う視点で見出すことができるかもしれない。


## 研究のコンセプト
- 日本語によってコードの説明をすることでユーザー自らが一からフレームワークを実装してもらい、深層学習の原理を探究してもらうこと。

- 日本語のコミュニティを構築しやすい国産のフレームワークを、機械学習で開発途上のRustで実装することで、さらなる日本でのRustにおける深層学習のコミュニティを活発にし、開発を促すこと。

- Rustという抽象的でモダンな設計論理の言語をもとに、深層学習の新たな理論を求める実験の場としての役割を果たすこと。



## 研究にあたって
本研究は下の著書『ゼロから作るDeep Learning③フレームワーク編』をもとにして実装しています。著者である斎藤康毅氏に著書の考えや表現の使用を許可していただいたことに感謝を申し上げるとともに、この著書オリジナルのフレームワークDeZeroも研究の参考として利用させていただいています。
<div style="display:flex; gap:100px;">
  <img style="width:31%;" alt="Image" src="https://github.com/user-attachments/assets/4bd79c56-4805-471d-af1c-af0c2fb5d890" />
  <img style="width:15%;" alt="Image" src="https://github.com/user-attachments/assets/d5d1ca74-79cb-4de3-b55c-537c705788f7" />
</div>


## ニュース
本研究は第19回高校生理科研究発表会に出場し、最優秀賞をいただきました。参考にさせていただいた研究の方、審査していただいた方々に感謝申し上げます。大会に提出したポスターはassetsフォルダーでみることができます。


## ドキュメント


開発した深層学習のフレームワーク「StuCrs」の実装までのコードの説明をこちらのドキュメントで見ることができます。これを読んでぜひ一からRustでフレームワークを実装してみましょう！
- [『StuCrsドキュメント・基礎編』](https://chitono.github.io/StuCrs-Book/book-basic/)
- [『StuCrsドキュメント・CNN編』](https://chitono.github.io/StuCrs-Book/book-cnn/)





## ファイル構成

|ファイル・フォルダ名 |説明         |
|:--        |:--                  |
|[stucrs](/stucrs)       |StuCrsフレームワークのソースコード|
|[stucrs_test](/stucrs_test)|stucrsを外部クレートとして読み込み学習させるコード|
|[example](/examples)|フレームワークを用いた実装例のコード|
|[assets](/assets)     |StuCrsを用いて様々な実験した際のデータや画像|
|[PERFORMANCE](/PERFORMANCE.md)| フレームワークのパフォーマンスの試験データ|
|[ROAD_MAP](/ROAD_MAP.md)|課題・今後の展望|
|[REFERENCES](/REFERENCES.md)|参考にした文献・資料|


## 使用した外部のクレート

本研究で必要とする外部クレートとバージョンは下記の通りです。

- [ndarray-0.16.0](https://docs.rs/ndarray/0.16.0/ndarray/index.html)
- [ndarray_stats-0.6.0](https://docs.rs/ndarray-stats/0.6.0/ndarray_stats/index.html)
- [ndarray-rand-0.15.0](https://docs.rs/ndarray-rand/latest/ndarray_rand/index.html)
- [mnist-0.6.0](https://docs.rs/mnist/latest/mnist/index.html)
- [rand-0.8](https://docs.rs/rand/latest/rand/index.html)
- [rand_distr-0.4.0](https://docs.rs/rand_distr/0.4.0/rand_distr/index.html)
- [fxhash-0.2.1](https://docs.rs/fxhash/latest/fxhash/index.html)


NVIDIAのGPUで実行できる機能も提供しています。その場合はfeaturesをcudaに指定して実行してください。


## 実行方法
はじめにDockerfileとcompose.yamlファイルを用いてdockerでコンテナを立ち上げてください。
フォルダーのstucrsをダウンロードしていただき、外部クレートとしてご利用ください。また、こちらのクレートはバグといった不具合の対応が不十分だと判断し、ライブラリクレートとしては公開しておりません。またオプションとして、NVIDIAのGPUで実行できる機能も提供しています。
詳しくは[研究の解説ドキュメント](https://chitono.github.io/StuCrs-Book/book-basic/)をご参照ください。

## MNISTの学習の実装例
他の学習コードなどはこちらの[examples](examples)をご覧ください。
```

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

    let x_train = x_train.reshape(vec![50000, 28 * 28])?;
    let x_test = x_test.reshape(vec![10000, 28 * 28])?;

    let y_train = tensor2d_to_one_hot(y_train, 10)?;
    let y_test = tensor2d_to_one_hot(y_test, 10)?;


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

            let y_batch = y_train.axis_slice(0, chunk_indices)?.rv();

            let y = model.call(&x_batch)?;

            let mut loss = softmax_cross_entropy_simple(&y, &y_batch)?;

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
```


## 試験データ

StuCrsを実際に実行して処理速度などを計測した[試験データ](PERFORMANCE.md)を公開しております。　

## 今後の課題
本研究が課題として解決したいと考えている問題はこちらの[Discussions](https://github.com/chitono/StuCrs/discussions)に載せています。ぜひご覧ください。
## 参考文献

詳細な参考文献一覧は[REFERENCES.md](REFERENCES.md)を参照してください。

## 最後に
私たちの研究に目を通していただきありがとうございます。
本研究は素人である高校生が独自に研究したものであり、Rustのパフォーマンス的に、もしくは習わし的にふさわしくないコード、また深層学習の知識における間違いが多くあるかと思います。もし気になる点や改善した方がいいというご意見がございましたら、是非ともお手柔らかにお知らせください。たくさんのご意見、ご感想をお待ちしております。

## ご意見・ご感想はこちら
どんなご意見・ご感想でも大歓迎です!
### 例えば
- 見た立場 (学生 / 研究者 / エンジニアなど)
- 面白いと思った点
- わかりにくいと感じた点
- 改善案・関連研究　　　

### 本研究におけるアドバイスや質問はこちらの[Discussions](https://github.com/chitono/StuCrs/discussions)まで。  

