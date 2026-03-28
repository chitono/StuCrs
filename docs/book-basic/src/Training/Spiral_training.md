# スパイラルデータの学習

ではいよいよ実際に多クラス分類の学習を行います。先ほどの説明でスパイラルデータの準備は整いました。三色に分類するのでクラス数は3です。この学習は今までのものよりも少し規模が大きくなるので、いくつか準備が必要です。

## ミニバッチ処理
ここではじめてバッチというものが登場しました。バッチとはデータを分割して、一度にモデルに渡すデータの数を調整することです。ここで一度に渡すデータ数をバッチ数と言います。今までのデータはデータ数が少なかったのですが、実際に使われる場合は数万個に及ぶデータを扱います。全データを一度にモデルに渡すと、メモリ使用量も莫大になり、パラメーターも収束しづらいです。なので、大きいデータ数の場合は、分割して渡すのが主流です。私たちが計算を行列で行ってきたのも、このミニバッチ学習に対応するためなのです。全データからいくつかのデータとラベルを取り出し、モデルに渡します。この行列データからランダムに複数の行を取り出す必要がありますが、その処理をndarrayに搭載される **chunks()** メソッドで行えます。

>バッチサイズが1の時、つまりデータを一つずつモデルに渡す学習をオンライン学習、バッチサイズが全データ数の時、つまり全データをまとめてモデルに渡す学習をバッチ学習、その中間をミニバッチ学習と言います。

```rust
use ndarray::*;
use ndarray_stats::QuantileExt;
use std::array;
use std::cell::RefCell;
use std::f32::consts::PI;
use std::rc::Rc;
use std::time::Instant;
use stucrs::config;
use stucrs::core_new::ArrayDToRcVariable;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::datasets::*;
use stucrs::functions_new::{self as F, accuracy, sum};
use stucrs::layers::{self as L, Activation, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::*;

fn main() {
    let max_epoch = 300;
    let lr = 1.0;
    let batch_size = 30;
    
    let train_spiral = Spiral::new(true);
    let test_spiral = Spiral::new(true);

    let x_train = train_spiral.data.view();
    let y_train = train_spiral.label.view();
    let y_train = to_one_hot(y_train, 3);

    let x_test = test_spiral.data.view();
    let y_test = test_spiral.label.view();
    let y_test = to_one_hot(y_test, 3);

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(10, true, None, Activation::Sigmoid));
    model.stack(L::Linear::new(3, true, None));

    let data_size = x_train.shape()[0];
    let mut optimizer = SGD::new(lr);
    optimizer.setup(&model);

    for epoch in 0..max_epoch {
        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
        let mut sum_loss = array![0.0f32];
        let mut sum_acc = array![0.0f32];

        // バッチを取り出す
        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_train.select(Axis(0), chunk_indices).to_owned().rv();
            let y_batch = y_train.select(Axis(0), chunk_indices).to_owned().rv();

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

            sum_loss = &sum_loss + &epoch_loss; //全てのバッチで合計を取る
            sum_acc = sum_acc + acc * (y_batch.len() as f32); //全てのバッチで合計を取る
        }

        let average_loss = &sum_loss / (data_size as f32); //平均をとる
        let average_acc = sum_acc / (data_size as f32);    //平均をとる

        println!(
            "epoch = {:?}, train_loss = {:?}, accuracy = {}",
            epoch + 1,
            average_loss,
            average_acc
        );

        //推論
        config::set_grad_false();

        let mut indices: Vec<usize> = (0..data_size).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
        let mut sum_loss = array![0.0f32];
        let mut sum_acc = array![0.0f32];

        for chunk_indices in indices.chunks(batch_size) {
            let x_batch = x_test.select(Axis(0), chunk_indices).to_owned().rv();
            let y_batch = y_test.select(Axis(0), chunk_indices).to_owned().rv();

            //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

            let y = model.call(&x_batch);
            let mut loss = F::softmax_cross_entropy_simple(&y, &y_batch);
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

        let average_loss = &sum_loss / (data_size as f32);
        let average_acc = sum_acc / (data_size as f32);

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

for文でバッチごとに誤差と正解率を算出し、すべてのデータを学習し終わったらそれらの平均を取ります。この値が、1エポックにおける学習の評価となります。今回は最初の設定から、エポック数300、学習率1.0、バッチ数30としています。前半でバックプロパゲーションを行い、パラメーターを更新します。ここが、学習のところです。そして、後半のコードは推論を行います。つまり、テストです。

エポックの学習を繰り返していくうちに、誤差が減少し、正解率が上がるのがわかると思います。

以上により、スパイラルデータで多クラス分類の学習を行うことができました。

では最後に、フレームワークの実験でよく用いられるデータセット、**MNIST** を学習させて、フレームワークの基礎編を終わりたいと思います。