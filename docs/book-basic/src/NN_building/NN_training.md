# ニューラルネットワークの学習

では早速今まで実装してきた構造体を組み合わせて、はじめの[手動でのニューラルネットワーク](../NN_building/NN_syudou_jissou/syudou_training.md)を自動化して学習させます。

```rust
use ndarray::{array, Array};
use std::array;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::functions_new as F;
use stucrs::layers::{self as L, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

use std::f32::consts::PI;

use stucrs::core_new::ArrayDToRcVariable;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

fn main() {
    let x_data = Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();

    let y_data =
        x_data.mapv(|x| (2.0 * PI * x).sin()) + Array::random((100, 1), Uniform::new(0.0f32, 1.0));

    let x = x_data.rv();

    let y = y_data.rv();

    let l1 = L::Dense::new(10, false, None, L::Activation::Sigmoid);
    let l2 = L::Linear::new(10, false, None);

    let mut model = BaseModel::new();
    model.stack(l1);
    model.stack(l2);

    let lr = 0.1;

    let mut optimizer = SGD::new(lr);

    optimizer.setup(&model);


    let iters = 10000;
    for i in 0..iters {
        let y_pred = model.call(&x);

        let mut loss = F::mean_squared_error(&y, &y_pred);

        model.cleargrad();
        loss.backward(false);

        optimizer.update();

        if i % 1000 == 0 {
            println!("i = {:?}", i);
            println!("loss = {:?}\n", loss.data());
        }
    }
}
```
前回のコードを見てもらうとわかるように、パラメーターを処理するコードが劇的に少なくなりました。また  **predict()** という関数をModelとして設定し、Optimizerを設定することで、パラメーター更新も **update()** のみですべてのパラメーターを更新することができました。では改めて、同じデータで学習させてみましょう。

<br>

ここまでで、 **ニューラルネットワークの骨格となるシステムが構築しました。** あとはよりニューラルネットワークのフレームワークとして機能を仕上げていきます。次に実装する機能は **損失関数** です。


