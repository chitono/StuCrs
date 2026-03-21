# 手動による学習

では手動でニューラルネットワークを構築していきます。使用するのは先ほどまで実装してきた**Linear関数**、**活性化関数**、**二乗平均誤差関数**です。これらを組み合わせ、一つの関数としてバックプロパゲーションを行い、勾配を用いてパラメーターを更新していきます。  

## 学習の準備
では学習するに当たってデータを用意します。今回は非線型性のデータの代表的な\\(sin\\)関数を用意します。ここでは\\(y = sin(2\pi\cdot x) + b\\)とします。\\(b\\)はランダムな値で、\\(y\\)のデータにノイズを与えます。
<br>


```rust
use ndarray::{array, Array, IxDyn};
use std::array;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::functions_new::{linear_simple,  sigmoid_simple};

use std::f32::consts::PI;

use stucrs::core_new::ArrayDToRcVariable;
use stucrs::functions_new::mean_squared_error;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

fn main() {
    let x_data = Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();

    let y_data =
        x_data.mapv(|x| (2.0 * PI * x).sin()) + Array::random((100, 1), Uniform::new(0.0f32, 1.0));

    let x = x_data.rv();

    let y = y_data.rv();

    let I = 1;
    let H = 10;
    let O = 1;

    let mut W1 = (0.01f32 * Array::random((I, H), StandardNormal)).rv();

    let mut B1 = Array::zeros(H).rv();

    let mut W2 = (0.01f32 * Array::random((H, O), StandardNormal)).rv();

    let mut B2 = Array::zeros(O).rv();

    fn predict(
        x: &RcVariable,
        w1: &RcVariable,
        b1: &Option<RcVariable>,
        w2: &RcVariable,
        b2: &Option<RcVariable>,
    ) -> RcVariable {
        let t0 = linear_simple(&x, &w1, &b1);
        let t1 = sigmoid_simple(&t0);
        let y = linear_simple(&t1, &w2, &b2);
        y
    }

    //let y_data = 2.0*x_data.clone() +5.0 + Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    //println!("y_data = {:?}",y_data.clone());

    //let x=x_data.rv();

    //let y=y_data.rv();

    //let mut w = Array::zeros((1,1)).rv();
    //let mut b = Array::zeros(1).rv();
    let lr = 0.2;
    let iters = 1000;
    for i in 0..iters {
        let y_pred = predict(&x, &W1, &Some(B1.clone()), &W2, &Some(B2.clone()));

        

        let mut loss = mean_squared_error(&y, &y_pred);

       

        W1.cleargrad();
        W2.cleargrad();
        B1.cleargrad();
        B2.cleargrad();

        loss.backward();
        

        let w1_data = W1.data();
        let b1_data = B1.data();
        let w2_data = W2.data();
        let b2_data = B2.data();

        let current_grad_w1 = W1.grad().unwrap().data();
        let current_grad_b1 = B1.grad().unwrap().data();
        let current_grad_w2 = W2.grad().unwrap().data();
        let current_grad_b2 = B2.grad().unwrap().data();

        

        W1.0.borrow_mut().data = w1_data - lr * current_grad_w1;
        B1.0.borrow_mut().data = b1_data - lr * current_grad_b1;
        W2.0.borrow_mut().data = w2_data - lr * current_grad_w2;
        B2.0.borrow_mut().data = b2_data - lr * current_grad_b2;
        
        if i % 100 == 0 {
            println!("i = {:?}", i);
            println!("loss = {:?}\n", loss.data());
        }
    }
}
```

はじめにこのニューラルネットワークで用いる変数(パラメーター)を用意します。\\(x\\)は学習データ、\\(y\\)は正解ラベル(xとyのペアのデータが教師データです。)、\\(w\\)と\\(b\\)は重み(パラメーター)です。この時パラメーターは乱数を用いて最初の重みを設定します。行列でランダムな値を持つ行列を作成するためにArray型で提供される[ndarray-rand](https://docs.rs/ndarray-rand/latest/ndarray_rand/index.html)を使用します。これは依存関係を別途追加しておく必要がありますが、これによりArrayにランダムな値を生成する機能が自動で追加されます。

>パラメーターの初期値は0～1の範囲で乱数を用いて設定します。もし初期値をある値(0.0など)に設定すると、すべてのパラメーターが画一的なふるまいをしてしまいます。これでは深層学習の複雑な分類は行えません。

TODO:データ、エポック数の説明追加


次に関数を構築します。**Linear**と活性化関数の**sigmoid**を通すだけで、自動でモデルとなる関数が生まれます。それを正解ラベルとともに**mean_squared_error関数**に流すことで、誤差を求める関数となります。この関数を **predict()** 関数としてまとめ、推論を行います。

predictで出力したモデルの予測値\\(y\\)と正解ラベル\\(y_pred\\)を**mean_squared_error関数**に渡し、誤差を**loss**として求めます。その後、この誤差の変数である **loss** を起点にバックプロパゲーションを行います。それが **loss.backward()** です。これにより各パラメーターの勾配を導出します。

最後に求めたパラメーターの勾配をもとに各パラメーターを一つずつ更新します。今回は勾配降下法を用います。更新方法は前の[誤差逆伝播法](gosa_gyakudenpan.md)で確認してください。

以上より手動による学習を行うことができます。最後のコードにある **println!("loss = {:?}\n", loss.data());** より、学習する度に **loss** すなわち予測値と正解との誤差が小さくなるのがわかります。

線形、非線型変換を繰り返し行うことで、sin関数といった非線型な曲線に対応することができました。

今の状態では、パラメーターの初期化、更新を自分たちで行わなければなりません。これがものすごく深い層となった場合、私たちの手に負えなくなります。これから先は、学習にあたり今までの処理を自動化し、フレームワークとしての機能を加えていきます。