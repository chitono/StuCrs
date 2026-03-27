# クロスエントロピー誤差

では最後にクロスエントロピー誤差を実装していきます。今回は、最初の説明のように **softmax** 関数もセットにした関数にします。
```rust
pub fn softmax_cross_entropy_simple(x: &RcVariable, t: &RcVariable) -> RcVariable {
    if x.data().shape() != t.data().shape() {
        panic!("交差エントロピー誤差でのxとtの形状が異なります。tがone-hotベクトルでない可能性があります。")
    }

    let n = x.data().shape()[0] as f32;

    let p = softmax_simple(&x);

    let clamped_p = clamp(&p, 1.0e-15, 1.0); // pを1.0e-15～1.0に収める

    let log_p = log(&clamped_p, None); // 0の値は渡されない

    let tlog_p = log_p * t.clone();

    let y = (-sum(&tlog_p, None)) / n.rv();
    y
}
```
引数の\\(x\\)はモデルの出力、\\(t\\)は正解ラベルを想定しています。
先ほどの説明のように、正解ラベルのデータは **one_hotベクトル** に変換することでモデルの出力と同じ形状にし、掛け算をします。なので形状一致を確認します。   

次にここで初めて登場した **clamp** を説明します。この関数は引数に **p** **1.0e-15**,  **1.0** の二つの値を渡していますが、最初の変数の値をその二つの値内に収める処理を行います。つまり、**p** が\\(1.0^{-15}\leqq p\leqq 1.0\\)に収まるならそのまま値を流し、この範囲からはみ出たら、例えば2.0なら1.0にするというように修正して返します。これは次に処理する **log関数** でlogに0の値を渡すことを防ぐためのものです。この関数に関しては **補足** で説明を行います。   

TODO:clamp関数 補足で説明   


では実装したクロスエントロピー誤差を試してみましょう。データやModelを自由にを用意、構築し、誤差を求めてみましょう。
```rust
fn main() {
    let x = array![[0.1f32, -0.4], [0.3, 0.6], [-1.3, -1.2], [3.1, -0.5]].rv();
    let t = array![
        [0.0f32, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ]
    .rv(); // one_hotベクトルで渡す

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(5, false, None, Activation::Sigmoid));
    model.stack(L::Dense::new(3, false, None, Activation::Sigmoid));

    let y = model.call(&x);

    let loss = F::softmax_cross_entropy_simple(&y, &t);

    println!("loss = {:?}", loss.clone().data()); //誤差が求まる
}
```

これらにより、Modelは多クラスにおける予測値と答えとの誤差を求めることができました。次はこの多クラス分類を用いてより実践的な学習を試みます。