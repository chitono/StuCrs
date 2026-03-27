# Softmax関数

ではsoftmax関数の数式、そしてどのような処理なのか見ていきます。

$${p_i = \frac{exp(y_i)}{\sum_{j=1}^N exp(y_j)}}$$

---

ではデータ数が2の場合を見てみましょう。

<br>

$$
X:\begin{pmatrix}
x_0 & x_1 & x_2 \\\\
x_3 & x_4 & x_6 
\end{pmatrix}
\xrightarrow{\text{予測}}
Y:\begin{pmatrix}
y_0 & y_1 & y_2\\\\
y_3 & y_4 & y_5
\end{pmatrix}
$$

<br>

$$
Y:\begin{pmatrix}
y_0 & y_1 & y_2\\\\
y_3 & y_4 & y_5
\end{pmatrix}
\xrightarrow{\text{Softmax}}
P:\begin{pmatrix}
\frac{e^{y_0}}{e^{y_0}+e^{y_1}+e^{y_2}} & \frac{e^{y_1}}{e^{y_0}+e^{y_1}+e^{y_2}} & \frac{e^{y_2}}{e^{y_0}+e^{y_1}+e^{y_2}}\\\\
\frac{e^{y_3}}{e^{y_3}+e^{y_4}+e^{y_5}} & \frac{e^{y_4}}{e^{y_3}+e^{y_4}+e^{y_5}} & \frac{e^{y_5}}{e^{y_3}+e^{y_4}+e^{y_5}}
\end{pmatrix}
$$

<br>

文字が小さくなってしまいましたが、要は同じ行の中でネイピア数の指数にかけて平均を取っているようなものです。ではなぜ指数を取るのかというとそれは負の値であっても処理できるからです。指数は正負かかわらず必ず正になるので、どんな値でも処理することができます。

ではsoftmax関数を実装していきます。この処理は[二乗平均誤差](../NN_building/NN_syudou_jissou/gosa_gyakudenpan.md)と同じように、分母となる値を **sum** で求め、それを行列として割ればよいのです。    


$$
Y:\begin{pmatrix}
y_0 & y_1 & y_2\\\\
y_3 & y_4 & y_5
\end{pmatrix}
\xrightarrow{\text{Exp}}
Y_{exp}:\begin{pmatrix}
e^{y_0} & e^{y_1} & e^{y_2}\\\\
e^{y_3} & e^{y_4} & e^{y_5}
\end{pmatrix}
\xrightarrow{\text{Sum(axis=1)}}
Y_{sum-exp}:\begin{pmatrix}
e^{y_0} + e^{y_1} + e^{y_2} \\\\ 
e^{y_3} + e^{y_4} + e^{y_5}
\end{pmatrix}
$$

<br>

$$
Y_{exp}:\begin{pmatrix}
e^{y_0} & e^{y_1} & e^{y_2}\\\\
e^{y_3} & e^{y_4} & e^{y_5}
\end{pmatrix} \div
Y_{sum-exp}:\begin{pmatrix}
e^{y_0} + e^{y_1} + e^{y_2} \\\\ 
e^{y_3} + e^{y_4} + e^{y_5}
\end{pmatrix}
\xrightarrow{\text{Softmax}}
P:\begin{pmatrix}
\frac{e^{y_0}}{e^{y_0}+e^{y_1}+e^{y_2}} & \frac{e^{y_1}}{e^{y_0}+e^{y_1}+e^{y_2}} & \frac{e^{y_2}}{e^{y_0}+e^{y_1}+e^{y_2}}\\\\
\frac{e^{y_3}}{e^{y_3}+e^{y_4}+e^{y_5}} & \frac{e^{y_4}}{e^{y_3}+e^{y_4}+e^{y_5}} & \frac{e^{y_5}}{e^{y_3}+e^{y_4}+e^{y_5}}
\end{pmatrix}
$$

**Exp** で行列の要素全体に指数を施し、Sumで軸を1に指定して合計することで、分母となる値を持つ行列を計算します。あとはその行列で割れば、求める処理を実現できます。ではこれをコードで実装します。

```rust
pub fn softmax_simple(x: &RcVariable) -> RcVariable {
    let exp_y = exp(&x);

    let sum_y = sum(&exp_y, Some(1)); //sumで軸1指定

    let y = exp_y.clone() / sum_y.clone();
    y
}
```
では実装した **softmax関数** をテストします。   
TODO: softmax関数テストコード


このようにsoftmax関数は出力されたデータを確立に変換します。この変換はニューラルネットワークの分類において重要な処理です。

では続いてクロスエントロピー誤差を実装するにあたり必要な **Log関数** を **Function構造体** として実装します。