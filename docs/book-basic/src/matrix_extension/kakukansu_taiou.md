# 各関数の行列への対応
私たちはこれまで様々な関数をStuCrsに実装してきました。例えば**add**,**mul**,**sin**関数などです。しかしながら私たちはそれらの関数を実装するにあたって入力と出力がすべて「スカラ」であることを想定してきました。なので「テンソル」として各関数を行列へと拡張していきましょう。今回はsin関数を例にとって説明したいと思います。   

まずは理論的なところから説明します。今まではxという「スカラ」にsin関数を適用する場合sin(x)とすればよかったわけです。もしｘが「テンソル」の場合、たとえば行列の場合はどうなるでしょうか。その場合は要素ごとにsin関数が適用されます。  

例えばｘ＝[1,2,3,4,5,6]という2次元行列にsin関数を適用した場合、下に表したように要素ごとにsin関数が適用されるわけです。このことを念頭において実装してみましょう。


$$
Sin(X):
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
\mapsto
\begin{bmatrix}
sin(1) & sin(2) & sin(3)\\
sin(4) & sin(5) & sin(6)
\end{bmatrix}
$$   


※変更する関数のみ表示します。
```rust
impl Function for Sin {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Sinは一変数関数です。inputsの個数が一つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[RcVariable]) -> RcVariable {
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.sin()); // <- mapv()を用いて要素ごとに計算

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx = cos(x) * gy.clone(); // <- Cos構造体を呼び出してRcVariableとして処理
        let gxs = vec![gx];
        gxs
    }
}
```
Function構造体でのArrayD型への変更は実はほとんどありません。なぜなら、Function構造体はデータをRcVariableですべて管理しているからです。前回Variable、RcVariableへの対応は済みましたので、特に変更する点はありません。唯一変更する点は **forward** の計算処理の変更です。forwardの中はArray型で計算処理されるので、変更しなければならないところが生まれます。Array型のメソッドの **mapv()** を用いて計算します。backwardの処理は全て、RcVariableで処理されるので、変更点はありません。   



一方で四則演算の関数はどうでしょうか。実はforwardの処理のところでさえ変更する必要はありません。なぜなら、**Array型は演算子に対応しているからです。** つまり、Array型でそのまま+や-が使えるということです。これは私たちが先ほど **RcVariable** に実装した演算子のオーバーロードと同じです。あの演算子のメソッドが自動で呼び出されるので、f32の記法のままで同じ処理をしてくれます。 


ここで重要なのは微分する際のbackwardのときのcosの計算も要素ごとに行わなければならないということです。そこで、cosも同じように構造体を変更して用いれば、正しく微分を行えます。図で表すとこのようになります。



$$
Forward: 
\begin{bmatrix}
x_0 & x_1 & x_2\\
x_3 & x_4 & x_5
\end{bmatrix}
\xrightarrow{Sin}
\begin{bmatrix}
sin(x_0) & sin(x_1) & sin(x_2)\\
sin(x_3) & sin(x_4) & sin(x_5)
\end{bmatrix}
$$   


$$
Backward: 
\begin{bmatrix}
cos(x_0) \cdot gy_0 & cos(x_1) \cdot gy_1 & cos(x_2) \cdot gy_2\\
cos(x_3) \cdot gy_3 & cos(x_4) \cdot gy_4 & cos(x_5) \cdot gy_5
\end{bmatrix}
\xleftarrow{Sin'}
\begin{bmatrix}
gy_0 & gy_1 & gy_2\\
gy_3 & gy_4 & gy_5
\end{bmatrix}
$$   

今までスカラーでバックプロパゲーションを行っていたのが、行列を導入することで、要素ごとのバックプロパゲーションを実現することができました。   

ではSin関数と同じように他の関数も同様に変更してみましょう。四則演算の関数は変更するところはありません。