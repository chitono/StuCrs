# 利便性を高める拡張
前のステップでは実数による自動微分を実装しました。しかしまだ演算子に対応しておらず計算をするための関数をいちいち書かなくてはいけません。このステップの目標としては＋や * などの演算子に対応することです。例をあげるなら、aとbをVariable構造体として、a,bを掛けるとき、**mul(a,b)**と書かなければなりません。**a*b** と書けると便利になります。これから＋や*など演算子が扱えるようにVariable構造体を拡張していきます。

## 四則演算を行う関数の作成
まず演算子を使えるようにVariable構造体を拡張する前に私たちのフレームワークに掛け算や割り算をする関数を実装する必要があります。足し算をするadd関数はすでに実装したので、**これを参考に引き算のsub関数、掛け算のmul関数、割り算のdiv関数を実装してみましょう。** 基本的に+を-、*、/に変更するだけですが、backwardのところは偏微分が関わっているので、微分の計算式を載せます。5.3のadd関数のバックプロパゲーションのグラフを参考に考えてみましょう。答えはgithubのコードで確認しましょう。

- 足し算（add）  
forward: \\(z = x_0 + x_1\\)   
backward: \\(\partial z/\partial x_0 = 1、 \partial z/\partial x_1 = 1\\)   
\\(x_0,x_1\\)どちらとも偏微分の値は1なので、上流からきた微分の値に1をかける,つまりそのまま流せばよい


- 掛け算（mul）  
forward: \\(z = x_0 \cdot x_1\\)   
backward: \\(\partial z/\partial x_0 = x_1、 \partial z/\partial x_1 = x_0\\)    
\\(x_0\\)の偏微分は\\(x_1\\)なので、上流からきた微分の値に\\(x_1\\)をかけて流す。  
\\(x_1\\)の偏微分は\\(x_0\\)なので、上流からきた微分の値に\\(x_0\\)をかけて流す。

- 引き算（sub）  
forward: \\(z = x_0 - x_1\\)   
backward: \\(\partial z/\partial x_0 = 1、 \partial z/\partial x_1 = -1\\)    
\\(x_0\\)の偏微分は1なので、上流からきた微分の値をそのまま流す。  
\\(x_1\\)の偏微分は-1なので、上流からきた微分の値をマイナスにして流す。

- 割り算（div）  
forward: \\(z = x_0 / x_1\\)   
backward: \\(\partial z/\partial x_0 = 1/x_1、 \partial z/\partial x_1 = -x_0/x_1^2\\)    
\\(x_0\\)の偏微分\\(1/x_1\\)なので、上流からきた微分の値に\\(1/x_1\\)をかけて流す。  
\\(x_1\\)の偏微分は\\(-x_0/x_1^2\\)なので、上流からきた微分の値に\\(-x_0/x_1^2\\)をかけて流す。

- 負数（neg） ・・・この関数は一変数関数なので、普通の微分です。   
forward: \\(z = -x_0\\)   
backward: \\(\partial z/\partial x_0 = -1\\)  
\\(x\\)の微分は-1なので、上流からきた微分の値をマイナスにして流す。  


また、四則演算と呼べるのかは定かではありませんが、ここで定義しておくと便利なので累乗を計算する関数 **pow** も実装しましょう。pow関数は \\(z = x^c\\) で表され、cは定数とします。すると、微分は、\\(\partial z/\partial x = cx^{c-1}\\)となります。

表にまとめると、このようになります。

| 関数 | Forward | Backward (\\(\partial z/\partial x_0\\)) | Backward (\\(\partial z/\partial x_1\\)) |
|-----------|---------------|---------------|------------|
| add | \\(z = x_0 + x_1\\) | \\(\partial z/\partial x_0 = 1\\) | \\(\partial z/\partial x_1 = 1\\) |
| mul | \\(z = x_0 \cdot x_1\\) | \\(\partial z/\partial x_0 = x_1\\) | \\(\partial z/\partial x_1 = x_0\\) |
| sub | \\(z = x_0 - x_1\\) | \\(\partial z/\partial x_0 = 1\\) | \\(\partial z/\partial x_1 = -1\\) |
| div | \\(z = x_0 / x_1\\) | \\(\partial z/\partial x_0 = 1/x_1\\) | \\(\partial z/\partial x_1 = -x_0/x_1^2\\) |
| neg | \\(z = -x_0\\) | \\(\partial z/\partial x_0 = -1\\) | なし |
| pow | \\(z = x^c\\) | \\(\partial z/\partial x = cx^{c-1}\\) | なし |

TODO: 四則演算関数の微分が正しく行われるか試すコードを追加予定

## 演算子のオーバーロード
この章ではいままで実装してきた演算子の関数をオーバーロードしていきます。Add関数とNeg関数を例にして説明します。

```rust
use std::ops::{Add, Div, Mul, Neg, Sub};

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[self.clone(), rhs.clone()]);
        add_y
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[self.clone()]);
        neg_y
    }
}
```
そもそもオーバーロードとは何でしょうか？実際のところ、オーバーロードは関数オーバーロードというものを指すことが多く、pythonやC系言語などに用いられるものであり、Rustにはそのような関数を再定義する機能は一般的ではありません。その代わり、トレイトを用いて似たような実装ができます。そして、演算子という一部の機能ではオーバーロードといった関数の再定義が行えます。ここではイメージがしやすいように、より一般的なオーバーロードという言葉を用いて説明していきます。  

オーバーロードにおいて大事なところは、はじめのstdでインポートした{Add、Div …}です。これらはある自分で実装した構造体に演算子を実装したいときに用いるものです。コードを見ると、これはAddトレイトというstdに含まれる提供されたトレイトを、自作したRcVariableに実装しています。また、`type Output`、といったあまり見慣れないコードもありますが、これはオーバーロードするトレイトの独自の機能です。ではここでAddの部分のコードがどのような振る舞いをするのか見てみましょう。

はじめに型を指定します。足し算といった2項演算子の場合、z = x・yという形になりますが、ここで`self` は左の **x** 、`rhs` は右の **y** 、`Output` は **z** の型を示します。これらは全て **RcVariable** なので、コードのように指定します。`self` については **RcVariable** にオーバーロードするので型を明示する必要はありません。そのあとは左と右のRcVariableを **add関数** に渡すという処理です。これは、**+** 演算子を用いると、add関数が自動で呼び出されるということです。
これらのオーバーロードを用いると、このようにコードを**シンプルに**書けるようになります。

```rust
fn main() {
    let a = RcVariable::new(1.0);
    let b = RcVariable::new(2.0);

    // 今まで足し算
    let c = add(a.clone(),b.clone());

    // オーバーロードを用いた足し算
    let c = a.clone() + b.clone();
}
```
二つのコードを比べてみてください。コードの量自体はあまり変わっていませんが、明らかに計算の構造が理解しやすくなりました。これはまだただの足し算なのであまり良さを感じませんが、今後複雑な関数を実装していくことで、このオーバーロードの恩恵を実感することになります。
この要領で他の演算子もオーバーロードしていきましょう。   
>**neg** は一変数関数なので、注意してください。また、オーバーロードに対応しているのはadd,mul,sub,div,negのみなので、**pow** はできません。よくわからない場合は**stucrs**のプログラムの[lib](https://github.com/chitono/StuCrs/blob/main/stucrs/src/lib.rs)をご覧ください。

TODO: RcVariableの追加実装の説明追加予定
## RcVariableの追加実装
ここではよりRcVariableを便利に、そして可読性が増すように、RcVariableに独自のメソッドを追加していきます。

### データからRcVariableを直接生成する関数
**RcVariable** を生成する際、今まではRcVariableのメソッドである `new(data)` という関数をもちいていました。実はこれをもっと簡単に、シンプルに生成する方法があります。前に用いた **オーバーロード** を用いるのです。※ここで言うオーバーロードも厳密にはオーバーロードではありません。   

具体的にはf32型に自身のデータを保持するRcVariableを生成する関数をf32にオーバーロードするのです。試しにコードを作成しましょう。

```rust
pub trait F32ToRcVariable {
    fn rv(&self) -> RcVariable;
}

impl F32ToRcVariable for f32 {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.clone()) // <- 中の処理はRcVariableのnew()と同じ
    }
}
```
はじめにトレイトで関数を定義します。`rv()` という関数名です。続いてこの関数の処理を書き、f32に実装します。関数の処理はただのRcVariableの `new()` と同じです。これにより、f32型独自のメソッドを実装できました。ではこの関数を用いてRcVariableを生成してみましょう。

```rust
fn main() {
    let a = RcVariable::new(1.0); // <- 今までの生成方法
    let b = 2.0.rv();             // <- 新たな生成方法
}
```
この二つの生成を比べると、変数となる **a,b** がどのような値を持つRcVariableか一目で判断がつきます。今後このメソッドを多用しますので、忘れずに実装しておきましょう。