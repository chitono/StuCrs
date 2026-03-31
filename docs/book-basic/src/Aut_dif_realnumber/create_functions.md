# 関数の作成

先ほど変数を実装しましたが、変数はどのように作り出されるのでしょうか。それは **関数** です。変数を入力として関数に渡し、出力として新たな変数を生み出します。それでは関数を実装していきましょう。


## Functionトレイトの実装

はじめにRust独自の仕様であるトレイトを理解する必要があります。トレイトとは様々な構造体がある特定のふるまいをすることを保証するものです。ここでいうふるまいとは、トレイトを継承した構造体がすべて同じある関数を保持しているということです。　　
今後関数は様々な構造体で使用するのでトレイトを使用して関数を定義します。では実際に関数をトレイトで実装してみましょう


```rust
fn main() {
 　let x = Variable::new(2.0);
    　println!("{}", x.data); //2.0
   　 let f = Square {};
    　let y = f.call(&x);
   　 println!("{}", y.data); //4.0
 }

trait Function {
    fn call(&self, input: &Variable) -> Variable;
    fn forward(&self, x: f32) -> f32;
}

struct Square {}

impl Function for Square {

    fn call(&self, input: &Variable) -> Variable {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::new(y);
        output
    }

    fn forward(&self, x: f32) -> f32 {
        x.powf(2.0)
    }
}
```
まずはFunctionというtraitを実装します。ここでは `call` と `forward` というメソッドを作ります。今後このtraitには多くの関数（Exp関数やsin関数など）を追加するためcallにはすべての関数に共通する「Variableからデータを取り出す」、「計算結果をVariable型にして返す」という２つの機能のみを追加します。具体的な計算はforwardにやらせます。ここで大事なのは **input** 、**output** はVariable型、x,yはf32型であるということです。

次に **Squre** という構造体を実装します. その後implキーワードという定義されたメソッドをimplブロック内で具体的に実装できる機能を用いることでSqure関数という２乗の計算を実装します。main関数を実行すると２の２乗の結果がでます。

## Exp関数の実装

新しい関数を１つ実装します。今回は**e**(ネイピア数)のｘ乗という関数を実装します。

```rust
struct Exp {}

impl Function for Exp {

    fn call(&self, input: &Variable) -> Variable {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::new(y);
        output
    }
    fn forward(&self, x: f32) -> f32 {
        x.exp()
    }
}
```
Squre構造体と同じように実装します。変更点はなかの計算がeのｘ乗に変わっただけです。
※今後、このExp構造体のように、Functionトレイトを実装した関数の構造体(SinやTanhなど)を多く実装していきます。その際、これらの構造体を **Function構造体** と呼ぶことにします。

## Function構造体を呼び出す関数

**「Functionトレイトの実装」** のコードを見てみましょう。Function構造体を用いて計算する際、`let f = Square{}` と、 `let y = f.call(&x)` という二つのコードで実行しています。この処理はSquareというFunction構造体を作成し、作成した構造体に変数を渡して計算するという今後多く使われる基本的な処理です。これを一つの関数として定義しましょう。

```rust
fn square(&x:Variable) -> Variable {
    let f = Square{};
    let y = f.call(x);
    y
}
```
このように一つの関数としてまとめると、前のコードをよりわかりやすく書けます。
```rust
fn main() {
 　let x = Variable::new(2.0);
    　println!("{}", x.data); //2.0
    　let y = square(&x);   
   　 println!("{}", y.data); //4.0
 }
```
ここで大事なのは、構造体ははじめの一文字は **大文字**で、関数の場合はすべて小文字で書くというルールを決めることです。そうすることで、このコードは **構造体**、**関数** どちらを指しているのかがすぐわかります。では同じようにExpにも実装してみましょう。
