# 変数の作成

変数とはデータを格納する箱のようなものです。はじめにこの変数をVariableという名前で構造体として実装してみましょう。

```rust
fn main() {
    let mut a = Variable::new(1.0);
    println!("{}", a.data); //1.0
    a.data = 5.0;
    println!("{}", a.data); //5.0
}

struct Variable {
    data: f32,
}

impl Variable {
    fn new(data: f32) -> Self {
        Variable { data }
    }
}
```

まず **Variable** という構造体を実装します。フィールドとして小数をあつかえるf32型を `data` として保持します。また、コンストラクタを生成するための初期化の関数 `new()` を定義します。これにより、f32のデータを渡すことでデータを保持する変数、Variable型を生成することができます。main関数を実行すると、Variableのデータを見ることができます。  

>pythonを学んだ人からすると、関数の戻り値がしっかり示されていないと感じるかもしれません。pythonだと戻り値はreturnを用いますが、Rustの場合、戻り値となるものには後ろにセミコロンをつけないのが通常です。この場合、 `fn new()` の戻り値は `Variable{data}` ですが、後ろにセミコロンがないので戻り値として扱われます。
