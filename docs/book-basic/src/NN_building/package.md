# パッケージ化
私たちは今まで様々な関数を実装してきました。これらのたくさんの機能をまとめ、整ったフレームワークにするために、複数のファイルに分割しましょう。

## ライブラリクレートの作成
私たちが開発するフレームワークを外部から読み込むライブラリクレートとして実装します。**cargo new - lib** で作成しましょう。今回はフレームワークの名前から**stucrs** とします。   

次にこのようなディレクトリ構成にします。  

![alt text](<スクリーンショット 2026-03-15 170532_transparent.png>)   
※ 一部ファイルを省略しています。

---

今まで実装してきた**main.rs** ファイルは**testフォルダ** の中にありますが、それとは別に**stucrs** ライブラリのためをクレートを先ほどの説明で用意します。その後、**stucrs**の中の**src**に**functions**フォルダ、**core.rs**ファイルを入れます。また、**functions**フォルダに **math.rs**, **matrix.rs**, **mod.rs** ファイルを入れます。  

ではこれらの追加したフォルダ、ファイルについて説明します。**functions** フォルダは今まで実装してきた**Function構造体**である関数を入れるところであり、たくさんの種類があるため、種類ごとに関数を分けて保存します。今のところ、**math.rs**, **matrix.rs** を用意していますが、今後も関数を実装していくにつれ、新たなファイルが必要になるため、追加していきます。**mod.rs** は追加していったファイル(**math.rs**, **matrix.rs**)などを管理するためのファイルです。  
**core.rs** ファイルはフレームワークの核となる機能を保存します。具体的には**Variable、RcVariable,Functionトレイト** などです。なお、四則演算の関数はFunction構造体ですが、基本的な関数なので、**core.rs** に入れます。
>core.rsのファイルは現在のリポジトリでは**core_new.rs** という名前になっています。  

また、**lib.rs** には**RcVariable** にオーバーロードするコードを移します。

**lib.rs** の中身
```rust
use core_new::RcVariable;
use core_new::{add, div, mul, neg, sub};
use std::ops::{Add, Div, Mul, Neg, Sub};

//演算子のオーバーロード

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[self.clone(), rhs.clone()]);
        add_y
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[self.clone(), rhs.clone()]);
        mul_y
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[self.clone(), rhs.clone()]);
        sub_y
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[self.clone(), rhs.clone()]);
        div_y
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[self.clone()]);
        neg_y
    }
}


pub mod core;
pub mod functions;  
　　　　　　　　　　　// <- 今後新しいファイルができたらここに追加
```

---

**mod.rs** の中身
```rust
pub mod math;
pub mod matrix; 
　　　　　　　　　　// <- 今後新しいファイルができたらここに追加
```


基本的には今まで実装してきた関数のコードをそのままファイルに貼り付けるだけですが、一つだけ変更しなければならない点があります。

## pubをつける
今まで実装して関数を外部クレートとして読み込むようになるため、一部の関数を外からアクセスできるよう**pub** を付けます。これはユーザーがライブラリのあらゆる関数にアクセスできるのを防ぐために、外からアクセスできるものを設定するためのものです。

色々説明してきましたが、移行の仕方は実際のコードを見るのが一番手っ取り早いです。ここからリポジトリの[stucrs](https://github.com/chitono/StuCrs/tree/main/stucrs)を見てどう移すか、またどの関数に**pub**を付けるのかを確認してください。