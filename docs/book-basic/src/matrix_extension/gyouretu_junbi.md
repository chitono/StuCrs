# 行列計算への前準備
いままで私たちは変数として主に「 **スカラ** 」を扱ってきました。しかし機械学習ではベクトルや行列などの「 **テンソル** 」を扱ってきます。本章ではテンソルを使うための前準備としてフレームワークを行列へと拡張する方法について説明していきます。
今回は多次元配列を扱うライブラリとして[ndarray](https://docs.rs/ndarray/latest/ndarray/)というライブラリを使用します。
これはpythonでいうところの **Numpy** に相当します。使い方に関しては上のリンクを参照してください。

## f32からArrayDへの変更
いままではf32型を用いてStuCrsフレームワークを構築していました。しかしこれから私たちは「テンソル」を扱いたいです。そこで「ndarray」ライブラリのArrayD型を使用しようと思います。   

**ArrayD型** とは実行中に次元数が決定する「テンソル」を扱う **動的**な型です。
なぜArray1型やArray2型などの型を使わずArrayD型を使用するかというと私たちのこれから複雑な計算をさせる関数を実装するときに、ある関数では入力データが４次元であったり、また他の関数では入力データが２次元であったりするためです。これらのようにコンパイル時に次元数がわからず、様々な次元数を用いる関数を共通して管理したいためArrayD型を用います。
では実際にf32型をArrayD型に変更してみましょう。

### Variableの変更
はじめに構造体の変更です。**dataフィールド** の型を **f32** から **ArrayD&lt;f32&gt;** に変更します。

```rust
use ndarray::{array,ArrayD};
struct Variable {
    pub data: ArrayD<f32>,     // <- f32から変更
    grad: Option<RcVariable>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    pub name: Option<String>,
    pub generation: i32,
    pub id: usize,
}
```
今回用いるArrayDの要素の型はf32に指定しました。ジェネリック型を用いれば、f64など、他の型にも対応できるようになりますが、コードが少し複雑になるため、f32に限定します。  

```rust
impl Variable {
    pub fn new_rc(data: ArrayD<f32>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Variable {
            data: data,
            grad: None,
            creator: None,
            name: None,
            generation: 0,
            id: id_generator(),
        }))
    }

    pub fn set_creator(&mut self, func: Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::clone(&func));
        self.generation = func.borrow().get_generation() + 1;
    }

    fn backward(&self, double_grad: bool) {
        let mut funcs: Vec<Rc<RefCell<dyn Function>>> =
            vec![Rc::clone(self.creator.as_ref().unwrap())];

        let mut seen_set = HashSet::new();

        fn add_func(
            funcs_list: &mut Vec<Rc<RefCell<dyn Function>>>,
            seen_set: &mut HashSet<usize>,
            f: Rc<RefCell<dyn Function>>,
        ) {
            if seen_set.insert(f.borrow().get_id()) {
                funcs_list.push(Rc::clone(&f));
                funcs_list.sort_by(|a, b| {
                    a.borrow()
                        .get_generation()
                        .cmp(&b.borrow().get_generation())
                });
            }
        }
        //let first_grad = ArrayD::<f32>::ones(self.data.shape()).rv();

        //&selfで最初の変数はborrowされるので場合分け
        let mut last_variable = true;

        while let Some(f_rc) = funcs.pop() {

            let f_borrowed = f_rc.borrow();
            if double_grad == true {
                set_grad_true();
            } else {
                set_grad_false();
            }

            let xs = f_borrowed.get_inputs();

            let y = f_borrowed.get_output();

            let y_grad: RcVariable;

            if last_variable {
                y_grad = ArrayD::<f32>::ones(self.data.shape()).rv(); // <- 変更点

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし
                y_grad = y.0.borrow().grad.as_ref().unwrap().clone();
            }

            let xs_grad = f_borrowed.backward(&y_grad);

            for (x, x_grad) in zip(xs, &xs_grad) {
                // gradをすでに保持しているなら、元のgradに新たなgradを足す。
                // gradをまだ持っていないならそれを持たせる。
                if let Some(current_grad_data) = x.grad() {
                    x.0.borrow_mut().grad = Some(current_grad_data + x_grad.clone());
                } else {
                    x.0.borrow_mut().grad = Some(x_grad.clone());
                }

                // creatorがあるならその関数をfuncsに追加
                if let Some(func_creator) = &x.0.borrow().creator {
                    add_func(&mut funcs, &mut seen_set, func_creator.clone());
                }
            }
        }
        
    }
}
```
続いてVariableに実装したメソッドの変更点です。変更する点は二つです。一つ目は **new_rc()** の引数です。 **data** の型をf32からArrayDにしてください。二つ目は **y_grad** です。矢印で示したところをArrayDにして対応させてください。
>**y_grad** を変更する前に、下の説明の **rv()** の変更を先に行ってください。   

f32からArrayDへの変更であまり変更するところがない理由は、**データの値をVariableで扱っているため、Variableのデータさえ変更してしまえばあとは比較的同じように動作してくれるからです。** ArrayD型にも演算子などといった様々なメソッドが定義されているため、f32と同じように同じふるまいをしてくれるというわけです。

### ArrayDからRcVariableを生成する関数
前回、f32からRcVariableを生成する関数、**rv()** を実装しましたが、これも **ArrayD** から **RcVariable** を生成する関数も実装しましょう。
```rust
//array型からRcVariable型を生成
pub trait ArrayDToRcVariable {
    fn rv(&self) -> RcVariable;
}
//arrayは任意の次元に対応
impl<D: Dimension> ArrayDToRcVariable for ArrayBase<OwnedRepr<f32>, D> {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.view().into_dyn())
    }
}

pub trait F32ToRcVariable {
    fn rv(&self) -> RcVariable;
}

//f32からarray型に変換し、rv()でRcVariableを生成
impl F32ToRcVariable for f32 {
    fn rv(&self) -> RcVariable {
        let array = array![*self as f32]; // <- f32型をArray型に変換
        array.rv()                        // <- Array型を通してRcVariableを生成
    }
}
```
ArrayD型にもf32型と同様に、 **rv()** メソッドを定義し、オーバーロードします。この際、ArrayDはf32とは異なりプリミティブ型ではないので、ただ単に渡せばよいものではありません。 **view()** や **into_dyn()** に関しては **ArrayD** のドキュメントを参照してください。Array型からRcVariableへの変換を基準にして、f32からの変換の場合は、f32型をArray型に変換してからRcVariableを生成するという流れにします。
```rust
fn main() {
    let a = array![2.0,3.0].rv(); // <- 新たな生成方法
}
```
するとこのようにArray型の行列からRcVariableを生成できるようになりました。


