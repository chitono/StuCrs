# 微分の実装（複雑な関数）

## idの設定
世代を持たせる前に、idというものを設定します。さらに関数が複雑になり、変数(**Variable**)と関数(**Function**)が増えてくると、管理が大変なため、今後のためにそれぞれの構造体にidをつけます。idを個々の構造体に付与することで、後の複雑な関数の自動微分でバグが起きないように安全に処理することができます。

## idの生成
idを生成する**NEXT_ID**という**グローバル変数**を設定します。この変数はどこのプログラムからでもアクセスできる変数です。扱いやすいですが、同時にアクセスし、変更することができるので、安全ではありません。そこで使われるのが**Atomic**型です。複数のスレッドで同時に使用される変数の場合に使われる型です。**AtomicUsize**はその中のusize型を扱うものです。これを用いてグローバル変数を設定します。この変数の現在の値に1を加算し、生成されたVariableや関数のidとして渡します。そのようにして構造体が作成されるたびに新しいidが付与されます。イメージとしては整理券です。NEXT_IDは整理券を発行していて、券を取るたびに番号が1ずつ増えていくのです。このいわば券を取り、番号を1足す作業を**id_generator()**として関数にします。この関数を呼び出せば、その時点のidが返されます。次のVariableやFunction構造体の変更のところで用います。

```rust
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
/// idを生成する関数。構造体のコンストラクタを作成する際に、呼び出して、idを付ける
fn id_generator() -> usize {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

//例
fn main() {
    let first = id_generator();
    println!("first = {}",first);
    let second = id_generator();
    println!("second = {}",second);
}
```



### Variableの変更
Variableに **id** というフィールドを持たせましょう。このidの値を**usize型**として保持します。
```rust
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    name: Option<String>,
    id: usize,
}

impl Variable {
    fn new_rc(data: ArrayD<f32>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Variable {
            data: data,
            grad: None,
            creator: None,
            name: None,
            id: id_generator(),
        }))
    }
}
```
この時、idは先ほど作成した **id_generator()** 関数を用いることで、正しいidが付与されます。このidは呼び出されるたびに新たなidを返すので、重複することはありません。



### Functionトレイトおよび構造体の変更
はじめにFunctionトレイトを変更します。具体的にはFunction構造体のidを返す関数を追加します。


```rust
trait Function: Debug {
    fn call(&mut self) -> RcVariable;

    
    fn forward(&self, x: &[RcVariable]) -> RcVariable;
    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable>;

    //　関数クラス.inputs, .outputではvariableのbackwardの中でアクセスできないので、関数にして取得
    fn get_inputs(&self) -> &[RcVariable];
    fn get_output(&self) -> RcVariable;
    fn get_id(&self) -> usize;  //  <--  今回追加するもの 
}
```
トレイトにidを返す関数を追加する理由は、5.3で説明した**get_input** 、**get_output** と同じです。idフィールドもinputsフィールド、outputフィールドと同様に保持しているか不明なので、トレイト内で関数として定義しているということです。これは後の**generation** でも同じことなので、同じように**get_generation** を追加することになります。   

次にFunction構造体の変更です。Exp構造体を例にして変更しますので、その他のものはそれに従って変更してください。
```rust
struct Exp {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    id:usize,
}

impl Function for Exp {
    fn get_id(&self) -> usize {
        self.id
    }
}
```
先ほどのget_id()をFunctionトレイトして実装します。



以上により、VariableとFunctionトレイト・構造体の **id** への対応ができました。  


## 世代（ジェネレーション）の保持
では前節で説明した世代を持たせ、それに従って処理するよう変更していきます。

### Variableの変更
まずVariableに **generation** というフィールドを持たせましょう。このgenerationには世代の値を保持します。例えば前のグラフのXは第0世代なので0という値を持ちます。
```rust
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    name: Option<String>,
    id: usize,
    generation: i32,

}

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
}
```
generationの値は0以上の整数のみを扱うので、i32型に設定します。はじめは0として設定し、次に初期化したgenerationを正しいgenerationに変更する処理を加えます。

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

    fn set_creator(&mut self, func: Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::clone(&func));
        self.generation = func.borrow().get_generation() + 1;
    }
}
```
Variableのメソッドとしてset_creatorがありますが、これを変更します。Variableが自分のcreatorを覚える関数ですが、その時に自分の世代を、creatorの世代に1足して設定します。この作業は関数がoutputを出力するとき、自身の世代に1足してoutputに持たせる作業を指します。  

### Function構造体の変更
次にFunction構造体の変更です。Exp構造体を例にして変更しますので、その他のものはそれに従って変更してください。
```rust
struct Exp {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id:usize,
}

impl Function for Exp {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Expは一変数関数です。inputsの個数が一つではありません。")
        }

        let output = self.forward(inputs);

        //inputのgenerationで一番大きい値をFuncitonのgenerationとする
        self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる
        output.0.borrow_mut().set_creator(self_f.clone()); //先ほどset_creator()を変更したので、Variableの世代はここ関数によって変更される。
        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
}
```
はじめにVariableと同じく、構造体に**generation**フィールドを持たせましょう。初期値もVariableと同じく0です。  

次にcallの中で、世代をinputのVariableの世代に設定します。多変数関数の場合、inputが複数存在するので、inputの世代で最も値が大きいものを採用します。  

また、自身の世代を取り出す関数(get_generation)を作成します。この関数は　**FunctionTrait** で定義しましたが、その理由は後の **関数を取り出す処理の変更** のところで説明します。

以上により、VariableとFunction構造体のgenerationへの対応ができました。  

### 関数を取り出す処理の変更
前節の説明の通りに、Variableのbackwardの **funcsベクタ** から関数を取り出す処理を変更します。

```rust
impl Variable {
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

        //&selfで最初の変数はborrowされるので場合分け
        let mut last_variable = true;
        while let Some(f_rc) = funcs.pop() {
            //println!("f = {:?}\n", get_struct_name(&f_rc.borrow()));
            let f_borrowed = f_rc.borrow();
            let xs = f_borrowed.get_inputs();
            let y = f_borrowed.get_output();

            let y_grad: RcVariable;

            if last_variable {
                y_grad = ArrayD::<f32>::ones(self.data.shape()).rv();

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
backwardの中に **add_func()** という関数を追加します。この関数がまさに、**funcsベクタ** から関数を正しく取り出すための処理です。この **add_funcs** 関数が行っていることは主に二つです。一つ目は **関数の重複確認** です。

はじめに **backward** 内で **funcs** とは別に新たな配列 **seen_set** というものを用意します。これは **HashSet** という型です。**HashSet** とは普通の配列とは異なり、値の重複を検知することができます。これは **funcs** に関数が追加される際、今までに追加された関数と同じもの、すなわち重複したものが間違えて入っていないかを確かめるためのものです。前節の **微分の理論** を思い出してください。間違えた処理の場合、関数Aを2回取り出してしまいました。そこで **funcs** に追加された関数のidを記憶しておくことで、後に追加されるものが今までのものと重複しないかを確認し、もし重複していたら **funcs** に追加しないという処理を行えばよいのです。このような処理を加えることで、**バグの温床を減らす設計にすることができます。**  **if seen_set.insert(f.borrow().get_id())** のところでidを用いて重複しているかどうか確認します。もし重複していなかったら、**seen_set** にidを追加し、**funcs** に関数を追加します。もし重複していたら、今説明した処理は行われません。


二つ目は世代順への並び替えです。 **funcs** から関数が取り出されるときは、一番後ろから取り出されるので、世代の小さい順に並び変えてあげれば、世代の小さい方が先に取り出されるということを防げます。これを、**Vec型** で提供される **sort_by()** を用いて並べ替えます。

>**HashSet** や、**sort_by** についてはgithubリポジトリの[REFERENCES.md](https://github.com/chitono/StuCrs/blob/main/REFERENCES.md)の文献をご覧ください。

これらの設定により、 **add_funcs** を用いて **funcs** に関数を追加していけば、正しく関数を取り出せるようになりました。なので、あとはコードの最後の **//creatorがあるならその関数をfuncsに追加** のところをadd_funcsに変更すればよいだけです。

TODO: 並び替えや取り出す処理を簡単に説明するコード追加予定
TODO: 複雑な関数がうまくバックプロパゲーションしてくれるか確かめるコード追加予定
