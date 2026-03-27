# Log関数の実装

ではクロスエントロピー誤差に必要な関数である **Log関数** を実装していきます。この関数は以前の **Function構造体** なので同じように実装します。ただし、オプションが多いので、微分の式など処理が多少複雑なので、説明します。   

<br>

まず数学の分野では明示しなければLog関数の底は\\(e\\)というのが暗黙の了解なので、**sum関数** の時ど同様、引数が**None** の時を底\\(e\\)とします。そして底が他の値の場合は**Some(base)** として渡すようにします。もちろんですが、**base** の型はもちろん普通の数値と同じ **f32** です。   

また、この底の値で微分のふるまいが異なります。正確に言えば、統一した方法で計算するは可能ですが、分けた方が、数学的な理解としても、パフォーマンス的にしても良いので(\\(e\\)以外の底の値を使用することはまれなので)、分けます。


では実装する前にlogの微分を考えてます。今回は底で場合分けして考えます。まず\\(y = \log x\\)、もしくは\\(y = \ln x\\)の時の微分は

$$\frac{dy}{dx} = \frac{1}{x}$$

となります。次に底が指定された場合、つまり\\(y = \log_a x\\)の時の微分は

$$\frac{dy}{dx} = \frac{1}{x\cdot \ln a}$$

となります。ではこれらをもとに **Log関数** を実装していきます。

```rust
struct Log {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    base: Option<f32>,
    generation: i32,
    id: usize,
}

impl Function for Log {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Logは一変数関数です。inputsの個数が一つではありません。")
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
        let base = self.base;
        let x = &xs[0];
        let y_data;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            y_data = x.data().mapv(|x| x.log(base_data));
        } else {
            y_data = x.data().mapv(|x| x.ln());
        }
        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx;

        let base = self.base;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            gx = 1.0.rv() / (x.clone() * base_data.ln().rv()) * gy.clone();
        } else {
            gx = (1.0.rv() / x.clone()) * gy.clone();
        }
        let gxs = vec![gx];
        gxs
    }

    fn get_inputs(&self) -> &[RcVariable] {
        &self.inputs
    }

    fn get_output(&self) -> RcVariable {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        RcVariable(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
}
impl Log {
    fn new(inputs: &[RcVariable], base: Option<f32>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            base: base,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn log(x: &RcVariable, base: Option<f32>) -> RcVariable {
    let y = log_f(&[x.clone()], base);
    y
}

fn log_f(xs: &[RcVariable], base: Option<f32>) -> RcVariable {
    Log::new(xs, base).borrow_mut().call()
}
```

底の指定を **Sum関数** と同じように**Option型** で渡し、**forward,backward** で場合分けして処理します。log関数の計算に慣れていればそれほど難しくはないでしょう。

では底で場合分けしてテストします。

```rust
#[test]
    fn log_test() {
        use crate::core_new::ArrayDToRcVariable;

        let a = array![3.0, 3.0, 3.0].rv();
        let b = array![3.0, 3.0, 3.0].rv();

        let mut y0 = log(&a, None); //底がe
        let mut y1 = log(&b, Some(2.0)); //底が2.0

        println!("y0 = {}", y0.data()); // 1.098...
        println!("y1 = {}", y1.data()); // 1.584...

        y0.backward(false);
        y1.backward(false);

        println!("a_grad = {:?}", a.grad().unwrap().data()); // 0.3333...
        println!("b_grad = {:?}", b.grad().unwrap().data()); // 0.4808...
    }
```