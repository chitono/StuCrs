use core::panic;
use std::cell::RefCell;
use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use std::rc::{Rc, Weak};

//use std::thread;
//use std::time::Duration;
use std::time::Instant;

use std::ops::{Add, Div, Mul, Neg, Sub};
//use std::ops::Div;
//use std::ops::Sub;

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

fn main() {
    let start = Instant::now();

    let x = 4.0.rv();
    //let b = 2.0.rv();
    //let c = RcVariable::new(1.0);

    //b[0].as_ref().unwrap().borrow_mut().name = Some("b".to_string());

    let mut y = x.clone() / 2.0.rv();

    y.backward();
    println!("y={:?}", y.clone());

    println!("x={:?}", x.clone());
    //println!("b={:?}", b);
    //println!("c={:?}", c);

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}

#[derive(Debug, Clone)]
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    name: Option<String>,
    generation: i32,
    id: u32,
}
impl Variable {
    fn new(data: f32) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Variable {
            data,
            grad: None,
            creator: None,
            name: None,
            generation: 0,
            id: id,
        }
    }

    fn new_rc(data: f32) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Variable {
            data,
            grad: None,
            creator: None,
            name: None,
            generation: 0,
            id: id,
        }))
    }

    fn set_creator(&mut self, func: Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::clone(&func));
        self.generation = func.borrow().get_generation() + 1;
    }

    fn backward(&self) {
        let mut funcs: Vec<Rc<RefCell<dyn Function>>> =
            vec![Rc::clone(self.creator.as_ref().unwrap())];

        let mut seen_set = HashSet::new();

        /*
        if !seen_set.insert(user) {
            println!("重複しています: {:?}", user);
        } else {
            println!("重複していません: {:?}", user);
        } */

        fn add_func(
            funcs_list: &mut Vec<Rc<RefCell<dyn Function>>>,
            seen_set: &mut HashSet<u32>,
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

        while let Some(f) = funcs.pop() {
            let xs = f.borrow().get_inputs();

            let mut y_grad: [Option<f32>; 2] = [None, None];

            if last_variable {
                y_grad = [Some(1.0), None];

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし
                y_grad[0] = f.borrow().get_outputs()[0]
                    .as_ref()
                    .unwrap()
                    .borrow_mut()
                    .grad;
            }

            let xs_grad = f.borrow().backward(y_grad);

            // gradを置き換えまたは足していくので、Noneか判別
            let current_grad_0 = xs[0]
                .clone()
                .as_ref()
                .unwrap()
                .borrow_mut()
                .grad
                .unwrap_or_else(|| 0.0);

            xs[0].as_ref().unwrap().borrow_mut().grad = Some(current_grad_0 + xs_grad[0].unwrap());

            //xs[0]にcreatorがあるか確認、あったらfuncに追加
            if let Some(func_creator) = &xs[0].as_ref().unwrap().borrow().creator {
                add_func(&mut funcs, &mut seen_set, func_creator.clone());
                //funcs.push(Rc::clone(&func_creator));
            }

            //xs[1]はfが一変数関数の時、NoneなのでNoneか判別
            if let Some(xs_1) = xs[1].clone() {
                let current_grad_1 = xs_1.borrow_mut().grad.unwrap_or_else(|| 0.0);

                xs_1.borrow_mut().grad = Some(current_grad_1 + xs_grad[1].unwrap());

                //xs[1]にcreatorがあるか確認、あったらfuncに追加
                if let Some(func_creator) = &xs_1.borrow().creator {
                    add_func(&mut funcs, &mut seen_set, func_creator.clone());
                    //funcs.push(Rc::clone(&func_creator));
                }
            }
        }
    }

    fn cleargrad(&mut self) {
        self.grad = None;
    }
}

#[derive(Debug, Clone)]
struct RcVariable(Rc<RefCell<Variable>>);

impl RcVariable {
    fn new(data: f32) -> Self {
        RcVariable(Variable::new_rc(data))
    }

    fn backward(&mut self) {
        self.0.borrow_mut().backward();
    }

    fn pow(self, c: f32) -> RcVariable {
        let pow_y = pow(&[Some(self.0.clone()), None]);
        RcVariable(pow_y[0].as_ref().unwrap().clone())
    }
}

trait Function: Debug {
    fn call(
        &mut self,
        input: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2];

    //  forward,backwardはVariableの数値のみを計算する
    fn forward(&self, x: [Option<f32>; 2]) -> [Option<f32>; 2];
    fn backward(&self, gy: [Option<f32>; 2]) -> [Option<f32>; 2];

    //　関数クラス.inputs, .outputsではvariableのbackwardの中でアクセスできないので、関数にして取得
    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2];
    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2];
    fn get_generation(&self) -> i32;
    fn get_id(&self) -> u32;
}

#[derive(Debug, Clone)]
struct Square {
    //Square Class
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Square {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        if let None = &inputs[0] {
            panic!("Squareは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Squareは一変数関数です。input[1]がNoneである必要があります")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();
        self.generation = inputs[0].as_ref().unwrap().borrow().generation;

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_square: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2
        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_square.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];
        let y = xs[0].expect("数値がありません").powf(2.0);

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let x0_data = self.inputs[0].as_ref().unwrap().borrow().data;

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(2.0 * x0_data * gy0_data)
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Square {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

//fn square(input_x:&Rc<RefCell<Variable>>)

fn square(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    //xを一変数から配列に変換

    Square::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Exp {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Exp {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        if let None = &inputs[0] {
            panic!("Expは一変数関数です。input[0]がNoneです")
        }
        if let Some(variable) = &inputs[1] {
            panic!("Expは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();
        self.generation = inputs[0].as_ref().unwrap().borrow().generation;

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_exp: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2
        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_exp.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];
        let y = xs[0].expect("数値がありません").exp();

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let x0_data = self.inputs[0].as_ref().unwrap().borrow().data;

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(x0_data.exp() * gy0_data)
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Exp {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn exp(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    //xを一変数から配列に変換

    Exp::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Add_f {
    //Add Class
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Add_f {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Addは二変数関数です。input[0]がNoneです。")
        }

        if let None = &inputs[1] {
            panic!("Addは二変数関数です。input[1]がNoneです。")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );
        xs_data[1] = Some(
            inputs[1]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();

        //inputsのgenerationをそれぞれ取得
        let input_0_generation = inputs[0].as_ref().unwrap().borrow().generation;
        let input_1_generation = inputs[1].as_ref().unwrap().borrow().generation;

        //inputのgenerationで大きい値の方をFuncitonのgenerationとする
        self.generation = match input_0_generation >= input_1_generation {
            true => input_0_generation,
            false => input_1_generation,
        };

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_add: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_add.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        // y = xs[0] + xs[1]
        let y = xs[0].expect("数値が存在するはず in forward")
            + xs[1].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let gy = gys[0].clone();
        let gxs = [gy, gy];

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Add_f {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn add(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Add_f::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Mul_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Mul_f {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Mulは二変数関数です。input[0]がNoneです。")
        }

        if let None = &inputs[1] {
            panic!("Mulは二変数関数です。input[1]がNoneです。")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );
        xs_data[1] = Some(
            inputs[1]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();

        //inputsのgenerationをそれぞれ取得
        let input_0_generation = inputs[0].as_ref().unwrap().borrow().generation;
        let input_1_generation = inputs[1].as_ref().unwrap().borrow().generation;

        //inputのgenerationで大きい値の方をFuncitonのgenerationとする
        self.generation = match input_0_generation >= input_1_generation {
            true => input_0_generation,
            false => input_1_generation,
        };

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_mul: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_mul.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        let y = xs[0].expect("数値が存在するはず in forward")
            * xs[1].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let x0_data = self.inputs[0].as_ref().unwrap().borrow().data;
        let x1_data = self.inputs[1].as_ref().unwrap().borrow().data;

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(gy0_data * x1_data);
            gxs[1] = Some(gy0_data * x0_data);
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Mul_f {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn mul(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Mul_f::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Sub_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Sub_f {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Subは二変数関数です。input[0]がNoneです。")
        }

        if let None = &inputs[1] {
            panic!("Subは二変数関数です。input[1]がNoneです。")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );
        xs_data[1] = Some(
            inputs[1]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();

        //inputsのgenerationをそれぞれ取得
        let input_0_generation = inputs[0].as_ref().unwrap().borrow().generation;
        let input_1_generation = inputs[1].as_ref().unwrap().borrow().generation;

        //inputのgenerationで大きい値の方をFuncitonのgenerationとする
        self.generation = match input_0_generation >= input_1_generation {
            true => input_0_generation,
            false => input_1_generation,
        };

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_sub: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_sub.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        let y = xs[0].expect("数値が存在するはず in forward")
            - xs[1].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(gy0_data);
            gxs[1] = Some(-gy0_data);
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Sub_f {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn sub(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Sub_f::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Div_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Div_f {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Divは二変数関数です。input[0]がNoneです。")
        }

        if let None = &inputs[1] {
            panic!("Divは二変数関数です。input[1]がNoneです。")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );
        xs_data[1] = Some(
            inputs[1]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();

        //inputsのgenerationをそれぞれ取得
        let input_0_generation = inputs[0].as_ref().unwrap().borrow().generation;
        let input_1_generation = inputs[1].as_ref().unwrap().borrow().generation;

        //inputのgenerationで大きい値の方をFuncitonのgenerationとする
        self.generation = match input_0_generation >= input_1_generation {
            true => input_0_generation,
            false => input_1_generation,
        };

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_div: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_div.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        let y = xs[0].expect("数値が存在するはず in forward")
            / xs[1].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let x0_data = self.inputs[0].as_ref().unwrap().borrow().data;
        let x1_data = self.inputs[1].as_ref().unwrap().borrow().data;

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(gy0_data / x1_data);
            gxs[1] = Some(gy0_data * (-x0_data / x1_data.powf(2.0)));
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Div_f {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn div(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Div_f::new().borrow_mut().call(xs)
}
#[derive(Debug, Clone)]
struct Neg_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Neg_f {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Negは一変数関数です。input[0]がNoneです。")
        }

        if let Some(_variable) = &inputs[1] {
            panic!("Negは一変数関数です。input[1]がNoneである必要があります")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();
        self.generation = inputs[0].as_ref().unwrap().borrow().generation;

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_neg: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_neg.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        let y = -xs[0].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(-gy0_data);
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Neg_f {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn neg(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Neg_f::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Pow_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
    generation: i32,
    id: u32,
}

impl Function for Pow_f {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Powは一変数関数です。input[0]がNoneです。")
        }

        if let Some(_variable) = &inputs[1] {
            panic!("Powは一変数関数です。input[1]がNoneである必要があります")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new_rc(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();
        self.generation = inputs[0].as_ref().unwrap().borrow().generation;

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_pow: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_pow.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        let y = -xs[0].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(-gy0_data);
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Pow_f {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
            generation: 0,
            id: id,
        }))
    }
}

fn pow(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Pow_f::new().borrow_mut().call(xs)
}

//演算子のオーバーロード

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(add_y[0].as_ref().unwrap().clone())
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(mul_y[0].as_ref().unwrap().clone())
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(sub_y[0].as_ref().unwrap().clone())
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(div_y[0].as_ref().unwrap().clone())
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[Some(self.0.clone()), None]);
        RcVariable(neg_y[0].as_ref().unwrap().clone())
    }
}

trait f32_to_RcVariable {
    fn rv(&self) -> RcVariable;
}

//rustの数値のデフォルトがf64なので、f32に変換してからRcVariableを生成
impl f32_to_RcVariable for f64 {
    fn rv(&self) -> RcVariable {
        RcVariable::new(*self as f32)
    }
}

/*
#[cfg(test)]
impl Drop for Mul_f {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
} */
