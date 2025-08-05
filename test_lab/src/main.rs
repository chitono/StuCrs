use core::panic;
use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::atomic::{self, AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::{
    array, Array, ArrayBase, ArrayD, ArrayView, ArrayView2, ArrayViewD, Data, Dim, Dimension, IntoDimension, IxDyn, IxDynImpl, OwnedRepr, Shape
};
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;
use std::time::Instant;

use std::ops::{Add, Div, Mul, Neg, Sub};

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

static GRAD_CONFIG: Mutex<bool> = Mutex::new(true);

fn set_grad_true() {
    let mut flag = GRAD_CONFIG.lock().unwrap();
    *flag = true;
}

fn set_grad_false() {
    let mut flag = GRAD_CONFIG.lock().unwrap();
    *flag = false;
}

fn get_grad_status() -> bool {
    let flag = GRAD_CONFIG.lock().unwrap();
    *flag
}

/*

fn sphere(x: &RcVariable, y: &RcVariable) -> RcVariable {
    let z = x.clone().pow(2.0) + y.clone().pow(2.0);
    z
}

fn matyas(x: &RcVariable, y: &RcVariable) -> RcVariable {
    let z = 0.26.rv() * (x.pow(2.0) + y.pow(2.0)) - 0.48.rv() * x.clone() * y.clone();
    z
}

fn goldstein(x: &RcVariable, y: &RcVariable) -> RcVariable {
    let z = (1.0.rv()
        + (x.clone() + y.clone() + 1.0.rv()).pow(2.0)
            * (19.0.rv() - 14.0.rv() * x.clone() + 3.0.rv() * x.pow(2.0) - 14.0.rv() * y.clone()
                + 6.0.rv() * x.clone() * y.clone()
                + 3.0.rv() * y.pow(2.0)))
        * (30.0.rv()
            + (2.0.rv() * x.clone() - 3.0.rv() * y.clone()).pow(2.0)
                * (18.0.rv() - 32.0.rv() * x.clone()
                    + 12.0.rv() * x.clone().pow(2.0)
                    + 48.0.rv() * y.clone()
                    - 36.0.rv() * x.clone() * y.clone()
                    + 27.0.rv() * y.pow(2.0)));
    z
}

fn rosenbrock(x0: &RcVariable, x1: &RcVariable) -> RcVariable {
    let y =
        100.0.rv() * (x1.clone() - x0.clone().pow(2.0)).pow(2.0) + (x0.clone() - 1.0.rv()).pow(2.0);
    y
}

fn f(x: &RcVariable) -> RcVariable {
    let y = x.clone().pow(4.0) - 2.0.rv() * x.clone().pow(2.0);
    y
}

fn gx2(x: &RcVariable) -> RcVariable {
    let y = 12.0.rv() * x.clone().pow(2.0) - 4.0.rv();
    y
} */

fn main() {
    let start = Instant::now();

    //set_grad_false();
    let iters = 1;
    for _i in 0..iters {
        let x0 = array![[1.0f32, 2.0, 3.0],[4.0, 5.0, 6.0]].rv();
        //let x1 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        //let  shape_array = [1,6];

    // `&[usize; 2]`を`IxDyn`に変換
        //let dyn_shape = IxDyn(&shape_array);
        let mut y = x0.T() ;
        y.backward();

        println!("y_data={:?}\n",y.data());

        println!("x0_grad={:?}\n",x0.grad());

        //println!("x2_grad={:?}\n",x2.grad());
    }
    /*

    let lr = 0.001;
    let iters = 1000;






    for i in 0..iters {



        println!("{:?}, {:?}",x0.data() ,x1.data());


        let mut y = rosenbrock(&x0, &x1);


        x0.cleargrad();
        x1.cleargrad();
        y.backward();




        let current_data_0 = x0.data();
        let current_data_1 = x1.data();

        let current_grad_0 =x0.grad().unwrap();
        let current_grad_1 =x1.grad().unwrap();

        x0.0.borrow_mut().data =current_data_0- lr*current_grad_0;
        x1.0.borrow_mut().data = current_data_1- lr*current_grad_1;




    }*/
    //println!("(x0,x1)=({:?},{:?})", x0.0.borrow().data,x1.0.borrow().data);

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration / iters);
}

#[derive(Debug, Clone)]
struct Variable {
    data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    name: Option<String>,
    generation: i32,
    id: u32,
}
impl Variable {
    fn new_rc(data: ArrayD<f32>) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Variable {
            data: data,
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
        let first_grad = ArrayD::<f32>::ones(self.data.shape());

        //&selfで最初の変数はborrowされるので場合分け
        let mut last_variable = true;

        while let Some(f_rc) = funcs.pop() {
            let xs = f_rc.borrow().get_inputs();
            let y_rc;
            let y;

            let y_grad;

            if last_variable {
                y_grad = first_grad.view();

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし
                y_rc = f_rc.borrow().get_output();
                y = y_rc.borrow();
                y_grad = y.grad.as_ref().unwrap().view();
            }

            let xs_grad = f_rc.borrow().backward(y_grad.clone());

            // gradを置き換えまたは足していくので、Noneか判別
            let mut xs_0 = xs[0].as_ref().unwrap().borrow_mut();
            let current_grad_0 = xs_0
                .grad
                .as_ref()
                .cloned()
                .unwrap_or_else(|| ArrayD::<f32>::zeros(xs_0.data.shape()));

            xs_0.grad = Some(&current_grad_0 + xs_grad[0].as_ref().unwrap());

            //xs[0]にcreatorがあるか確認、あったらfuncに追加
            if let Some(func_creator) = &xs_0.creator {
                add_func(&mut funcs, &mut seen_set, func_creator.clone());
                //funcs.push(Rc::clone(&func_creator));
            }

            //xs[1]はfが一変数関数の時、NoneなのでNoneか判別
            if let Some(xs_1) = &xs[1] {
                let current_grad_1 = xs_1
                    .borrow()
                    .grad
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| Array::zeros(xs_1.borrow().data.dim()));

                xs_1.borrow_mut().grad = Some(current_grad_1 + xs_grad[1].as_ref().unwrap());

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
    fn new(data: ArrayViewD<f32>) -> Self {
        RcVariable(Variable::new_rc(data.to_owned()))
    }

    fn backward(&mut self) {
        self.0.borrow_mut().backward();
    }

    fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    fn cleargrad(&mut self) {
        self.0.borrow_mut().cleargrad();
    }

    

    fn pow(&self, c: f64) -> RcVariable {
        let y = pow(&[Some(self.0.clone()), None], c);
        RcVariable(y.clone())
    } 

    fn exp(&self) -> RcVariable {
        let y = exp_f(&[Some(self.0.clone()), None]);
        RcVariable(y.clone())
    }
    
    fn reshape(&self,shape: IxDyn) -> RcVariable {
        let y = reshape_f(&[Some(self.0.clone()), None],shape);
        RcVariable(y.clone())
    }

    fn T(&self) -> RcVariable {
        let y = transpose_f(&[Some(self.0.clone()), None]);
        RcVariable(y.clone())
    }


    
}

trait Function: Debug {
    fn call(&mut self, input: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>>;

    //  forward,backwardはVariableの数値のみを計算する
    fn forward(&self, x: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32>;
    fn backward(&self, gy: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2];

    //　関数クラス.inputs, .outputではvariableのbackwardの中でアクセスできないので、関数にして取得
    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2];
    fn get_output(&self) -> Rc<RefCell<Variable>>;
    fn get_generation(&self) -> i32;
    fn get_id(&self) -> u32;
}

#[derive(Debug, Clone)]
struct Square {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Square {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Squareは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Squareは一変数関数です。input[1]がNoneである必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output: Rc<RefCell<Variable>>;

        output = Variable::new_rc(ys_data);

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y = xs[0]
            .as_ref()
            .expect("数値がありません")
            .mapv(|x| x.powf(2.0));

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();

        let x0_data = x0_borrow.data.view();

        gxs[0] = Some(2.0 * &x0_data * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

//fn square(input_x:&Rc<RefCell<Variable>>)

fn square(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    //xを一変数から配列に変換

    let y = Square::new().borrow_mut().call(xs);

    let output_list = [Some(y), None];
    output_list
}

#[derive(Debug, Clone)]
struct Exp {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Exp {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Expは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Expは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y = xs[0].as_ref().expect("数値がありません").mapv(|x| x.exp());

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x0_data = x0_borrow.data.view();

        gxs[0] = Some(x0_data.mapv(|gx| gx.exp()) * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn exp(x: &RcVariable) -> RcVariable {
    let y = exp_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn exp_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Exp::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Sin {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Sin {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Sinは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Sinは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y = xs[0].as_ref().expect("数値がありません").mapv(|x| x.sin());

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x0_data = x0_borrow.data.view();

        gxs[0] = Some(x0_data.mapv(|gx| gx.cos()) * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Sin {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn sin(x: &RcVariable) -> RcVariable {
    let y = sin_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn sin_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Sin::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Cos {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Cos {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Cosは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Cosは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y = xs[0].as_ref().expect("数値がありません").mapv(|x| x.cos());

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x0_data = x0_borrow.data.view();

        gxs[0] = Some(x0_data.mapv(|gx| -gx.sin()) * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Cos {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn cos(x: &RcVariable) -> RcVariable {
    let y = cos_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn cos_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Cos::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Tanh {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Tanh {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Tanhは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Tanhは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y = xs[0].as_ref().expect("数値がありません").mapv(|x| x.tanh());

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();

        let y_data = x0_borrow.data.view().mapv(|x| x.tanh().powf(2.0));

        gxs[0] = Some((1.0 - y_data) * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Tanh {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn tanh(x: &RcVariable) -> RcVariable {
    let y = tanh_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn tanh_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Tanh::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Add_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Add_f {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Addは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Addは二変数関数です。input[1]がNoneです")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();
        let inputs_1 = inputs[1].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());
        xs_data[1] = Some(inputs_1.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            let input_0_generation = inputs_0.generation;
            let input_1_generation = inputs_1.generation;

            //inputのgenerationで大きい値の方をFuncitonのgenerationとする
            self.generation = match input_0_generation >= input_1_generation {
                true => input_0_generation,
                false => input_1_generation,
            };

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y =
            xs[0].as_ref().expect("数値がありません") + xs[1].as_ref().expect("数値がありません");

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];

        gxs = [Some(gys.to_owned()), Some(gys.to_owned())];

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn add(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Add_f::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Mul_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Mul_f {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Mulは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Mulは二変数関数です。input[1]がNoneです")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();
        let inputs_1 = inputs[1].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());
        xs_data[1] = Some(inputs_1.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            let input_0_generation = inputs_0.generation;
            let input_1_generation = inputs_1.generation;

            //inputのgenerationで大きい値の方をFuncitonのgenerationとする
            self.generation = match input_0_generation >= input_1_generation {
                true => input_0_generation,
                false => input_1_generation,
            };

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y =
            xs[0].as_ref().expect("数値がありません") * xs[1].as_ref().expect("数値がありません");

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];

        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x1_borrow = self.inputs[1].as_ref().unwrap().borrow();

        let x0_data = x0_borrow.data.view();
        let x1_data = x1_borrow.data.view();

        gxs[0] = Some(&x1_data * &gys);
        gxs[1] = Some(&x0_data * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn mul(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Mul_f::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Sub_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Sub_f {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Subは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Subは二変数関数です。input[1]がNoneです")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();
        let inputs_1 = inputs[1].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());
        xs_data[1] = Some(inputs_1.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            let input_0_generation = inputs_0.generation;
            let input_1_generation = inputs_1.generation;

            //inputのgenerationで大きい値の方をFuncitonのgenerationとする
            self.generation = match input_0_generation >= input_1_generation {
                true => input_0_generation,
                false => input_1_generation,
            };

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y =
            xs[0].as_ref().expect("数値がありません") - xs[1].as_ref().expect("数値がありません");

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];

        gxs[0] = Some(gys.to_owned());
        gxs[1] = Some(-gys.to_owned());

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn sub(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Sub_f::new().borrow_mut().call(&xs)
}





#[derive(Debug, Clone)]
struct Div_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Div_f {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Divは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Divは二変数関数です。input[1]がNoneです")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();
        let inputs_1 = inputs[1].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());
        xs_data[1] = Some(inputs_1.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            let input_0_generation = inputs_0.generation;
            let input_1_generation = inputs_1.generation;

            //inputのgenerationで大きい値の方をFuncitonのgenerationとする
            self.generation = match input_0_generation >= input_1_generation {
                true => input_0_generation,
                false => input_1_generation,
            };

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y =
            xs[0].as_ref().expect("数値がありません") / xs[1].as_ref().expect("数値がありません");

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];

        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x1_borrow = self.inputs[1].as_ref().unwrap().borrow();

        let x0_data = x0_borrow.data.view();
        let x1_data = x1_borrow.data.view();

        gxs[0] = Some(&x1_data * &gys);
        gxs[1] = Some(&x0_data * &gys);


        gxs[0] = Some(&gys / &x1_data);
        gxs[1] = Some(&gys * (-&x0_data/&x1_data.mapv(|x|x.powf(2.0))) );

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}

fn div(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Div_f::new().borrow_mut().call(&xs)
}


#[derive(Debug, Clone)]
struct Neg_f {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Neg_f {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Expは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Expは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y = -xs[0].as_ref().expect("数値がありません");

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        
        gxs[0] = Some(-&gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
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
            output: None,
            generation: 0,
            id: id,
        }))
    }
}


fn neg(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Neg_f::new().borrow_mut().call(&xs)
}


#[derive(Debug, Clone)]
struct Pow {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    c: f32,
    generation: i32,
    id: u32,
}

impl Function for Pow {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Powは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Powは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let c = self.c;
        let y = xs[0].as_ref().expect("数値がありません").mapv(|x| x.powf(c));

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x0_data = x0_borrow.data.view();
        let c = self.c;

        gxs[0] = Some(c*x0_data.mapv(|gx| gx.powf(c-1.0)) * &gys);

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Pow {
    fn new(c:f64) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            c: c as f32,
            generation: 0,
            id: id,
        }))
    }
}



fn pow(xs: &[Option<Rc<RefCell<Variable>>>; 2],c:f64) -> Rc<RefCell<Variable>> {
    Pow::new(c).borrow_mut().call(&xs)
}



#[derive(Debug, Clone)]
struct Log {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    base: Option<f32>,
    generation: i32,
    id: u32,
}

impl Function for Log {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Logは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Logは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let base = self.base;
        let y ;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            y = xs[0].as_ref().expect("数値が存在する").mapv(|x|x.log(base_data));
        } else {
            y = xs[0].as_ref().expect("数値が存在するはず").mapv(|x|x.ln());
        }
        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x0_data = x0_borrow.data.view();
        let base = self.base;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            gxs[0] = Some(1.0/(&x0_data*base_data.ln())*&gys);
            
        } else {
            gxs[0] = Some((1.0/&x0_data)*&gys);
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Log {
    fn new(base:Option<f32>) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            base: base,
            generation: 0,
            id: id,
        }))
    }
}




fn log(x: &RcVariable,base:Option<f32>) -> RcVariable {
    let y = log_f(&[Some(x.0.clone()), None],base);
    RcVariable(y.clone())
}

fn log_f(xs: &[Option<Rc<RefCell<Variable>>>; 2],base:Option<f32>) -> Rc<RefCell<Variable>> {
    Log::new(base).borrow_mut().call(&xs)
}


/*









*/


// 行列計算用関数
#[derive(Debug, Clone)]
struct Reshape {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: u32,
}

impl Function for Reshape {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Reshapeは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Reshapeは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {
        let y_shape = self.shape.clone();
        let y = xs[0].as_ref().expect("数値がありません").to_owned().into_shape(y_shape).unwrap();

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x_shape = x_borrow.data.shape();
        
        gxs[0] = Some(gys.to_owned().into_shape(x_shape).unwrap());
        
        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Reshape {
    fn new(shape: IxDyn) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            shape: shape,
            generation: 0,
            id: id,
        }))
    }
}



fn reshape_f(xs: &[Option<Rc<RefCell<Variable>>>; 2],shape: IxDyn) -> Rc<RefCell<Variable>> {
    Reshape::new(shape).borrow_mut().call(&xs)
}

fn reshape(x: &RcVariable,shape: IxDyn) -> RcVariable {
    let y = reshape_f(&[Some(x.0.clone()), None],shape);
    RcVariable(y.clone())
}





#[derive(Debug, Clone)]
struct Transpose {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Transpose {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Transposeは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Transposeは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        let inputs_0 = inputs[0].as_ref().unwrap().borrow();

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(inputs_0.data.view());

        let ys_data = self.forward(xs_data);

        let output;

        //ys_dataは一変数なので、outputs[1]は必要なし
        output = Variable::new_rc(ys_data);

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();
            self.generation = inputs[0].as_ref().unwrap().borrow().generation;

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(Rc::downgrade(&output));

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: [Option<ArrayViewD<f32>>; 2]) -> ArrayD<f32> {

        let y = xs[0].as_ref().expect("数値がありません").t().to_owned();

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        
        gxs[0] = Some(gys.t().to_owned());
        
        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
        let output;
        output = self
            .output
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Transpose {
    fn new() -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id,
        }))
    }
}



fn transpose_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Transpose::new().borrow_mut().call(&xs)
}

fn transpose(x: &RcVariable) -> RcVariable {
    let y = transpose_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}



//演算子のオーバーロード


impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(add_y.clone())
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(mul_y.clone())
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(sub_y.clone())
    }
}



impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(div_y.clone())
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[Some(self.0.clone()), None]);
        RcVariable(neg_y.clone())
    }
}


//array型からRcVariable型を生成
trait ArrayDToRcVariable {
    fn rv(&self) -> RcVariable;
    
}
//arrayは任意の次元に対応
impl<D:Dimension> ArrayDToRcVariable for ArrayBase<OwnedRepr<f32>, D> {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.view().into_dyn())
    }
}

trait f32ToRcVariable {
    fn rv(&self) -> RcVariable;
}

//rustの数値のデフォルトがf64なので、f32に変換する
//f32からarray型に変換し、rv()でRcVariableを生成
impl f32ToRcVariable for f64 {
    fn rv(&self) -> RcVariable {
        let array = array![*self as f32];
        array.rv()
    }
}



/* 
//rustの数値のデフォルトがf64なので、f32に変換してからRcVariableを生成
impl ArrayDToRcVariable for ArrayBase<OwnedRepr<f32>, Dim<[usize;1]>> {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.view().into_dyn())
    }
}*/
