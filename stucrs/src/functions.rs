use core::panic;
use std::cell::RefCell;
//use std::clone;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use crate::core::*;
use ndarray::{array, ArrayD, ArrayViewD, Axis, Dimension, Ix1, Ix2, IxDyn};

use std::collections::HashSet;
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

static GRAD_CONFIG: Mutex<bool> = Mutex::new(true);

pub fn set_grad_true() {
    let mut flag = GRAD_CONFIG.lock().unwrap();
    *flag = true;
}

pub fn set_grad_false() {
    let mut flag = GRAD_CONFIG.lock().unwrap();
    *flag = false;
}

pub fn get_grad_status() -> bool {
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

#[derive(Debug, Clone)]
pub struct Square {
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

pub fn square(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    //xを一変数から配列に変換

    let y = Square::new().borrow_mut().call(xs);

    let output_list = [Some(y), None];
    output_list
}

#[derive(Debug, Clone)]
pub struct Exp {
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

pub fn exp(x: &RcVariable) -> RcVariable {
    let y = exp_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn exp_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Exp::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Sin {
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

pub fn sin(x: &RcVariable) -> RcVariable {
    let y = sin_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn sin_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Sin::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Cos {
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

pub fn cos(x: &RcVariable) -> RcVariable {
    let y = cos_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn cos_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Cos::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Tanh {
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

pub fn tanh(x: &RcVariable) -> RcVariable {
    let y = tanh_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

fn tanh_f(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    Tanh::new().borrow_mut().call(&xs)
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
        let y;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            y = xs[0]
                .as_ref()
                .expect("数値が存在する")
                .mapv(|x| x.log(base_data));
        } else {
            y = xs[0].as_ref().expect("数値が存在するはず").mapv(|x| x.ln());
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
            gxs[0] = Some(1.0 / (&x0_data * base_data.ln()) * &gys);
        } else {
            gxs[0] = Some((1.0 / &x0_data) * &gys);
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
    fn new(base: Option<f32>) -> Rc<RefCell<Self>> {
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

pub fn log(x: &RcVariable, base: Option<f32>) -> RcVariable {
    let y = log_f(&[Some(x.0.clone()), None], base);
    RcVariable(y.clone())
}

fn log_f(xs: &[Option<Rc<RefCell<Variable>>>; 2], base: Option<f32>) -> Rc<RefCell<Variable>> {
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
        let y = xs[0]
            .as_ref()
            .expect("数値がありません")
            .to_owned()
            .into_shape(y_shape)
            .unwrap();

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

fn reshape_f(xs: &[Option<Rc<RefCell<Variable>>>; 2], shape: IxDyn) -> Rc<RefCell<Variable>> {
    Reshape::new(shape).borrow_mut().call(&xs)
}

pub fn reshape(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = reshape_f(&[Some(x.0.clone()), None], shape);
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

pub fn transpose(x: &RcVariable) -> RcVariable {
    let y = transpose_f(&[Some(x.0.clone()), None]);
    RcVariable(y.clone())
}

#[derive(Debug, Clone)]
struct Sum {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    axis: Option<u16>,
    generation: i32,
    id: u32,
}

impl Function for Sum {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Sumは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Sumは一変数関数です。input[1]がNoneではある必要があります")
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
        let axis = self.axis;

        let y;

        if let Some(axis_data) = axis {
            if axis_data != 0 && axis_data != 1 {
                panic!("axisは0か1の値のみ指定できます")
            }

            y = xs[0]
                .as_ref()
                .expect("数値がありません")
                .to_owned()
                .sum_axis(Axis(axis_data as usize));
        } else {
            let scalar_y = xs[0].as_ref().expect("数値がありません").to_owned().sum();
            y = array![scalar_y].into_dyn();
        }

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x_shape = x_borrow.data.shape();

        gxs[0] = Some(array_sum_to(&gys, IxDyn(x_shape)));

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
impl Sum {
    fn new(axis: Option<u16>) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            axis: axis,

            generation: 0,
            id: id,
        }))
    }
}

fn sum_f(xs: &[Option<Rc<RefCell<Variable>>>; 2], axis: Option<u16>) -> Rc<RefCell<Variable>> {
    Sum::new(axis).borrow_mut().call(&xs)
}

pub fn sum(x: &RcVariable, axis: Option<u16>) -> RcVariable {
    let y = sum_f(&[Some(x.0.clone()), None], axis);
    RcVariable(y.clone())
}

#[derive(Debug, Clone)]
struct BroadcastTo {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: u32,
}

impl Function for BroadcastTo {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("BroadcastToは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("BroadcastToは一変数関数です。input[1]がNoneではある必要があります")
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
        let y = xs[0]
            .as_ref()
            .expect("数値がありません")
            .broadcast(y_shape)
            .unwrap()
            .mapv(|x| x.clone());

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x_shape = IxDyn(x_borrow.data.shape());

        gxs[0] = Some(array_sum_to(&gys, x_shape));

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
impl BroadcastTo {
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

fn broadcast_to_f(xs: &[Option<Rc<RefCell<Variable>>>; 2], shape: IxDyn) -> Rc<RefCell<Variable>> {
    BroadcastTo::new(shape).borrow_mut().call(&xs)
}

pub fn broadcast_to(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = broadcast_to_f(&[Some(x.0.clone()), None], shape);
    RcVariable(y.clone())
}

#[derive(Debug, Clone)]
struct SumTo {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: u32,
}

impl Function for SumTo {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("SumToは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("SumToは一変数関数です。input[1]がNoneではある必要があります")
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
        let x_data = xs[0].as_ref().expect("数値がありません");
        let y_shape = self.shape.clone();
        let y = array_sum_to(x_data, y_shape);

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x_shape = x_borrow.data.shape();

        gxs[0] = Some(gys.broadcast(x_shape).unwrap().mapv(|x| x.clone()));

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
impl SumTo {
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

pub fn array_sum_to(x_array: &ArrayViewD<f32>, shape: IxDyn) -> ArrayD<f32> {
    let x_shape = x_array.shape();

    let mut axes_to_sum = HashSet::new();

    // 合計する軸を特定する
    for i in 0..x_shape.len() {
        if i >= shape.ndim() || x_shape[i] != shape[i] {
            axes_to_sum.insert(i);
        }
    }

    let mut y = x_array.to_owned();

    // HashSetの要素をVecに収集し、ソートして逆順にイテレーションする
    let mut sorted_axes: Vec<_> = axes_to_sum.into_iter().collect();
    sorted_axes.sort_unstable();

    // 特定した軸を合計する
    for &axis in sorted_axes.iter().rev() {
        y = y.sum_axis(Axis(axis));
    }

    y
}

fn sum_to_f(xs: &[Option<Rc<RefCell<Variable>>>; 2], shape: IxDyn) -> Rc<RefCell<Variable>> {
    SumTo::new(shape).borrow_mut().call(&xs)
}

pub fn sum_to(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = sum_to_f(&[Some(x.0.clone()), None], shape);
    RcVariable(y.clone())
}

#[derive(Debug, Clone)]
struct MatMul {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for MatMul {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("MatMulは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("MatMulは二変数関数です。input[1]がNoneです")
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
        //xs[0]の方をX, xs[1]の方をWとする
        let x = xs[0].as_ref().unwrap();
        let w = xs[1].as_ref().unwrap();

        //match以降の場合分けを関数にしたい
        let y = match (x.ndim(), w.ndim()) {
            // 1D × 1D → スカラー出力
            (1, 1) => {
                let x = x.clone().into_dimensionality::<Ix1>().unwrap();
                let w = w.clone().into_dimensionality::<Ix1>().unwrap();
                let y = x.dot(&w);
                ArrayD::from_elem(ndarray::IxDyn(&[]), y) // スカラーとして返す
            }

            // 2D × 1D → 1Dベクトル
            (2, 1) => {
                let x = x.clone().into_dimensionality::<Ix2>().unwrap();
                let w = w.clone().into_dimensionality::<Ix1>().unwrap();
                let y = x.dot(&w);
                y.into_dyn()
            }

            // 2D × 2D → 行列積
            (2, 2) => {
                let x = x.clone().into_dimensionality::<Ix2>().unwrap();
                let w = w.clone().into_dimensionality::<Ix2>().unwrap();
                let y = x.dot(&w);
                y.into_dyn()
            }

            _ => {
                panic!("3次元以上の行列積は未実装");
            }
        };

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];

        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x1_borrow = self.inputs[1].as_ref().unwrap().borrow();

        let x0_data = x0_borrow.data.view();
        let x1_data = x1_borrow.data.view();

        let mut gx0 = &x1_data * &gys;
        let mut gx1 = &x0_data * &gys;

        gxs[0] = Some(gx0);
        gxs[1] = Some(gx1);

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
impl MatMul {
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

pub fn matmul(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    MatMul::new().borrow_mut().call(&xs)
}

/*
//rustの数値のデフォルトがf64なので、f32に変換してからRcVariableを生成
impl ArrayDToRcVariable for ArrayBase<OwnedRepr<f32>, Dim<[usize;1]>> {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.view().into_dyn())
    }
}*/
