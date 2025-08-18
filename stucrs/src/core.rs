use core::panic;
use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::{Array, ArrayD, ArrayViewD, IxDyn};
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::functions::*;

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
pub struct Variable {
    pub data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    pub name: Option<String>,
    pub generation: i32,
    pub id: u32,
}
impl Variable {
    pub fn new_rc(data: ArrayD<f32>) -> Rc<RefCell<Self>> {
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

    pub fn set_creator(&mut self, func: Rc<RefCell<dyn Function>>) {
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
pub struct RcVariable(pub Rc<RefCell<Variable>>);

impl RcVariable {
    pub fn new(data: ArrayViewD<f32>) -> Self {
        RcVariable(Variable::new_rc(data.to_owned()))
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().backward();
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn cleargrad(&mut self) {
        self.0.borrow_mut().cleargrad();
    }

    pub fn pow(&self, c: f64) -> RcVariable {
        let y = pow(&[Some(self.0.clone()), None], c);
        RcVariable(y.clone())
    }

    pub fn exp(&self) -> RcVariable {
        let y = exp(&self);
        y
    }

    pub fn reshape(&self, shape: IxDyn) -> RcVariable {
        let y = reshape(&self, shape);
        y
    }

    pub fn t(&self) -> RcVariable {
        let y = transpose(&self);
        y
    }

    pub fn sum(&self, axis: Option<u16>) -> RcVariable {
        let y = sum(&self, axis);
        y
    }
}

pub trait Function: Debug {
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
struct AddF {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for AddF {
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
        let mut gx0 = gys.clone().to_owned();
        let mut gx1 = gys.clone().to_owned();

        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x1_borrow = self.inputs[1].as_ref().unwrap().borrow();

        let x0_shape = IxDyn(x0_borrow.data.shape());
        let x1_shape = IxDyn(x1_borrow.data.shape());

        if x0_shape != x1_shape {
            gx0 = array_sum_to(&gys, x0_shape);
            gx1 = array_sum_to(&gys, x1_shape);
        }

        let gxs = [Some(gx0.to_owned()), Some(gx1.to_owned())];

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
impl AddF {
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

pub fn add(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    AddF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct MulF {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for MulF {
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

        let mut gx0 = &x1_data * &gys;
        let mut gx1 = &x0_data * &gys;

        let x0_shape = IxDyn(x0_borrow.data.shape());
        let x1_shape = IxDyn(x1_borrow.data.shape());

        if x0_shape != x1_shape {
            gx0 = array_sum_to(&gx0.view(), x0_shape);
            gx1 = array_sum_to(&gx1.view(), x1_shape);
        }

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
impl MulF {
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

pub fn mul(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    MulF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct SubF {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for SubF {
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
        let mut gx0 = gys.clone().to_owned();
        let mut gx1 = -gys.clone().to_owned();

        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x1_borrow = self.inputs[1].as_ref().unwrap().borrow();

        let x0_shape = IxDyn(x0_borrow.data.shape());
        let x1_shape = IxDyn(x1_borrow.data.shape());

        if x0_shape != x1_shape {
            gx0 = array_sum_to(&gys, x0_shape);
            gx1 = array_sum_to(&gys, x1_shape);
        }

        let gxs = [Some(gx0.to_owned()), Some(gx1.to_owned())];

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
impl SubF {
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

pub fn sub(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    SubF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct DivF {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for DivF {
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

        let mut gx0 = &gys / &x1_data;
        let mut gx1 = &gys * (-&x0_data / &x1_data.mapv(|x| x.powf(2.0)));

        let x0_shape = IxDyn(x0_borrow.data.shape());
        let x1_shape = IxDyn(x1_borrow.data.shape());

        if x0_shape != x1_shape {
            gx0 = array_sum_to(&gx0.view(), x0_shape);
            gx1 = array_sum_to(&gx1.view(), x1_shape);
        }

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
impl DivF {
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

pub fn div(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    DivF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct NegF {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for NegF {
    fn call(&mut self, inputs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
        if let None = &inputs[0] {
            panic!("Negは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Negは一変数関数です。input[1]がNoneではある必要があります")
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
impl NegF {
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

pub fn neg(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> Rc<RefCell<Variable>> {
    NegF::new().borrow_mut().call(&xs)
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
        let y = xs[0]
            .as_ref()
            .expect("数値がありません")
            .mapv(|x| x.powf(c));

        y
    }

    fn backward(&self, gys: ArrayViewD<f32>) -> [Option<ArrayD<f32>>; 2] {
        let mut gxs = [None, None];
        let x0_borrow = self.inputs[0].as_ref().unwrap().borrow();
        let x0_data = x0_borrow.data.view();
        let c = self.c;

        gxs[0] = Some(c * x0_data.mapv(|gx| gx.powf(c - 1.0)) * &gys);

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
    fn new(c: f64) -> Rc<RefCell<Self>> {
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

pub fn pow(xs: &[Option<Rc<RefCell<Variable>>>; 2], c: f64) -> Rc<RefCell<Variable>> {
    Pow::new(c).borrow_mut().call(&xs)
}

/*
//rustの数値のデフォルトがf64なので、f32に変換してからRcVariableを生成
impl ArrayDToRcVariable for ArrayBase<OwnedRepr<f32>, Dim<[usize;1]>> {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.view().into_dyn())
    }
}*/
