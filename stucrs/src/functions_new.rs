use core::panic;
use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;

//use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::*;
use ndarray_rand::rand_distr::Standard;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::config::{get_grad_status, get_test_flag_status, id_generator, TEST_FLAG};
use crate::core_new::*;
use crate::datasets::arr1d_to_one_hot;

//static NEXT_ID: AtomicU32 = AtomicU32::new(1);

/*

static KEEP_GRAD: Mutex<bool> = Mutex::new(false);

fn set_keep_grad_true() {
    let mut flag = KEEP_GRAD.lock().unwrap();
    *flag = true;
}

fn set_keep_grad_false() {
    let mut flag = KEEP_GRAD.lock().unwrap();
    *flag = false;
}

fn get_keep_grad_status() -> bool {
    let flag = KEEP_GRAD.lock().unwrap();
    *flag
} */

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

/*

fn main() {
    let start = Instant::now();

    //set_grad_false();
    set_keep_grad_true();

    let iters = 10000;
    for _i in 0..iters {
        set_grad_true();
        let  x0 = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();
        //let x1 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        //let  shape_array = [1,6];


        // `&[usize; 2]`を`IxDyn`に変換
        //let dyn_shape = IxDyn(&shape_array);
        let mut y = exp(&(2.0.rv()*x0.clone()));
        //println!("y_data = {:?}\n",y.0.borrow().data);

        y.backward(false);


        //println!("x_grad = {:?}\n",x0.0.borrow().grad.as_ref().unwrap().data());

        //let mut gx = x0.grad().clone();
        //println!("x0 = {:?}", x0.clone());


        //gx.0.borrow_mut().data = Array::zeros(gx)
        //x0.cleargrad();
        //println!("x0 = {:?}", x0.clone());




        //gx.as_mut().unwrap().backward(false);


        //println!("{:?}", x0.grad().as_ref().unwrap().data());

        //println!("x2_grad={:?}\n", x2.grad());
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
} */

#[derive(Debug, Clone)]
pub struct Square {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Square {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Squareは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];

        let y_data = x.data().mapv(|x| x.powf(2.0));

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx = 2.0.rv() * x.clone() * gy.clone();
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
impl Square {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

//fn square(input_x:&Rc<RefCell<Variable>>)

pub fn square(x: &RcVariable) -> RcVariable {
    let y = square_f(&[x.clone()]);
    y
}

fn square_f(xs: &[RcVariable]) -> RcVariable {
    Square::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
pub struct Exp {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Exp {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Expは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.exp());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx = x.exp().clone() * gy.clone();

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
impl Exp {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn exp(x: &RcVariable) -> RcVariable {
    let y = exp_f(&[x.clone()]);
    y
}

fn exp_f(xs: &[RcVariable]) -> RcVariable {
    Exp::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
pub struct Sin {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Sin {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Sinは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.sin());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx = cos(x) * gy.clone();
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
impl Sin {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn sin(x: &RcVariable) -> RcVariable {
    let y = sin_f(&[x.clone()]);
    y
}

fn sin_f(xs: &[RcVariable]) -> RcVariable {
    Sin::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
pub struct Cos {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Cos {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Cosは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.cos());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];

        let gx = -sin(x) * gy.clone();

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
impl Cos {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn cos(x: &RcVariable) -> RcVariable {
    let y = cos_f(&[x.clone()]);
    y
}

fn cos_f(xs: &[RcVariable]) -> RcVariable {
    Cos::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
pub struct Tanh {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Tanh {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Tanhは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.tanh());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];

        let gx = gy.clone() / cosh(x).pow(2.0);

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
impl Tanh {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn tanh(x: &RcVariable) -> RcVariable {
    let y = tanh_f(&[x.clone()]);
    y
}

fn tanh_f(xs: &[RcVariable]) -> RcVariable {
    Tanh::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
pub struct Sinh {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Sinh {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Sinhは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.sinh());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx = cosh(x) * gy.clone();
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
impl Sinh {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn sinh(x: &RcVariable) -> RcVariable {
    let y = sinh_f(&[x.clone()]);
    y
}

fn sinh_f(xs: &[RcVariable]) -> RcVariable {
    Sinh::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
pub struct Cosh {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Cosh {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Coshは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.cosh());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let gx = sinh(x) * gy.clone();
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
impl Cosh {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn cosh(x: &RcVariable) -> RcVariable {
    let y = cosh_f(&[x.clone()]);
    y
}

fn cosh_f(xs: &[RcVariable]) -> RcVariable {
    Cosh::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
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

// 行列計算用関数

#[derive(Debug, Clone)]
struct Reshape {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: usize,
}

impl Function for Reshape {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Reshapeは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_shape = self.shape.clone();
        let y_data = x.data().to_shape(y_shape).unwrap().to_owned();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let x_shape = IxDyn(x.data().shape());
        let gx = reshape(gy, x_shape);
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
impl Reshape {
    fn new(inputs: &[RcVariable], shape: IxDyn) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            shape: shape,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn reshape_f(xs: &[RcVariable], shape: IxDyn) -> RcVariable {
    Reshape::new(xs, shape).borrow_mut().call()
}

pub fn reshape(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = reshape_f(&[x.clone()], shape);
    y
}

#[derive(Debug, Clone)]
struct Transpose {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Transpose {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Transposeは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().t().to_owned();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let gxs = vec![gy.t().to_owned()];

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
impl Transpose {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn transpose_f(xs: &[RcVariable]) -> RcVariable {
    Transpose::new(xs).borrow_mut().call()
}

pub fn transpose(x: &RcVariable) -> RcVariable {
    let y = transpose_f(&[x.clone()]);
    y
}

#[derive(Debug, Clone)]
struct PermuteAxes {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    axes: Vec<usize>,
    generation: i32,
    id: usize,
}

impl Function for PermuteAxes {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("PermuteAxesは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let axes = self.axes.clone();

        let y_data = x.data().permuted_axes(axes);

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let axes_len = self.axes.len();
        let new_axes: Vec<usize> = self
            .axes
            .clone()
            .into_iter()
            .map(|axis| axis % axes_len)
            .collect();
        let mut idx: Vec<usize> = (0..axes_len).collect();
        idx.sort_by_key(|&i| new_axes[i]);

        let gx = permute_axes(gy, idx);
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
impl PermuteAxes {
    fn new(inputs: &[RcVariable], axes: Vec<usize>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            axes: axes,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn permute_axes_f(xs: &[RcVariable], axes: Vec<usize>) -> RcVariable {
    PermuteAxes::new(xs, axes).borrow_mut().call()
}

pub fn permute_axes(x: &RcVariable, axes: Vec<usize>) -> RcVariable {
    let y = permute_axes_f(&[x.clone()], axes);
    y
}

#[derive(Debug, Clone)]
struct Sum {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    axis: Option<u16>,
    generation: i32,
    id: usize,
}

impl Function for Sum {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Sumは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let axis = self.axis;

        let y_data;

        if let Some(axis_data) = axis {
            if axis_data != 0 && axis_data != 1 {
                panic!("axisは0か1の値のみ指定できます")
            }

            y_data = x
                .data()
                .sum_axis(Axis(axis_data as usize))
                .insert_axis(Axis(axis_data as usize));
        } else {
            let scalar_y = x.data().sum();
            y_data = array![scalar_y].into_dyn();
        }

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let x_shape = IxDyn(x.data().shape());
        let gx = broadcast_to(gy, x_shape);
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
impl Sum {
    fn new(inputs: &[RcVariable], axis: Option<u16>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            axis: axis,

            generation: 0,
            id: id_generator(),
        }))
    }
}

fn array_sum(x_array: &ArrayViewD<f32>, axis: Option<u16>) -> ArrayD<f32> {
    let y;

    if let Some(axis_data) = axis {
        if axis_data != 0 && axis_data != 1 {
            panic!("axisは0か1の値のみ指定できます")
        }

        y = x_array.to_owned().sum_axis(Axis(axis_data as usize));
    } else {
        let scalar_y = x_array.to_owned().sum();
        y = array![scalar_y].into_dyn();
    }

    y
}

fn sum_f(xs: &[RcVariable], axis: Option<u16>) -> RcVariable {
    Sum::new(xs, axis).borrow_mut().call()
}

pub fn sum(x: &RcVariable, axis: Option<u16>) -> RcVariable {
    let y = sum_f(&[x.clone()], axis);
    y
}

#[derive(Debug, Clone)]
struct BroadcastTo {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: usize,
}

impl Function for BroadcastTo {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("BroadcastToは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];

        let y_shape = self.shape.clone();

        // 実際の形状を `IxDynImpl` からスライスとして抽出

        let y_data = x.data().broadcast(y_shape).unwrap().mapv(|x| x.clone());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let x_shape = IxDyn(x.data().shape());

        let gx = sum_to(gy, x_shape);
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
impl BroadcastTo {
    fn new(inputs: &[RcVariable], shape: IxDyn) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            shape: shape,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn broadcast_to_f(xs: &[RcVariable], shape: IxDyn) -> RcVariable {
    BroadcastTo::new(xs, shape).borrow_mut().call()
}

pub fn broadcast_to(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = broadcast_to_f(&[x.clone()], shape);
    y
}

#[derive(Debug, Clone)]
struct SumTo {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: usize,
}

impl Function for SumTo {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("SumToは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_shape = self.shape.clone();
        let y_data = array_sum_to(&x.data().view(), y_shape);

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];

        let x_shape = IxDyn(x.data().shape());

        let gx = broadcast_to(gy, x_shape);
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
impl SumTo {
    fn new(inputs: &[RcVariable], shape: IxDyn) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            shape: shape,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn array_sum_to(x_array: &ArrayViewD<f32>, shape: IxDyn) -> ArrayD<f32> {
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
        y = y.sum_axis(Axis(axis)).insert_axis(Axis(axis));
    }

    y
}

fn sum_to_f(xs: &[RcVariable], shape: IxDyn) -> RcVariable {
    SumTo::new(xs, shape).borrow_mut().call()
}

pub fn sum_to(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y;
    let x_shape = IxDyn(x.data().shape());
    if x_shape == shape {
        y = x.clone();
    } else {
        y = sum_to_f(&[x.clone()], shape);
    }

    y
}

#[derive(Debug, Clone)]
struct MatMul {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for MatMul {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            panic!("Matmulは二変数関数です。inputsの個数が二つではありません。")
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
        //xs[0]の方をX, xs[1]の方をWとする
        let x = &xs[0];
        let w = &xs[1];

        let x_data = x.data();
        let w_data = w.data();

        //match以降の場合分けを関数にしたい
        let y_data = array_matmul(&x_data.view(), &w_data.view());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let w = &self.inputs[1];

        let gx = matmul(gy, &w.t());
        let gw = matmul(&x.t(), gy);
        let gxs = vec![gx, gw];

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
impl MatMul {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn array_matmul(x_array: &ArrayViewD<f32>, w_array: &ArrayViewD<f32>) -> ArrayD<f32> {
    let y = match (x_array.ndim(), w_array.ndim()) {
        // 1D × 1D → スカラー出力
        (1, 1) => {
            let x = x_array.clone().into_dimensionality::<Ix1>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix1>().unwrap();

            let y = x.dot(&w);
            ArrayD::from_elem(ndarray::IxDyn(&[]), y) // スカラーとして返す
        }

        // 2D × 1D
        (2, 1) => {
            let x = x_array.clone().into_dimensionality::<Ix2>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix1>().unwrap();
            let y = x.dot(&w);
            y.into_dyn()
        }

        // 1D × 2D
        (1, 2) => {
            let x = x_array.clone().into_dimensionality::<Ix1>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix2>().unwrap();
            let y = x.dot(&w);
            y.into_dyn()
        }

        // 2D × 2D
        (2, 2) => {
            let x = x_array.clone().into_dimensionality::<Ix2>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix2>().unwrap();
            let y = x.dot(&w);
            y.into_dyn()
        }

        _ => {
            panic!("3次元以上の行列積は未実装");
        }
    };

    y
}

fn matmul_f(xs: &[RcVariable]) -> RcVariable {
    MatMul::new(xs).borrow_mut().call()
}

pub fn matmul(x: &RcVariable, w: &RcVariable) -> RcVariable {
    let y = matmul_f(&[x.clone(), w.clone()]);
    y
}

/// 軸を指定できるよう拡張する予定
#[derive(Debug, Clone)]
struct TensorDot {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for TensorDot {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            panic!("TensorDotは二変数関数です。inputsの個数が二つではありません。")
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
        //xs[0]の方をX, xs[1]の方をWとする
        let x = &xs[0];
        let w = &xs[1];

        let x_data = x.data();
        let w_data = w.data();

        //match以降の場合分けを関数にしたい
        let y_data = array_tensordot(x_data.view(), w_data.view());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        let w = &self.inputs[1];

        let (gx, gw) = tensordot_backward(gy, x, w);
        let gxs = vec![gx, gw];

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
impl TensorDot {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn array_tensordot(x_array: ArrayViewD<f32>, w_array: ArrayViewD<f32>) -> ArrayD<f32> {
    let y = match (x_array.ndim(), w_array.ndim()) {
        // 3D × 2D
        //(N,k,l) ×　(l,m) -> (N,k,m)
        (3, 2) => {
            let x = x_array.clone().into_dimensionality::<Ix3>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix2>().unwrap();
            if x.shape()[2] != w.shape()[0] {
                panic!("array_tensorの(3,2)での計算でxとwの次元が適合しません。")
            }
            let n = x.shape()[0];
            let k = x.shape()[1];
            let m = w.shape()[1];

            let mut y = Array3::<f32>::zeros((n, k, m));
            // xからバッチのように2次元の行列を取り出し、2次元の行列積
            for b in 0..n {
                let x_matrix = x.slice(s![b, .., ..]);
                let result = x_matrix.dot(&w);
                y.slice_mut(s![b, .., ..]).assign(&result);
            }
            y.into_dyn()
        }

        // 2D × 3D
        //(k,l) ×　(N,l,m) -> (N,k,m)
        (2, 3) => {
            let x = x_array.clone().into_dimensionality::<Ix2>().unwrap();
            let w = w_array.clone().into_dimensionality::<Ix3>().unwrap();

            if x.shape()[1] != w.shape()[1] {
                panic!("array_tensorの(2,3)での計算でxとwの次元が適合しません。")
            }
            let n = w.shape()[0];
            let k = x.shape()[0];
            let m = w.shape()[2];

            let mut y = Array3::<f32>::zeros((n, k, m));
            // xからバッチのように2次元の行列を取り出し、2次元の行列積
            for b in 0..n {
                let w_matrix = w.slice(s![b, .., ..]);
                let result = x.dot(&w_matrix);
                y.slice_mut(s![b, .., ..]).assign(&result);
            }
            y.into_dyn()
        }

        // 3D × 3D
        (3, 3) => {
            panic!("3次元と3次元の行列積は未実装。今後実装予定。");
        }

        _ => {
            panic!("4次元以上または2次元以下の行列積は未実装。");
        }
    };

    y
}

fn tensordot_backward(gy: &RcVariable, x: &RcVariable, w: &RcVariable) -> (RcVariable, RcVariable) {
    let (gx, gw) = match (x.data().ndim(), w.data().ndim()) {
        // 3D × 2D
        //(N,k,l) ×　(l,m) -> (N,k,m)の場合
        (3, 2) => {
            let n = x.data().shape()[0];
            let k = x.data().shape()[1];
            let l = x.data().shape()[2];
            let m = w.data().shape()[1];

            let gx = tensordot(gy, &w.t());
            let gw = matmul(
                &x.reshape(IxDyn(&[n * k, l])).t(),
                &gy.reshape(IxDyn(&[n * k, m])),
            );

            (gx, gw)
        }

        // 2D × 3D
        //(k,l) ×　(N,l,m) -> (N,k,m)
        (2, 3) => {
            let k = x.data().shape()[0];
            let l = x.data().shape()[1];
            let n = w.data().shape()[0];
            let m = w.data().shape()[2];

            //(n,k,m) -> (k,n,m) -> (k,n*m)
            let gy1 = permute_axes(&gy, vec![1, 0, 2]).reshape(IxDyn(&[k, n * m]));
            //(n,l,m) -> (l,n,m) -> (l,n*m) -> (n*m,l)
            let w1 = permute_axes(&w, vec![1, 0, 2])
                .reshape(IxDyn(&[l, n * m]))
                .t();
            let gx = matmul(&gy1, &w1); //(k,n*m) @ (n*m,l) -> (k,l)

            let gw = tensordot(&x.t(), gy); //(l,k) @' (n,k,m) -> (n,l,m)

            (gx, gw)
        }

        // 3D × 3D
        (3, 3) => {
            panic!("3次元と3次元の行列積は未実装。今後実装予定。");
        }

        _ => {
            panic!("4次元以上または2次元以下の行列積は未実装。");
        }
    };

    (gx, gw)
}

fn tensordot_f(xs: &[RcVariable]) -> RcVariable {
    TensorDot::new(xs).borrow_mut().call()
}

/// 2次元と3次元の行列積の関数
/// (2D×3D), (3D×2D), (3D×3D)に対応
pub fn tensordot(x: &RcVariable, w: &RcVariable) -> RcVariable {
    let y = tensordot_f(&[x.clone(), w.clone()]);
    y
}

#[derive(Debug, Clone)]
struct MeanSquaredError {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for MeanSquaredError {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("MeanSquaredErrorは二変数関数です。inputsの個数が二つではありません。")
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
        //xs[0]の方をX, xs[1]の方をWとする
        let x0 = &xs[0];
        let x1 = &xs[1];

        let diff = &x0.data() - &x1.data();
        let len = diff.len() as f32;

        let error_data = array_sum(&diff.mapv(|x| x.powf(2.0)).view(), None) / len;

        error_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let diff = x0.clone() - x1.clone();
        let diff_shape = IxDyn(diff.data().shape());
        let gy = broadcast_to(&gy, diff_shape);

        let gx0 = gy.clone() * diff.clone() * (2.0.rv() / (diff.len() as f32).rv());
        let gx1 = -gx0.clone();
        let gxs = vec![gx0, gx1];

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
impl MeanSquaredError {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn mean_squared_error_f(xs: &[RcVariable]) -> RcVariable {
    MeanSquaredError::new(xs).borrow_mut().call()
}

pub fn mean_squared_error(x0: &RcVariable, x1: &RcVariable) -> RcVariable {
    let y = mean_squared_error_f(&[x0.clone(), x1.clone()]);
    y
}

pub fn linear_simple(x: &RcVariable, w: &RcVariable, b: &Option<RcVariable>) -> RcVariable {
    let t = matmul(&x, &w);

    let y;

    if let Some(b_rc) = b {
        y = t + b_rc.clone();
    } else {
        y = t;
    }

    y
}

pub fn sigmoid_simple(x: &RcVariable) -> RcVariable {
    let mainasu_x = -x.clone();
    let y = 1.0f32.rv() / (1.0f32.rv() + exp(&mainasu_x));
    y
}

#[derive(Debug, Clone)]
pub struct Relu {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Relu {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Reluは一変数関数です。inputsの個数が一つではありません。")
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
        let x = &xs[0];
        let y_data = x.data().mapv(|x| if x > 0.0 { x } else { 0.0 });

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];
        //xが0以上なら微分の値は1で、0以下なら0になる。
        let gx = x.data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }).rv() * gy.clone();
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
impl Relu {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn relu(x: &RcVariable) -> RcVariable {
    let y = relu_f(&[x.clone()]);
    y
}

fn relu_f(xs: &[RcVariable]) -> RcVariable {
    Relu::new(xs).borrow_mut().call()
}

pub fn softmax_simple(x: &RcVariable) -> RcVariable {
    let exp_y = exp(&x);

    let sum_y = sum(&exp_y, Some(1));

    let y = exp_y.clone() / sum_y.clone();
    y
}

// ここで渡すtはone-hotベクトル状態の教師データ
pub fn softmax_cross_entropy_simple(x: &RcVariable, t: &RcVariable) -> RcVariable {
    if x.data().shape() != t.data().shape() {
        panic!("交差エントロピー誤差でのxとtの形状が異なります。tがone-hotベクトルでない可能性があります。")
    }

    let n = x.data().shape()[0] as f32;

    let p = softmax_simple(&x);

    let clamped_p = clamp(&p, 1.0e-15, 1.0);

    let log_p = log(&clamped_p, None);

    let tlog_p = log_p * t.clone();

    let y = (-sum(&tlog_p, None)) / n.rv();
    y
}

/// 最大値を返す関数。
/// 現在は3次元までの行列の最大値に対応。また軸の指定にも対応.
/// 軸指定は一部の軸のみ、今後拡張予定。
/// 返す行列はinputの行列と同じ次元数。
#[derive(Debug, Clone)]
pub struct Max {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    axis: Option<u16>,
    generation: i32,
    id: usize,
}

impl Function for Max {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Maxは一変数関数です。inputsの個数が一つではありません。")
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
        let x_array = xs[0].data();

        let y_data = match x_array.ndim() {
            1 => {
                let x_data = x_array
                    .max()
                    .expect("max関数で1次元の行列の最大値の取得に失敗しました。")
                    .clone();

                array![x_data].into_dyn()
            }

            2 => {
                let x_array = x_array.into_dimensionality::<Ix2>().unwrap();

                let y_data = match self.axis {
                    Some(0) => {
                        let y_data: Array1<f32> = x_array
                            .axis_iter(Axis(1))
                            .map(|col| col.max().unwrap().clone())
                            .collect();
                        let y_data = y_data.insert_axis(Axis(0));
                        y_data.into_dyn()
                    }
                    Some(1) => {
                        let y_data: Array1<f32> = x_array
                            .axis_iter(Axis(0))
                            .map(|row| row.max().unwrap().clone())
                            .collect();
                        let y_data = y_data.insert_axis(Axis(1));
                        y_data.into_dyn()
                    }

                    None => {
                        let y_data = x_array.max().unwrap().clone();
                        array![y_data].into_dyn()
                    }

                    _ => {
                        panic!("指定した軸には対応していません。")
                    }
                };

                y_data.into_dyn()
            }

            3 => {
                let x_array = x_array.into_dimensionality::<Ix3>().unwrap();

                // 3次元での軸指定は2のみ対応。今後拡張予定。
                let y_data = match self.axis {
                    Some(2) => {
                        let n = x_array.shape()[0];
                        let h = x_array.shape()[1];

                        let mut y_data = Array2::<f32>::zeros((n, h));
                        for b in 0..n {
                            let x_matrix = x_array.slice(s![b, .., ..]);

                            let result: Array1<f32> = x_matrix
                                .axis_iter(Axis(0))
                                .map(|row| row.max().unwrap().clone())
                                .collect();

                            y_data.slice_mut(s![b, ..]).assign(&result);
                        }
                        let y_data = y_data.insert_axis(Axis(2));
                        y_data
                    }
                    _ => {
                        panic!("指定した軸は未対応。");
                    }
                };
                y_data.into_dyn()
            }
            _ => {
                panic!("1,2,3次元以外の行列は未対応。")
            }
        };

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x_data = &self.inputs[0].data();
        let x_shape = x_data.shape();
        let mut mask_array = ArrayD::<f32>::zeros(x_shape);

        let x_argmax_array = argmax_array(x_data.view(), self.axis);

        match x_shape.len() {
            1 => {
                mask_array[x_argmax_array[0]] = 1.0;
            }
            2 => match self.axis {
                None => {
                    panic!("2次元のmax関数のbackwardでの軸を指定なしは後で対応")
                }
                Some(0) => {
                    for (i, index) in x_argmax_array.iter().enumerate() {
                        mask_array[[index.clone(), i]] = 1.0;
                    }
                }
                Some(1) => {
                    for (i, index) in x_argmax_array.iter().enumerate() {
                        mask_array[[i, index.clone()]] = 1.0;
                    }
                }
                _ => {
                    panic!("指定した軸は未対応。")
                }
            },
            3 => match self.axis {
                None => {
                    panic!("3次元のmax関数のbackwardでの軸を指定なしは後で対応")
                }
                Some(0) => {
                    panic!("3次元のmax関数のbackwardの軸0はまだ未対応")
                }
                Some(1) => {
                    let n = x_shape[0];
                    let w = x_shape[2];

                    for b in 0..n {
                        for width in 0..w {
                            let max_h_index = x_argmax_array[[b, width]];
                            mask_array[[b, max_h_index, width]] = 1.0;
                        }
                    }
                }
                Some(2) => {
                    let n = x_shape[0];
                    let h = x_shape[1];

                    for b in 0..n {
                        for height in 0..h {
                            let max_w_index = x_argmax_array[[b, height]];
                            mask_array[[b, height, max_w_index]] = 1.0;
                        }
                    }
                }
                _ => {
                    panic!("指定した軸は未対応。")
                }
            },
            _ => {
                panic!("1-3次元以外の次元には対応していません.")
            }
        }

        println!("gy_shape = {:?}", gy.data().shape());
        println!("x_shape = {:?}", x_shape);
        println!("mask_shape = {:?}", mask_array.shape());

        let broadcasted_gy = broadcast_to(gy, IxDyn(x_shape));

        println!(
            "broadecasted_gy_shape = {:?}",
            broadcasted_gy.data().shape()
        );

        let gx = mask_array.rv() * broadcasted_gy;
        println!("gx_shape = {:?}", gx.data().shape());

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
impl Max {
    fn new(inputs: &[RcVariable], axis: Option<u16>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            axis: axis,
            generation: 0,
            id: id_generator(),
        }))
    }
}

/// 最大値を返す関数。
/// 現在は3次元までの行列の最大値に対応。また軸の指定にも対応.
/// 軸指定は一部の軸のみ、今後拡張予定。
/// 返す行列はinputの行列と同じ次元数。
pub fn max(x: &RcVariable, axis: Option<u16>) -> RcVariable {
    let y = max_f(&[x.clone()], axis);
    y
}

fn max_f(xs: &[RcVariable], axis: Option<u16>) -> RcVariable {
    Max::new(xs, axis).borrow_mut().call()
}

/// Clamp関数は入力値xを設定された値、min,maxに収める関数。
/// xがminより小さいならminを、maxより大きいならmaxを、min以上max以下ならxの値をそのまま返す。
#[derive(Debug, Clone)]
pub struct Clamp {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    min: f32,
    max: f32,
    generation: i32,
    id: usize,
}

impl Function for Clamp {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Clampは一変数関数です。inputsの個数が一つではありません。")
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
        let x_data = xs[0].data();

        //最大値をはじめに調整
        let mut y_data = x_data.mapv(|x| if x > self.max { self.max } else { x });

        //最小値を調整
        y_data = y_data.mapv(|x| if x < self.min { self.min } else { x });
        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];

        let min_mask = x.data().mapv(|x| if x >= self.min { 1.0f32 } else { 0.0 });
        let max_mask = x.data().mapv(|x| if x <= self.max { 1.0f32 } else { 0.0 });

        let mask = (min_mask * max_mask).rv();

        let gx = gy.clone() * mask;

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
impl Clamp {
    fn new(inputs: &[RcVariable], min: f32, max: f32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            min: min,
            max: max,
            generation: 0,
            id: id_generator(),
        }))
    }
}

/// Clamp関数は入力値xを設定された値、min,maxに収める関数。
/// xがminより小さいならminを、maxより大きいならmaxを、min以上max以下ならxの値をそのまま返す。
pub fn clamp(x: &RcVariable, min: f32, max: f32) -> RcVariable {
    let y = clamp_f(&[x.clone()], min, max);
    y
}

fn clamp_f(xs: &[RcVariable], min: f32, max: f32) -> RcVariable {
    Clamp::new(xs, min, max).borrow_mut().call()
}

pub fn accuracy(y: ArrayView2<f32>, t: ArrayView2<f32>) -> f32 {
    if y.shape() != t.shape() {
        panic!("交差エントロピー誤差でのxとtの形状が異なります。tがone-hotベクトルでない可能性があります。")
    }
    let data_size = y.shape()[0] as f32;
    let num_class = t.shape()[1];
    let argmax_vec: Vec<u32> = y
        .outer_iter()
        .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| row.argmax().unwrap() as u32)
        .collect();
    let max_index = Array::from_vec(argmax_vec);
    let one_hot_y = arr1d_to_one_hot(max_index.view(), num_class);

    assert_eq!(one_hot_y.shape(), t.shape());

    let acc_matrix = &one_hot_y * &t;

    let accuracy = acc_matrix.sum() / data_size;

    accuracy
}

pub fn dropout(x: &RcVariable, ratio: f32) -> RcVariable {
    if get_test_flag_status() == false {
        let random_array: Array<f32, IxDyn> = Array::random(x.data().shape(), Standard);
        let mask = random_array
            .mapv(|x| if x > ratio { 1.0f32 } else { 0.0f32 })
            .rv();
        let scale = array![1.0f32 - ratio].rv();
        let y = x.clone() * mask / scale;
        y
    } else {
        x.clone()
    }
}

/// 行列の最大値のインデックスを返す。
/// 軸指定可能。
/// 1次元から3次元まで対応。
/// まだ一部の軸しか対応していない。
pub fn argmax_array(x_array: ArrayViewD<f32>, axis: Option<u16>) -> ArrayD<usize> {
    let y_array: ArrayD<usize> = match x_array.ndim() {
        1 => {
            let x_array = x_array.into_dimensionality::<Ix1>().unwrap();
            let index = x_array.argmax().unwrap();
            array![index].into_dyn()
        }
        2 => {
            let y_data = match axis {
                None => {
                    panic!("2次元のargmax関数の軸を指定なしは後で対応")
                }
                Some(0) => {
                    let x_array = x_array.into_dimensionality::<Ix2>().unwrap();
                    let max_array: Array1<usize> = x_array
                        .axis_iter(Axis(1))
                        .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| {
                            row.argmax().unwrap()
                        })
                        .collect();
                    max_array.into_dyn()
                }
                Some(1) => {
                    let x_array = x_array.into_dimensionality::<Ix2>().unwrap();
                    let max_array: Array1<usize> = x_array
                        .axis_iter(Axis(0))
                        .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| {
                            row.argmax().unwrap()
                        })
                        .collect();
                    max_array.into_dyn()
                }
                _ => {
                    panic!("指定した軸には対応していません。")
                }
            };
            y_data
        }
        3 => {
            let x_array = x_array.into_dimensionality::<Ix3>().unwrap();
            let y_array: Array2<usize> = match axis {
                None => {
                    panic!("3次元のargmax関数の軸を指定なしは後で対応")
                }
                Some(0) => {
                    panic!("3次元の軸0はまだ未対応")
                }
                Some(1) => {
                    let n = x_array.shape()[0];
                    let w = x_array.shape()[2];

                    let mut y_array = Array2::<usize>::zeros((n, w));
                    for b in 0..n {
                        let matrix = x_array.slice(s![b, .., ..]);
                        let max_array: Array1<usize> = matrix
                            .axis_iter(Axis(1))
                            .map(|col: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| {
                                col.argmax().unwrap()
                            })
                            .collect();
                        y_array.slice_mut(s![b, ..]).assign(&max_array);
                    }
                    y_array
                }
                Some(2) => {
                    let n = x_array.shape()[0];
                    let h = x_array.shape()[1];

                    let mut y_array = Array2::<usize>::zeros((n, h));
                    for b in 0..n {
                        let matrix = x_array.slice(s![b, .., ..]);
                        let max_array: Array1<usize> = matrix
                            .axis_iter(Axis(0))
                            .map(|row: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>| {
                                row.argmax().unwrap()
                            })
                            .collect();
                        y_array.slice_mut(s![b, ..]).assign(&max_array);
                    }
                    y_array
                }
                _ => {
                    panic!("その他の軸は対応していません")
                }
            };
            y_array.into_dyn()
        }
        _ => {
            panic!("1-3次元以外の次元には対応していません")
        }
    };
    y_array
}
