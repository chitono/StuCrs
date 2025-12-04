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

/// Clamp関数は入力値xを設定された値、min,maxに収める関数です。
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
