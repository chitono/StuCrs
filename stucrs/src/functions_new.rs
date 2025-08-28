use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::{
    array, s, Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMut1, ArrayViewMut2, Axis,
    Dimension, Ix1, Ix2, IxDyn, Slice, Zip,
};
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::core_new::*;

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

static GRAD_CONFIG: Mutex<bool> = Mutex::new(true);
static KEEP_GRAD: Mutex<bool> = Mutex::new(false);

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

fn main() {}

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
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Square {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Squareは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Squareは一変数関数です。input[1]がNoneである必要があります")
        }
        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();

        let y_data = x.data().mapv(|x| x.powf(2.0));

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let mut gxs = [None, None];
        let x = self.inputs[0].as_ref().unwrap();

        gxs[0] = Some(2.0.rv() * x.clone() * gy.clone());

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

pub fn square(x: &RcVariable) -> RcVariable {
    let y = square_f(&[Some(x.clone()), None]);
    y
}

fn square_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Square::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Exp {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Exp {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Expは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Expは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.exp());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let mut gxs = [None, None];
        let x = self.inputs[0].as_ref().unwrap();

        gxs[0] = Some(x.exp().clone() * gy.clone());

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    let y = exp_f(&[Some(x.clone()), None]);
    y
}

fn exp_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Exp::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Sin {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Sin {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Sinは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Sinは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.sin());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let mut gxs = [None, None];
        let x = self.inputs[0].as_ref().unwrap();

        gxs[0] = Some(cos(x) * gy.clone());

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    let y = sin_f(&[Some(x.clone()), None]);
    y
}

fn sin_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Sin::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Cos {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Cos {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Cosは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Cosは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.cos());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();

        let gx = -sin(x) * gy.clone();

        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    let y = cos_f(&[Some(x.clone()), None]);
    y
}

fn cos_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Cos::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Tanh {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Tanh {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Tanhは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Tanhは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.tanh());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();

        let gx = gy.clone() / cosh(x).pow(2.0);

        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    let y = tanh_f(&[Some(x.clone()), None]);
    y
}

fn tanh_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Tanh::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Sinh {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Sinh {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Sinhは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Sinhは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.sinh());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let gx = cosh(x) * gy.clone();
        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Sinh {
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

pub fn sinh(x: &RcVariable) -> RcVariable {
    let y = sinh_f(&[Some(x.clone()), None]);
    y
}

fn sinh_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Sinh::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
pub struct Cosh {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Cosh {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Coshは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Coshは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.cosh());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let gx = sinh(x) * gy.clone();
        let gxs = [Some(gx), None];
        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Cosh {
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

pub fn cosh(x: &RcVariable) -> RcVariable {
    let y = cosh_f(&[Some(x.clone()), None]);
    y
}

fn cosh_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Cosh::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Log {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    base: Option<f32>,
    generation: i32,
    id: u32,
}

impl Function for Log {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Logは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Logは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let base = self.base;
        let x = xs[0].as_ref().unwrap();
        let y_data;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            y_data = x.data().mapv(|x| x.log(base_data));
        } else {
            y_data = x.data().mapv(|x| x.ln());
        }
        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let gx;

        let base = self.base;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        if let Some(base_data) = base {
            gx = 1.0.rv() / (x.clone() * base_data.ln().rv()) * gy.clone();
        } else {
            gx = (1.0.rv() / x.clone()) * gy.clone();
        }
        let gxs = [Some(gx), None];
        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    let y = log_f(&[Some(x.clone()), None], base);
    y
}

fn log_f(xs: &[Option<RcVariable>; 2], base: Option<f32>) -> RcVariable {
    Log::new(base).borrow_mut().call(&xs)
}

/*









*/

// 行列計算用関数

#[derive(Debug, Clone)]
struct Reshape {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: u32,
}

impl Function for Reshape {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Reshapeは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Reshapeは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_shape = self.shape.clone();
        let y_data = x.data().to_shape(y_shape).unwrap().to_owned();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let x_shape = IxDyn(x.data().shape());
        let gx = reshape(gy, x_shape);
        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

fn reshape_f(xs: &[Option<RcVariable>; 2], shape: IxDyn) -> RcVariable {
    Reshape::new(shape).borrow_mut().call(&xs)
}

pub fn reshape(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = reshape_f(&[Some(x.clone()), None], shape);
    y
}

#[derive(Debug, Clone)]
struct Transpose {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for Transpose {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Transposeは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Transposeは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().t().to_owned();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let gxs = [Some(gy.t().to_owned()), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

fn transpose_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    Transpose::new().borrow_mut().call(&xs)
}

pub fn transpose(x: &RcVariable) -> RcVariable {
    let y = transpose_f(&[Some(x.clone()), None]);
    y
}

#[derive(Debug, Clone)]
struct Sum {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    axis: Option<u16>,
    generation: i32,
    id: u32,
}

impl Function for Sum {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Sumは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Sumは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
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

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let x_shape = IxDyn(x.data().shape());
        let gx = broadcast_to(gy, x_shape);
        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

fn sum_f(xs: &[Option<RcVariable>; 2], axis: Option<u16>) -> RcVariable {
    Sum::new(axis).borrow_mut().call(&xs)
}

pub fn sum(x: &RcVariable, axis: Option<u16>) -> RcVariable {
    let y = sum_f(&[Some(x.clone()), None], axis);
    y
}

#[derive(Debug, Clone)]
struct BroadcastTo {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: u32,
}

impl Function for BroadcastTo {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("BroadcastToは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("BroadcastToは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();

        let y_shape = self.shape.clone();

        // 実際の形状を `IxDynImpl` からスライスとして抽出

        let y_data = x.data().broadcast(y_shape).unwrap().mapv(|x| x.clone());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let x_shape = IxDyn(x.data().shape());

        let gx = sum_to(gy, x_shape);
        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

fn broadcast_to_f(xs: &[Option<RcVariable>; 2], shape: IxDyn) -> RcVariable {
    BroadcastTo::new(shape).borrow_mut().call(&xs)
}

pub fn broadcast_to(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y = broadcast_to_f(&[Some(x.clone()), None], shape);
    y
}

#[derive(Debug, Clone)]
struct SumTo {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    shape: IxDyn,
    generation: i32,
    id: u32,
}

impl Function for SumTo {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("SumToは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("SumToは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }
        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x = xs[0].as_ref().unwrap();
        let y_shape = self.shape.clone();
        let y_data = array_sum_to(&x.data().view(), y_shape);

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();

        let x_shape = IxDyn(x.data().shape());

        let gx = broadcast_to(gy, x_shape);
        let gxs = [Some(gx), None];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

fn sum_to_f(xs: &[Option<RcVariable>; 2], shape: IxDyn) -> RcVariable {
    SumTo::new(shape).borrow_mut().call(&xs)
}

pub fn sum_to(x: &RcVariable, shape: IxDyn) -> RcVariable {
    let y;
    let x_shape = IxDyn(x.data().shape());
    if x_shape == shape {
        y = x.clone();
    } else {
        y = sum_to_f(&[Some(x.clone()), None], shape);
    }

    y
}

#[derive(Debug, Clone)]
struct MatMul {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for MatMul {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("MatMulは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("MatMulは二変数関数です。input[1]がNoneです")
        }

        let xs_data = [
            Some(inputs[0].as_ref().unwrap().clone()),
            Some(inputs[1].as_ref().unwrap().clone()),
        ];

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            let input_0_generation = inputs[0].as_ref().unwrap().generation();
            let input_1_generation = inputs[1].as_ref().unwrap().generation();

            //inputのgenerationで大きい値の方をFuncitonのgenerationとする
            self.generation = match input_0_generation >= input_1_generation {
                true => input_0_generation,
                false => input_1_generation,
            };

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        //xs[0]の方をX, xs[1]の方をWとする
        let x = xs[0].as_ref().unwrap();
        let w = xs[1].as_ref().unwrap();

        let x_data = x.data();
        let w_data = w.data();

        //match以降の場合分けを関数にしたい
        let y_data = array_matmul(&x_data.view(), &w_data.view());

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();
        let w = self.inputs[1].as_ref().unwrap();

        let mut gxs = [None, None];

        let gx = matmul(gy, &w.t());
        let gw = matmul(&x.t(), gy);

        gxs[0] = Some(gx);
        gxs[1] = Some(gw);
        //println!("matmul_gx = {:?}\n", gxs.clone());
        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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

fn matmul_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    MatMul::new().borrow_mut().call(&xs)
}

pub fn matmul(x: &RcVariable, w: &RcVariable) -> RcVariable {
    let y = matmul_f(&[Some(x.clone()), Some(w.clone())]);
    y
}

#[derive(Debug, Clone)]
struct MeanSquaredError {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: u32,
}

impl Function for MeanSquaredError {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("MeanSquaredErrorは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("MeanSquaredErrorは二変数関数です。input[1]がNoneです")
        }

        let xs_data = [
            Some(inputs[0].as_ref().unwrap().clone()),
            Some(inputs[1].as_ref().unwrap().clone()),
        ];

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            let input_0_generation = inputs[0].as_ref().unwrap().generation();
            let input_1_generation = inputs[1].as_ref().unwrap().generation();

            //inputのgenerationで大きい値の方をFuncitonのgenerationとする
            self.generation = match input_0_generation >= input_1_generation {
                true => input_0_generation,
                false => input_1_generation,
            };

            //  outputsを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        //xs[0]の方をX, xs[1]の方をWとする
        let x0 = xs[0].as_ref().unwrap();
        let x1 = xs[1].as_ref().unwrap();

        let diff = &x0.data() - &x1.data();
        let len = diff.len() as f32;

        let error_data = array_sum(&diff.mapv(|x| x.powf(2.0)).view(), None) / len;

        error_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x0 = self.inputs[0].as_ref().unwrap();
        let x1 = self.inputs[1].as_ref().unwrap();

        let diff = x0.clone() - x1.clone();
        let diff_shape = IxDyn(diff.data().shape());
        let gy = broadcast_to(&gy, diff_shape);

        let gx0 = gy.clone() * diff.clone() * (2.0.rv() / (diff.len() as f32).rv());
        let gx1 = -gx0.clone();
        let gxs = [Some(gx0), Some(gx1)];

        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl MeanSquaredError {
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

fn mean_squared_error_f(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    MeanSquaredError::new().borrow_mut().call(&xs)
}

pub fn mean_squared_error(x0: &RcVariable, x1: &RcVariable) -> RcVariable {
    let y = mean_squared_error_f(&[Some(x0.clone()), Some(x1.clone())]);
    y
}

/*
fn add_at_1d(mut x: ArrayViewMut1<f32>, indices: &[usize], y: ArrayViewMut1<f32>) {
    Zip::from(indices).and(b).for_each(|&idx, b_data| {
        if let Some(a_data) = a.get_mut(idx) {
            *a_data += *b_data;
        }
    });
} */

/*
fn add_at_2d(mut a: ArrayViewMut2<f32>, indices: (&[usize], &[usize]), b: ArrayViewMut2<f32>) {
    let (row_indices, col_indices) = indices;
    Zip::from(row_indices)
        .and(col_indices)
        .and(b.to_shape(b.len()))
        .for_each(|&row, &col, b_data| {
            if let Some(a_data) = a.get_mut((row, col)) {
                *a_data += *b_data;
            }
        });
} */

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

pub fn softmax_simple(x: &RcVariable) -> RcVariable {
    let exp_y = exp(&x);

    let sum_y = sum(&exp_y, Some(1));

    let y = exp_y / sum_y;
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

    let y = -sum(&tlog_p, None) / n.rv();
    y
}

//clamp

#[derive(Debug, Clone)]
pub struct Clamp {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    min: f32,
    max: f32,
    generation: i32,
    id: u32,
}

impl Function for Clamp {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Clampは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Clampは一変数関数です。input[1]がNoneではある必要があります")
        }

        let xs_data = [Some(inputs[0].as_ref().unwrap().clone()), None];

        // inputのvariableからdataを取り出す

        let ys_data = self.forward(&xs_data);

        let output = ys_data.clone();

        //ここから下の処理はbackwardするときだけ必要。

        if get_grad_status() == true {
            //　inputsを覚える
            self.inputs = inputs.clone();

            self.generation = inputs[0].as_ref().unwrap().generation();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputに自分をcreatorとして覚えさせる　不変長　配列2
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        output
    }

    fn forward(&self, xs: &[Option<RcVariable>; 2]) -> RcVariable {
        let x_data = xs[0].as_ref().unwrap().data();

        //最大値をはじめに調整
        let mut y_data = x_data.mapv(|x| if x > self.max { self.max } else { x });

        //最小値を調整
        y_data = y_data.mapv(|x| if x < self.min { self.min } else { x });
        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x = self.inputs[0].as_ref().unwrap();

        let min_mask = x.data().mapv(|x| if x >= self.min { 1.0f32 } else { 0.0 });
        let max_mask = x.data().mapv(|x| if x <= self.max { 1.0f32 } else { 0.0 });

        let mask = (min_mask * max_mask).rv();

        let gx = gy.clone() * mask;

        let gxs = [Some(gx), None];
        gxs
    }

    fn get_inputs(&self) -> [Option<RcVariable>; 2] {
        self.inputs.clone()
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
    fn get_id(&self) -> u32 {
        self.id
    }
}
impl Clamp {
    fn new(min: f32, max: f32) -> Rc<RefCell<Self>> {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            min: min,
            max: max,
            generation: 0,
            id: id,
        }))
    }
}

pub fn clamp(x: &RcVariable, min: f32, max: f32) -> RcVariable {
    let y = clamp_f(&[Some(x.clone()), None], min, max);
    y
}

fn clamp_f(xs: &[Option<RcVariable>; 2], min: f32, max: f32) -> RcVariable {
    Clamp::new(min, max).borrow_mut().call(&xs)
}
