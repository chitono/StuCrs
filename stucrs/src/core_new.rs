use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;

use ndarray::{array, ArrayBase, ArrayD, ArrayViewD, Dimension, IxDyn, OwnedRepr};
use std::rc::{Rc, Weak};
use std::vec;

use crate::config::{get_grad_status, id_generator, set_grad_false, set_grad_true};
use crate::functions_new::*;

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
pub struct Variable {
    pub data: ArrayD<f32>,
    grad: Option<RcVariable>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    pub name: Option<String>,
    pub generation: i32,
    pub id: usize,
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

    pub fn set_creator(&mut self, func: Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::clone(&func));
        self.generation = func.borrow().get_generation() + 1;
    }

    fn backward(&self, double_grad: bool) {
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
        let current_grad_flag = get_grad_status();

        while let Some(f_rc) = funcs.pop() {
            //println!("f = {:?}\n", get_struct_name(&f_rc.borrow()));
            let f_borrowed = f_rc.borrow();
            if double_grad == true {
                set_grad_true();
            } else {
                set_grad_false();
            }

            let xs = f_borrowed.get_inputs();

            let y = f_borrowed.get_output();

            let y_grad: Option<RcVariable>;

            if last_variable {
                y_grad = Some(ArrayD::<f32>::ones(self.data.shape()).rv());

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし

                y_grad = y.0.borrow().grad.clone();
            }

            let xs_grad = f_borrowed.backward(&y_grad.as_ref().unwrap());

            // gradを置き換えまたは足していくので、Noneか判別
            let xs_0 = xs[0].as_ref().unwrap();

            if let Some(current_grad_0_data) = xs_0.grad() {
                xs_0.0.borrow_mut().grad =
                    Some(current_grad_0_data + xs_grad[0].as_ref().unwrap().clone());
            } else {
                xs_0.0.borrow_mut().grad = Some(xs_grad[0].as_ref().unwrap().clone())
            }

            //xs[0]にcreatorがあるか確認、あったらfuncに追加
            if let Some(func_creator) = &xs_0.0.borrow().creator {
                add_func(&mut funcs, &mut seen_set, func_creator.clone());
                //funcs.push(Rc::clone(&func_creator));
            }

            //xs[1]はfが一変数関数の時、NoneなのでNoneか判別
            if let Some(xs_1) = &xs[1] {
                if let Some(current_grad_1_data) = xs_1.grad() {
                    xs_1.0.borrow_mut().grad =
                        Some(current_grad_1_data + xs_grad[1].as_ref().unwrap().clone());
                } else {
                    xs_1.0.borrow_mut().grad = Some(xs_grad[1].as_ref().unwrap().clone())
                }

                //xs[1]にcreatorがあるか確認、あったらfuncに追加
                if let Some(func_creator) = &xs_1.0.borrow().creator {
                    add_func(&mut funcs, &mut seen_set, func_creator.clone());
                    //funcs.push(Rc::clone(&func_creator));
                }
            }
        }
        if current_grad_flag == true {
            set_grad_true();
        } else {
            set_grad_false();
        }
    }

    fn clear_grad_backward(&mut self) {
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

            let xs = f_borrowed.get_inputs();

            let mut y = f_borrowed.get_output();

            if last_variable {
                self.cleargrad();

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし

                y.cleargrad();
            }

            // gradを置き換えまたは足していくので、Noneか判別
            let xs_0 = xs[0].as_ref().unwrap();

            //xs[0]にcreatorがあるか確認、あったらfuncに追加
            if let Some(func_creator) = &xs_0.0.borrow().creator {
                add_func(&mut funcs, &mut seen_set, func_creator.clone());
                //funcs.push(Rc::clone(&func_creator));
            }

            //xs[1]はfが一変数関数の時、NoneなのでNoneか判別
            if let Some(xs_1) = &xs[1] {
                //xs[1]にcreatorがあるか確認、あったらfuncに追加
                if let Some(func_creator) = &xs_1.0.borrow().creator {
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

    pub fn backward(&mut self, double_grad: bool) {
        self.0.borrow_mut().backward(double_grad);
    }

    pub fn clear_grad_backward(&mut self) {
        self.0.borrow_mut().clear_grad_backward();
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<RcVariable> {
        self.0.borrow().grad.clone()
    }

    pub fn cleargrad(&mut self) {
        self.0.borrow_mut().cleargrad();
    }

    pub fn len(&self) -> u32 {
        self.data().shape()[0] as u32
    }

    pub fn id(&self) -> usize {
        self.0.borrow().id
    }

    pub fn generation(&self) -> i32 {
        self.0.borrow().generation
    }

    pub fn downgrade(&self) -> Weak<RefCell<Variable>> {
        Rc::downgrade(&self.0)
    }

    pub fn pow(&self, c: f32) -> RcVariable {
        let y = pow(&[Some(self.clone()), None], c);
        y
    }

    pub fn exp(&self) -> RcVariable {
        let y = exp(&self);
        y
    }

    pub fn reshape(&self, shape: IxDyn) -> RcVariable {
        let y = reshape(self, shape);
        y
    }

    pub fn t(&self) -> RcVariable {
        let y = transpose(self);
        y
    }

    pub fn sum(&self, axis: Option<u16>) -> RcVariable {
        let y = sum(self, axis);
        y
    }
}

pub trait Function: Debug {
    fn call(&mut self, input: &[Option<RcVariable>; 2]) -> RcVariable;

    //  forward,backwardはVariableの数値のみを計算する
    fn forward(&self, x: &[Option<RcVariable>; 2]) -> RcVariable;
    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2];

    //　関数クラス.inputs, .outputではvariableのbackwardの中でアクセスできないので、関数にして取得
    fn get_inputs(&self) -> [Option<RcVariable>; 2];
    fn get_output(&self) -> RcVariable;
    fn get_generation(&self) -> i32;
    fn get_id(&self) -> usize;
}

#[derive(Debug, Clone)]
struct AddF {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for AddF {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Addは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Addは二変数関数です。input[1]がNoneです")
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
        let x0 = xs[0].as_ref().unwrap();
        let x1 = xs[1].as_ref().unwrap();
        let y_data = x0.data() + x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let mut gx0 = gy.clone();
        let mut gx1 = gy.clone();

        let x0 = self.inputs[0].as_ref().unwrap();
        let x1 = self.inputs[1].as_ref().unwrap();

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
    fn get_id(&self) -> usize {
        self.id
    }
}
impl AddF {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn add(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    AddF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct MulF {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for MulF {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Mulは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Mulは二変数関数です。input[1]がNoneです")
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
        let x0 = xs[0].as_ref().unwrap();
        let x1 = xs[1].as_ref().unwrap();
        let y_data = x0.data() * x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x0 = self.inputs[0].as_ref().unwrap();
        let x1 = self.inputs[1].as_ref().unwrap();

        let mut gx0 = x1.clone() * gy.clone();
        let mut gx1 = x0.clone() * gy.clone();

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
    fn get_id(&self) -> usize {
        self.id
    }
}
impl MulF {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn mul(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    MulF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct SubF {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for SubF {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Subは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Subは二変数関数です。input[1]がNoneです")
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
        let x0 = xs[0].as_ref().unwrap();
        let x1 = xs[1].as_ref().unwrap();
        let y_data = x0.data() - x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let mut gx0 = gy.clone();
        let mut gx1 = -gy.clone();

        let x0 = self.inputs[0].as_ref().unwrap();
        let x1 = self.inputs[1].as_ref().unwrap();

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
    fn get_id(&self) -> usize {
        self.id
    }
}
impl SubF {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn sub(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    SubF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct DivF {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for DivF {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Divは二変数関数です。input[0]がNoneです")
        }
        if let None = &inputs[1] {
            panic!("Divは二変数関数です。input[1]がNoneです")
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
        let x0 = xs[0].as_ref().unwrap();
        let x1 = xs[1].as_ref().unwrap();
        let y_data = x0.data() / x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let x0 = self.inputs[0].as_ref().unwrap();
        let x1 = self.inputs[1].as_ref().unwrap();

        let mut gx0 = gy.clone() / x1.clone();
        let mut gx1 = gy.clone() * (-x0.clone() / x1.clone().pow(2.0));

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
    fn get_id(&self) -> usize {
        self.id
    }
}
impl DivF {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn div(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    DivF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct NegF {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for NegF {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Negは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Negは一変数関数です。input[1]がNoneではある必要があります")
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
        let y_data = -x.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let gxs = [Some(-gy.clone()), None];

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
    fn get_id(&self) -> usize {
        self.id
    }
}
impl NegF {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn neg(xs: &[Option<RcVariable>; 2]) -> RcVariable {
    NegF::new().borrow_mut().call(&xs)
}

#[derive(Debug, Clone)]
struct Pow {
    inputs: [Option<RcVariable>; 2],
    output: Option<Weak<RefCell<Variable>>>,
    c: f32,
    generation: i32,
    id: usize,
}

impl Function for Pow {
    fn call(&mut self, inputs: &[Option<RcVariable>; 2]) -> RcVariable {
        if let None = &inputs[0] {
            panic!("Powは一変数関数です。input[0]がNoneです")
        }
        if let Some(_variable) = &inputs[1] {
            panic!("Powは一変数関数です。input[1]がNoneではある必要があります")
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
        let c = self.c;
        let x = xs[0].as_ref().unwrap();
        let y_data = x.data().mapv(|x| x.powf(c));

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> [Option<RcVariable>; 2] {
        let mut gxs = [None, None];
        let x = self.inputs[0].as_ref().unwrap();

        let c = self.c;

        gxs[0] = Some(c.rv() * x.pow(c - 1.0f32) * gy.clone());

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
    fn get_id(&self) -> usize {
        self.id
    }
}
impl Pow {
    fn new(c: f32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            output: None,
            c: c as f32,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn pow(xs: &[Option<RcVariable>; 2], c: f32) -> RcVariable {
    Pow::new(c).borrow_mut().call(&xs)
}

//演算子のオーバーロード

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

//rustの数値のデフォルトがf64なので、f32に変換する
//f32からarray型に変換し、rv()でRcVariableを生成
impl F32ToRcVariable for f32 {
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
