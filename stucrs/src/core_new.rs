use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;
use std::iter::zip;

use ndarray::{array, ArrayBase, ArrayD, ArrayViewD, Dimension, IxDyn, OwnedRepr};
use std::rc::{Rc, Weak};
use std::vec;

use crate::config::{get_grad_status, id_generator, set_grad_false, set_grad_true};
use crate::functions_new::*;

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
        if current_grad_flag == true {
            set_grad_true();
        } else {
            set_grad_false();
        }
    }

    /*

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
    }*/

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

    /*
    pub fn clear_grad_backward(&mut self) {
        self.0.borrow_mut().clear_grad_backward();
    } */

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
        let y = pow(&[self.clone()], c);
        y
    }
    /*

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
    } */
}

pub trait Function: Debug {
    fn call(&mut self) -> RcVariable;

    //  forward,backwardはVariableの数値のみを計算する
    fn forward(&self, x: &[RcVariable]) -> RcVariable;
    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable>;

    //　関数クラス.inputs, .outputではvariableのbackwardの中でアクセスできないので、関数にして取得
    fn get_inputs(&self) -> &[RcVariable];
    fn get_output(&self) -> RcVariable;
    fn get_generation(&self) -> i32;
    fn get_id(&self) -> usize;
}

#[derive(Debug, Clone)]
struct AddF {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for AddF {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            panic!("Addは二変数関数です。inputsの個数が二つではありません。")
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
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = x0.data() + x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let mut gx0 = gy.clone();
        let mut gx1 = gy.clone();

        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
impl AddF {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn add(xs: &[RcVariable]) -> RcVariable {
    AddF::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
struct MulF {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for MulF {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            panic!("Mulは二変数関数です。inputsの個数が二つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputsのgenerationで一番大きい値をFuncitonのgenerationとする
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
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = x0.data() * x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let mut gx0 = x1.clone() * gy.clone();
        let mut gx1 = x0.clone() * gy.clone();

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
impl MulF {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn mul(xs: &[RcVariable]) -> RcVariable {
    MulF::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
struct SubF {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for SubF {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            panic!("Subは二変数関数です。inputsの個数が二つではありません。")
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
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = x0.data() - x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let mut gx0 = gy.clone();
        let mut gx1 = -gy.clone();

        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
impl SubF {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn sub(xs: &[RcVariable]) -> RcVariable {
    SubF::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
struct DivF {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for DivF {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            panic!("Divは二変数関数です。inputsの個数が二つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputsのgenerationで一番大きい値をFuncitonのgenerationとする
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
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = x0.data() / x1.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let mut gx0 = gy.clone() / x1.clone();
        let mut gx1 = gy.clone() * (-x0.clone() / x1.pow(2.0));

        let x0_shape = IxDyn(x0.data().shape());
        let x1_shape = IxDyn(x1.data().shape());

        if x0_shape != x1_shape {
            gx0 = sum_to(&gx0, x0_shape);
            gx1 = sum_to(&gx1, x1_shape);
        }

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
impl DivF {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn div(xs: &[RcVariable]) -> RcVariable {
    DivF::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
struct NegF {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for NegF {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Negは一変数関数です。inputsの個数が一つではありません。")
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
        let y_data = -x.data();

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let gxs = vec![-gy.clone()];

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
impl NegF {
    fn new(inputs: &[RcVariable]) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn neg(xs: &[RcVariable]) -> RcVariable {
    NegF::new(xs).borrow_mut().call()
}

#[derive(Debug, Clone)]
struct Pow {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    c: f32,
    generation: i32,
    id: usize,
}

impl Function for Pow {
    fn call(&mut self) -> RcVariable {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            panic!("Powは一変数関数です。inputsの個数が一つではありません。")
        }

        let output = self.forward(inputs);

        if get_grad_status() == true {
            //inputsのgenerationで一番大きい値をFuncitonのgenerationとする
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
        let c = self.c;
        let x = &xs[0];
        let y_data = x.data().mapv(|x| x.powf(c));

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> Vec<RcVariable> {
        let x = &self.inputs[0];

        let c = self.c;
        let gx = c.rv() * x.pow(c - 1.0f32) * gy.clone();
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
impl Pow {
    fn new(input: &[RcVariable], c: f32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: input.to_vec(),
            output: None,
            c: c as f32,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn pow(xs: &[RcVariable], c: f32) -> RcVariable {
    Pow::new(xs, c).borrow_mut().call()
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
