use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;

use std::iter::zip;

use std::rc::{Rc, Weak};
use std::vec;

use crate::error::{FrameError, FrameResult};

use crate::tensor::lib::TensorOps;
use crate::tensor::shape::Shape;
use crate::tensor::tensor::Tensor;

use crate::config::{get_grad_status, id_generator, set_grad_false, set_grad_true};
use crate::functions::math::exp;
use crate::functions::matrix::*;

#[derive(Debug, Clone)]
pub struct Variable {
    pub data: Tensor,
    grad: Option<RcVariable>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    pub name: Option<String>,
    pub generation: i32,
    pub id: usize,
}
impl Variable {
    pub fn new_rc(data: Tensor) -> Rc<RefCell<Self>> {
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

    fn backward(&self, double_grad: bool) -> FrameResult<()> {
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
                y_grad = Tensor::ones(self.data.shape().dims())?.rv();

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし

                y_grad = y.0.borrow().grad.as_ref().unwrap().clone();
            }

            let xs_grad = f_borrowed.backward(&y_grad)?;

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
        Ok(())
    }

    fn cleargrad(&mut self) {
        self.grad = None;
    }
}

#[derive(Debug, Clone)]
pub struct RcVariable(pub Rc<RefCell<Variable>>);

impl RcVariable {
    pub fn new(data: &Tensor) -> Self {
        RcVariable(Variable::new_rc(data.clone()))
    }

    pub fn backward(&mut self, double_grad: bool) {
        self.0.borrow_mut().backward(double_grad);
    }

    /*
    pub fn clear_grad_backward(&mut self) {
        self.0.borrow_mut().clear_grad_backward();
    } */

    pub fn data(&self) -> Tensor {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<RcVariable> {
        self.0.borrow().grad.clone()
    }

    pub fn cleargrad(&mut self) {
        self.0.borrow_mut().cleargrad();
    }

    pub fn len(&self) -> u32 {
        self.data().shape().dims()[0] as u32
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

    pub fn pow(&self, c: f32) -> FrameResult<RcVariable> {
        let y = pow(&[self.clone()], c);
        y
    }

    pub fn exp(&self) -> FrameResult<RcVariable> {
        let y = exp(&self);
        y
    }

    pub fn reshape(&self, shape: &Shape) -> FrameResult<RcVariable> {
        let y = reshape(self, shape);
        y
    }

    pub fn t(&self) -> FrameResult<RcVariable> {
        let y = transpose(self);
        y
    }

    pub fn permute_axes(&self, axes: Vec<usize>) -> FrameResult<RcVariable> {
        let y = permute_axes(self, axes);
        y
    }

    pub fn sum(&self, axis: Option<usize>) -> FrameResult<RcVariable> {
        let y = sum(self, axis);
        y
    }
}

pub trait Function: Debug {
    fn call(&mut self) -> FrameResult<RcVariable>;

    //  forward,backwardはVariableの数値のみを計算する
    fn forward(&self, x: &[RcVariable]) -> FrameResult<RcVariable>;
    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>>;

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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(FrameError::InvalidInputCount {
                function: "Add",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output = self.forward(inputs)?;

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        Ok(output)
    }

    fn forward(&self, xs: &[RcVariable]) -> FrameResult<RcVariable> {
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = (x0.data() + x1.data()).map_err(|e| FrameError::ForwardError {
            function: "Add",
            source: e,
        })?;
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let mut gx0 = gy.clone();
        let mut gx1 = gy.clone();

        let x0_data = &self.inputs[0].data();
        let x1_data = &self.inputs[1].data();

        let x0_shape = x0_data.shape();
        let x1_shape = x1_data.shape();

        if x0_shape != x1_shape {
            gx0 = if let Ok(gx0) = sum_to(&gx0, x0_shape) {
                gx0
            } else {
                return Err(FrameError::BackwardError { function: "Add" });
            };

            gx1 = if let Ok(gx1) = sum_to(&gx1, x1_shape) {
                gx1
            } else {
                return Err(FrameError::BackwardError { function: "Add" });
            };
        }

        let gxs = vec![gx0, gx1];

        Ok(gxs)
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

pub fn add(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(FrameError::InvalidInputCount {
                function: "Mul",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output = self.forward(inputs)?;

        if get_grad_status() == true {
            //inputsのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        Ok(output)
    }

    fn forward(&self, xs: &[RcVariable]) -> FrameResult<RcVariable> {
        let x0 = &xs[0];
        let x1 = &xs[1];

        let y_data = (x0.data() * x1.data()).map_err(|e| FrameError::ForwardError {
            function: "Mul",
            source: e,
        })?;
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let mut gx0 = x1.clone() * gy.clone();
        let mut gx1 = x0.clone() * gy.clone();

        let x0_data = &self.inputs[0].data();
        let x1_data = &self.inputs[1].data();

        let x0_shape = x0_data.shape();
        let x1_shape = x1_data.shape();

        if x0_shape != x1_shape {
            gx0 = if let Ok(gx0) = sum_to(&gx0, x0_shape) {
                gx0
            } else {
                return Err(FrameError::BackwardError { function: "Mul" });
            };

            gx1 = if let Ok(gx1) = sum_to(&gx1, x1_shape) {
                gx1
            } else {
                return Err(FrameError::BackwardError { function: "Mul" });
            };
        }

        let gxs = vec![gx0, gx1];
        Ok(gxs)
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

pub fn mul(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(FrameError::InvalidInputCount {
                function: "Sub",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output = self.forward(inputs)?;

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        Ok(output)
    }

    fn forward(&self, xs: &[RcVariable]) -> FrameResult<RcVariable> {
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = (x0.data() - x1.data()).map_err(|e| FrameError::ForwardError {
            function: "Sub",
            source: e,
        })?;
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let mut gx0 = gy.clone();
        let mut gx1 = -gy.clone();

        let x0_data = &self.inputs[0].data();
        let x1_data = &self.inputs[1].data();

        let x0_shape = x0_data.shape();
        let x1_shape = x1_data.shape();

        if x0_shape != x1_shape {
            gx0 = if let Ok(gx0) = sum_to(&gx0, x0_shape) {
                gx0
            } else {
                return Err(FrameError::BackwardError { function: "Sub" });
            };

            gx1 = if let Ok(gx1) = sum_to(&gx1, x1_shape) {
                gx1
            } else {
                return Err(FrameError::BackwardError { function: "Sub" });
            };
        }

        let gxs = vec![gx0, gx1];

        Ok(gxs)
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

pub fn sub(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(FrameError::InvalidInputCount {
                function: "Div",
                expected: 2,
                got: inputs.len(),
            });
        }

        let output = self.forward(inputs)?;

        if get_grad_status() == true {
            //inputsのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        Ok(output)
    }

    fn forward(&self, xs: &[RcVariable]) -> FrameResult<RcVariable> {
        let x0 = &xs[0];
        let x1 = &xs[1];
        let y_data = (x0.data() / x1.data()).map_err(|e| FrameError::ForwardError {
            function: "Div",
            source: e,
        })?;
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let mut gx0 = gy.clone() / x1.clone();
        let mut gx1 = gy.clone() * (-x0.clone() / (x1.pow(2.0))?);

        let x0_data = &self.inputs[0].data();
        let x1_data = &self.inputs[1].data();

        let x0_shape = x0_data.shape();
        let x1_shape = x1_data.shape();

        if x0_shape != x1_shape {
            gx0 = if let Ok(gx0) = sum_to(&gx0, x0_shape) {
                gx0
            } else {
                return Err(FrameError::BackwardError { function: "Div" });
            };

            gx1 = if let Ok(gx1) = sum_to(&gx1, x1_shape) {
                gx1
            } else {
                return Err(FrameError::BackwardError { function: "Div" });
            };
        }

        let gxs = vec![gx0, gx1];

        Ok(gxs)
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

pub fn div(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Neg",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output = self.forward(inputs)?;

        if get_grad_status() == true {
            //inputのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        Ok(output)
    }

    fn forward(&self, xs: &[RcVariable]) -> FrameResult<RcVariable> {
        let x = &xs[0];
        let y_data = (-x.data()).map_err(|e| FrameError::ForwardError {
            function: "Neg",
            source: e,
        })?;
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let gxs = vec![-gy.clone()];

        Ok(gxs)
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

pub fn neg(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Pow",
                expected: 1,
                got: inputs.len(),
            });
        }

        let output = self.forward(inputs)?;

        if get_grad_status() == true {
            //inputsのgenerationで一番大きい値をFuncitonのgenerationとする
            self.generation = inputs.iter().map(|input| input.generation()).max().unwrap();

            //  outputを弱参照(downgrade)で覚える
            self.output = Some(output.downgrade());

            let self_f: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

            //outputsに自分をcreatorとして覚えさせる
            output.0.borrow_mut().set_creator(self_f.clone());
        }

        Ok(output)
    }

    fn forward(&self, xs: &[RcVariable]) -> FrameResult<RcVariable> {
        let c = self.c;
        let x = &xs[0];
        let y_data = x.data().pow(c).map_err(|e| FrameError::ForwardError {
            function: "Pow",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];

        let c = self.c;
        let gx = c.rv() * x.pow(c - 1.0f32)? * gy.clone();
        let gxs = vec![gx];

        Ok(gxs)
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

pub fn pow(xs: &[RcVariable], c: f32) -> FrameResult<RcVariable> {
    Pow::new(xs, c).borrow_mut().call()
}

// TODO: Tensor,RcVariable生成系、Result対応予定

//Tensor型からRcVariable型を生成
pub trait TensorToRcVariable {
    fn rv(&self) -> RcVariable;
}

impl TensorToRcVariable for Tensor {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self)
    }
}

pub trait F32ToRcVariable {
    fn rv(&self) -> RcVariable;
}

//f32からTensor型に変換し、rv()でRcVariableを生成
impl F32ToRcVariable for f32 {
    fn rv(&self) -> RcVariable {
        let tensor = self.ts();
        tensor.rv()
    }
}

pub trait F32ToTensor {
    fn ts(&self) -> Tensor;
}

//f32からTensor型に変換
impl F32ToTensor for f32 {
    fn ts(&self) -> Tensor {
        let tensor = if let Ok(tensor) = Tensor::from_vec(vec![*self], vec![1]) {
            tensor
        } else {
            panic!("f32からTensorを生成できませんでした。");
        };
        tensor
    }
}
