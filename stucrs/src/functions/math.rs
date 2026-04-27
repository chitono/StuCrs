use core::panic;
use std::cell::RefCell;
//use std::clone;

use std::fmt::Debug;

//use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::*;

use ndarray_stats::QuantileExt;
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::config::{get_grad_status, id_generator};
use crate::core_new::*;
use crate::error::{FrameError, FrameResult};

use crate::functions::matrix::*;
use crate::tensor::lib::TensorOps;

#[derive(Debug, Clone)]
pub struct Square {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for Square {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Square",
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

        let y_data = x.data().pow(2.0).map_err(|e| FrameError::ForwardError {
            function: "Square",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let gx = 2.0.rv() * x.clone() * gy.clone();
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

pub fn square(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = square_f(&[x.clone()]);
    y
}

fn square_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Exp",
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
        let y_data = x.data().exp().map_err(|e| FrameError::ForwardError {
            function: "Exp",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let gx = x.exp()? * gy.clone();

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

pub fn exp(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = exp_f(&[x.clone()]);
    y
}

fn exp_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Sin",
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
        let y_data = x.data().sin().map_err(|e| FrameError::ForwardError {
            function: "Sin",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let gx = cos(x)? * gy.clone();
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

pub fn sin(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = sin_f(&[x.clone()]);
    y
}

fn sin_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Reshape",
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
        let y_data = x.data().cos().map_err(|e| FrameError::ForwardError {
            function: "Reshape",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];

        let gx = -sin(x)? * gy.clone();

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

pub fn cos(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = cos_f(&[x.clone()]);
    y
}

fn cos_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "Tanh",
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
        let y_data = x.data().tanh().map_err(|e| FrameError::ForwardError {
            function: "Tanh",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];

        let gx = gy.clone() / cosh(x)?.pow(2.0)?;

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

pub fn tanh(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = tanh_f(&[x.clone()]);
    y
}

fn tanh_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "Sinh",
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
        let y_data = x.data().sinh().map_err(|e| FrameError::ForwardError {
            function: "Sinh",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let gx = cosh(x)? * gy.clone();
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

pub fn sinh(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = sinh_f(&[x.clone()]);
    y
}

fn sinh_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "Cosh",
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
        let y_data = x.data().cosh().map_err(|e| FrameError::ForwardError {
            function: "Cosh",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let gx = sinh(x)? * gy.clone();
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

pub fn cosh(x: &RcVariable) -> FrameResult<RcVariable> {
    let y = cosh_f(&[x.clone()]);
    y
}

fn cosh_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Log",
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
        let base = self.base;
        let x = &xs[0];
        let y_data;

        //baseがeか他の値かで場合分け(eの場合、baseはNone)
        // TODO:Logの底指定未対応
        if let Some(_base_data) = base {
            return Err(FrameError::Unimplemented {
                context: "Log関数構造体の底指定は今後対応".to_string(),
                source: None,
            });
        } else {
            y_data = x.data().log().map_err(|e| FrameError::ForwardError {
                function: "Log",
                source: e,
            })?;
        }
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
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

pub fn log(x: &RcVariable, base: Option<f32>) -> FrameResult<RcVariable> {
    let y = log_f(&[x.clone()], base);
    y
}

fn log_f(xs: &[RcVariable], base: Option<f32>) -> FrameResult<RcVariable> {
    Log::new(xs, base).borrow_mut().call()
}

/*

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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Max",
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
                        unimplemented!("指定した軸には対応していません。")
                    }
                };

                y_data.into_dyn()
            }

            3 => {
                let x_array = x_array.into_dimensionality::<Ix3>().unwrap();

                // 3次元での軸指定は2のみ対応。今後拡張予定。
                let y_data = match self.axis {
                    Some(0) => {
                        todo!("3次元のmaxの軸0は未実装。")
                    }
                    Some(1) => {
                        todo!("3次元のmaxの軸1は未実装。")
                    }
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
                        unimplemented!("指定した軸は未対応。");
                    }
                };
                y_data.into_dyn()
            }
            _ => {
                unimplemented!("1,2,3次元以外の行列は未対応。");
            }
        };

        y_data.rv()
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
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
                    todo!("2次元のmax関数のbackwardでの軸を指定なしは後で対応")
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
                    unimplemented!("指定した軸は未対応。")
                }
            },
            3 => match self.axis {
                None => {
                    todo!("3次元のmax関数のbackwardでの軸を指定なしは後で対応")
                }
                Some(0) => {
                    todo!("3次元のmax関数のbackwardの軸0はまだ未対応")
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
                    unimplemented!("指定した軸は未対応。")
                }
            },
            _ => {
                unimplemented!("1-3次元以外の次元には対応していません.")
            }
        }

        let broadcasted_gy = broadcast_to(gy, IxDyn(x_shape));

        let gx = mask_array.rv() * broadcasted_gy;

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
pub fn max(x: &RcVariable, axis: Option<u16>) -> FrameResult<RcVariable> {
    let y = max_f(&[x.clone()], axis);
    y
}

fn max_f(xs: &[RcVariable], axis: Option<u16>) -> FrameResult<RcVariable> {
    Max::new(xs, axis).borrow_mut().call()
}

*/

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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(FrameError::InvalidInputCount {
                function: "Clamp",
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
        let x_data = xs[0].data();

        //最大値をはじめに調整
        let mut y_data = x_data
            .clamp_max(self.max)
            .map_err(|e| FrameError::ForwardError {
                function: "Clamp_max in Clamp",
                source: e,
            })?;

        //最小値を調整
        y_data = y_data
            .clamp_min(self.min)
            .map_err(|e| FrameError::ForwardError {
                function: "Clamp_min in Clamp",
                source: e,
            })?;
        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];

        let min_mask = x.data().max_for_clamp_grad()?;
        let max_mask = x.data().min_for_clamp_grad()?;

        let mask = (min_mask * max_mask)?.rv();

        let gx = gy.clone() * mask;

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
pub fn clamp(x: &RcVariable, min: f32, max: f32) -> FrameResult<RcVariable> {
    let y = clamp_f(&[x.clone()], min, max);
    y
}

fn clamp_f(xs: &[RcVariable], min: f32, max: f32) -> FrameResult<RcVariable> {
    Clamp::new(xs, min, max).borrow_mut().call()
}
