use core::panic;
use std::cell::RefCell;
//use std::clone;

use std::fmt::Debug;

//use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;
use ndarray::*;

use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::config::{get_grad_status, id_generator};
use crate::core::*;
use crate::error::{FrameError, FrameResult};
use crate::tensor::lib::TensorOps;
use crate::tensor::shape::Shape;

#[derive(Debug, Clone)]
struct Reshape {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    shape: Shape,
    generation: i32,
    id: usize,
}

impl Function for Reshape {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
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
        let y_data =
            x.data()
                .reshape(self.shape.dims.clone())
                .map_err(|e| FrameError::ForwardError {
                    function: "Reshape",
                    source: e,
                })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let gx = reshape(gy, &x.data().shape());

        if let Ok(gx) = gx {
            return Ok(vec![gx]);
        } else {
            return Err(FrameError::BackwardError {
                function: "Reshape",
            });
        }
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
    fn new(inputs: &[RcVariable], shape: &Shape) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            shape: shape.clone(),
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn reshape_f(xs: &[RcVariable], shape: &Shape) -> FrameResult<RcVariable> {
    Reshape::new(xs, shape).borrow_mut().call()
}

pub fn reshape(x: &RcVariable, shape: &Shape) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "Transpose",
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
        let y_data = x.data().transpose().map_err(|e| FrameError::ForwardError {
            function: "Transpose",
            source: e,
        })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        if let Ok(gx) = gy.t() {
            return Ok(vec![gx]);
        } else {
            return Err(FrameError::BackwardError {
                function: "Transpose",
            });
        }
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

fn transpose_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
    Transpose::new(xs).borrow_mut().call()
}

pub fn transpose(x: &RcVariable) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "PermuteAxes",
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
        let axes = self.axes.clone();

        let y_data = x
            .data()
            .permute(&axes)
            .map_err(|e| FrameError::ForwardError {
                function: "PermuteAxes",
                source: e,
            })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let axes_len = self.axes.len();
        let new_axes: Vec<usize> = self
            .axes
            .clone()
            .into_iter()
            .map(|axis| axis % axes_len)
            .collect();
        let mut idx: Vec<usize> = (0..axes_len).collect();
        idx.sort_by_key(|&i| new_axes[i]);

        if let Ok(gx) = permute_axes(gy, idx) {
            return Ok(vec![gx]);
        } else {
            return Err(FrameError::BackwardError {
                function: "PermuteAxes",
            });
        }
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

fn permute_axes_f(xs: &[RcVariable], axes: Vec<usize>) -> FrameResult<RcVariable> {
    PermuteAxes::new(xs, axes).borrow_mut().call()
}

pub fn permute_axes(x: &RcVariable, axes: Vec<usize>) -> FrameResult<RcVariable> {
    let y = permute_axes_f(&[x.clone()], axes);
    y
}

#[derive(Debug, Clone)]
struct Sum {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    axis: Option<usize>,
    generation: i32,
    id: usize,
}

impl Function for Sum {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "Sum",
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
        let axis = self.axis;

        let y_data = x
            .data()
            .sum(axis, true)
            .map_err(|e| FrameError::ForwardError {
                function: "Sum",
                source: e,
            })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];

        let gx = broadcast_to(gy, x.data().shape());
        if let Ok(gx) = gx {
            return Ok(vec![gx]);
        } else {
            return Err(FrameError::BackwardError { function: "Sum" });
        }
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
    fn new(inputs: &[RcVariable], axis: Option<usize>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            axis: axis,
            generation: 0,
            id: id_generator(),
        }))
    }
}

pub fn array_sum(x_array: &ArrayViewD<f32>, axis: Option<u16>) -> ArrayD<f32> {
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

fn sum_f(xs: &[RcVariable], axis: Option<usize>) -> FrameResult<RcVariable> {
    Sum::new(xs, axis).borrow_mut().call()
}

pub fn sum(x: &RcVariable, axis: Option<usize>) -> FrameResult<RcVariable> {
    let y = sum_f(&[x.clone()], axis);
    y
}

#[derive(Debug, Clone)]
struct BroadcastTo {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    shape: Shape,
    generation: i32,
    id: usize,
}

impl Function for BroadcastTo {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "BroadcastTo",
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

        let y_shape = self.shape.clone();

        let y_data = x
            .data()
            .broadcast_to(y_shape)
            .map_err(|e| FrameError::ForwardError {
                function: "BroadcastTo",
                source: e,
            })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x_data = self.inputs[0].data();
        let x_shape = x_data.shape();

        let gx = sum_to(gy, x_shape);

        if let Ok(gx) = gx {
            return Ok(vec![gx]);
        } else {
            return Err(FrameError::BackwardError {
                function: "BroadcastTo",
            });
        }
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
    fn new(inputs: &[RcVariable], shape: &Shape) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            shape: shape.clone(),
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn broadcast_to_f(xs: &[RcVariable], shape: &Shape) -> FrameResult<RcVariable> {
    BroadcastTo::new(xs, shape).borrow_mut().call()
}

pub fn broadcast_to(x: &RcVariable, shape: &Shape) -> FrameResult<RcVariable> {
    let y = broadcast_to_f(&[x.clone()], shape);
    y
}

#[derive(Debug, Clone)]
struct SumTo {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    shape: Shape,
    generation: i32,
    id: usize,
}

impl Function for SumTo {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 1 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "SumTo",
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
        let y_shape = self.shape.clone();
        let y_data = x
            .data()
            .sum_to(&y_shape)
            .map_err(|e| FrameError::ForwardError {
                function: "SumTo",
                source: e,
            })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];

        let gx = broadcast_to(gy, x.data().shape());
        if let Ok(gx) = gx {
            return Ok(vec![gx]);
        } else {
            return Err(FrameError::BackwardError { function: "SumTo" });
        }
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
    fn new(inputs: &[RcVariable], shape: &Shape) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: inputs.to_vec(),
            output: None,
            shape: shape.clone(),
            generation: 0,
            id: id_generator(),
        }))
    }
}

fn sum_to_f(xs: &[RcVariable], shape: &Shape) -> FrameResult<RcVariable> {
    SumTo::new(xs, shape).borrow_mut().call()
}

pub fn sum_to(x: &RcVariable, shape: &Shape) -> FrameResult<RcVariable> {
    if x.data().shape() == shape {
        Ok(x.clone())
    } else {
        sum_to_f(&[x.clone()], shape)
    }
}

#[derive(Debug, Clone)]
struct MatMul {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for MatMul {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "MatMul",
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
        //xs[0]の方をX, xs[1]の方をWとする
        let x = &xs[0];
        let w = &xs[1];

        let x_data = x.data();
        let w_data = w.data();

        let y_data = x_data
            .matmul(&w_data)
            .map_err(|e| FrameError::ForwardError {
                function: "MatMul",
                source: e,
            })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let w = &self.inputs[1];

        let gx = matmul(gy, &w.t()?);
        let gw = matmul(&x.t()?, gy);

        if let (Ok(gx), Ok(gw)) = (gx, gw) {
            Ok(vec![gx, gw])
        } else {
            return Err(FrameError::BackwardError { function: "MatMul" });
        }
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

fn matmul_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
    MatMul::new(xs).borrow_mut().call()
}

pub fn matmul(x: &RcVariable, w: &RcVariable) -> FrameResult<RcVariable> {
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
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "TensorDot",
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
        //xs[0]の方をX, xs[1]の方をWとする
        let x = &xs[0];
        let w = &xs[1];

        let x_data = x.data();
        let w_data = w.data();

        let y_data = x_data
            .tensordot(&w_data)
            .map_err(|e| FrameError::ForwardError {
                function: "TensorDot",
                source: e,
            })?;

        Ok(y_data.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x = &self.inputs[0];
        let w = &self.inputs[1];

        // TODO:Tensorにtensordotを実装してから修正
        let (gx, gw) = tensordot_backward(gy, x, w)?;

        let gxs = vec![gx, gw];

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

fn tensordot_backward(
    gy: &RcVariable,
    x: &RcVariable,
    w: &RcVariable,
) -> FrameResult<(RcVariable, RcVariable)> {
    let (gx, gw) = match (x.data().ndim(), w.data().ndim()) {
        // 3D × 2D
        //(N,k,l) ×　(l,m) -> (N,k,m)の場合
        (3, 2) => {
            let gx = tensordot(gy, &w.t()?)?;

            let gw = tensordot(&x.permute_axes(vec![0, 2, 1])?, gy)?.sum(Some(0))?;

            /*

            let n = x.data().shape().dims()[0];
            let k = x.data().shape().dims()[1];
            let l = x.data().shape().dims()[2];
            let m = w.data().shape().dims()[1];


            let gw = matmul(
                &x.reshape(&Shape::new(vec![n * k, l])?)?.t()?,
                &gy.reshape(&Shape::new(vec![n * k, m])?)?,
            )?;

            */

            (gx, gw)
        }

        // 2D × 3D
        //(k,l) ×　(N,l,m) -> (N,k,m)
        (2, 3) => {
            /*
            let k = x.data().shape().dims()[0];
            let l = x.data().shape().dims()[1];
            let n = w.data().shape().dims()[0];
            let m = w.data().shape().dims()[2];



            //(n,k,m) -> (k,n,m) -> (k,n*m)
            let gy1 = permute_axes(&gy, vec![1, 0, 2])?.reshape(&Shape::new(vec![k, n * m])?)?;

            //(n,l,m) -> (l,n,m) -> (l,n*m) -> (n*m,l)

            let w1 = permute_axes(&w, vec![1, 0, 2])?
                .reshape(&Shape::new(vec![l, n * m])?)?
                .t()?;
            let gx = matmul(&gy1, &w1)?; //(k,n*m) @ (n*m,l) -> (k,l)

            */

            let tmp = tensordot(gy, &w.permute_axes(vec![0, 2, 1])?)?;

            println!("tmp shape = {:?}", tmp.data().shape());

            let gx = tensordot(gy, &w.permute_axes(vec![0, 2, 1])?)?.sum(Some(0))?;

            println!("gx shape = {:?}", gx.data().shape());

            let gw = tensordot(&x.t()?, gy)?; //(l,k) @' (n,k,m) -> (n,l,m)

            println!("gw shape = {:?}", gw.data().shape());

            (gx, gw)
        }

        // 3D × 3D
        (3, 3) => {
            let w_t = permute_axes(&w, vec![0, 2, 1])?;
            let gx = tensordot(gy, &w_t)?;
            let x_t = permute_axes(&x, vec![0, 2, 1])?;
            let gw = tensordot(&x_t, gy)?;

            (gx, gw)
        }

        _ => {
            panic!("4次元以上または2次元以下の行列積は未実装。");
        }
    };

    Ok((gx, gw))
}

fn tensordot_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
    TensorDot::new(xs).borrow_mut().call()
}

/// 2次元と3次元の行列積の関数
/// (2D×3D), (3D×2D), (3D×3D)に対応
pub fn tensordot(x: &RcVariable, w: &RcVariable) -> FrameResult<RcVariable> {
    let y = tensordot_f(&[x.clone(), w.clone()]);
    y
}
