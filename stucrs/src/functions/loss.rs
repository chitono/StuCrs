use core::panic;
use std::cell::RefCell;
//use std::clone;

use std::fmt::Debug;

//use std::sync::Mutex;
//use std::future;
//use std::hash::Hash;
//use std::process::Output;

use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::config::{get_grad_status, id_generator};
use crate::core_new::*;

use crate::error::FrameResult;
use crate::functions::activation_funcs::*;
use crate::functions::math::*;
use crate::functions::matrix::*;
use crate::tensor::lib::TensorOps;
use crate::tensor::tensor::Tensor;

#[derive(Debug, Clone)]
struct MeanSquaredError {
    inputs: Vec<RcVariable>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Function for MeanSquaredError {
    fn call(&mut self) -> FrameResult<RcVariable> {
        let inputs = &self.inputs;
        if inputs.len() != 2 {
            return Err(crate::error::FrameError::InvalidInputCount {
                function: "MeanSquaredError",
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
        let x0 = &xs[0];
        let x1 = &xs[1];

        let diff = (x0.data() - x1.data())?;
        let len = diff.shape().dims()[0] as f32;
        let len_tensor = Tensor::from_vec(vec![len], vec![1])?;

        let error_data = diff.pow(2.0)?.sum(None, false)? / len_tensor;

        //let error_data = array_sum(&diff.mapv(|x| x.powf(2.0)).view(), None) / len;

        Ok(error_data?.rv())
    }

    fn backward(&self, gy: &RcVariable) -> FrameResult<Vec<RcVariable>> {
        let x0 = &self.inputs[0];
        let x1 = &self.inputs[1];

        let diff = x0.clone() - x1.clone();
        let diff_data = diff.data();
        let diff_shape = diff_data.shape();
        let gy = broadcast_to(&gy, diff_shape)?;

        let gx0 = gy.clone() * diff.clone() * (2.0.rv() / (diff.len() as f32).rv());
        let gx1 = -gx0.clone();
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

fn mean_squared_error_f(xs: &[RcVariable]) -> FrameResult<RcVariable> {
    MeanSquaredError::new(xs).borrow_mut().call()
}

pub fn mean_squared_error(x0: &RcVariable, x1: &RcVariable) -> FrameResult<RcVariable> {
    let y = mean_squared_error_f(&[x0.clone(), x1.clone()]);
    y
}

/// ここで渡すtはone-hotベクトル状態の教師データ
pub fn softmax_cross_entropy_simple(x: &RcVariable, t: &RcVariable) -> FrameResult<RcVariable> {
    if x.data().shape() != t.data().shape() {
        panic!("交差エントロピー誤差でのxとtの形状が異なります。tがone-hotベクトルでない可能性があります。")
    }

    let n = x.data().shape().dims()[0] as f32;

    let p = softmax_simple(&x)?;

    let clamped_p = clamp(&p, 1.0e-4, 1.0)?;

    let log_p = log(&clamped_p, None)?;

    let tlog_p = log_p * t.clone();

    let y = (-sum(&tlog_p, None)?) / n.rv();
    Ok(y)
}
