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

use ndarray_stats::QuantileExt;
use std::rc::{Rc, Weak};
use std::vec;

//use std::thread;
//use std::time::Duration;

use crate::config::{get_grad_status, id_generator};
use crate::core_new::*;

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
                    todo!("2次元のargmax関数の軸を指定なしは後で対応")
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
                    unimplemented!("指定した軸には対応していません。")
                }
            };
            y_data
        }
        3 => {
            let x_array = x_array.into_dimensionality::<Ix3>().unwrap();
            let y_array: Array2<usize> = match axis {
                None => {
                    todo!("3次元のargmax関数の軸を指定なしは後で対応")
                }
                Some(0) => {
                    todo!("3次元の軸0はまだ未対応")
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
                    unimplemented!("その他の軸は対応していません")
                }
            };
            y_array.into_dyn()
        }
        _ => {
            unimplemented!("1-3次元以外の次元には対応していません")
        }
    };
    y_array
}
