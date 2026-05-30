use crate::config::id_generator;
use crate::core::{F32ToTensor, TensorToRcVariable};
use crate::core::{RcVariable, Variable};
use crate::error::{FrameError, FrameResult};
use crate::functions::activation_funcs::{relu, sigmoid_simple};
use crate::functions::math::tanh;
use crate::functions::matrix::reshape;
use crate::functions::neural_funcs::{dropout, linear_simple};
use crate::functions_cnn::{conv2d_simple, max_pool2d_simple};
use crate::tensor::lib::{Shape, Tensor};

use fxhash::FxHashMap;

use std::cell::RefCell;

use thiserror::Error;

use std::fmt::Debug;

use std::rc::Weak;

///Model構造体が保持するLayerを表すトレイト。
///
/// # 概要
/// このトレイトを実装するとModelがLayerを保持、管理し、
///
/// 重みやバックプロパゲーションを自動でModel側から行える。
///
/// ## 実装上の注意
/// 重みをもたないLayerも作成可能(Maxpool2dや活性化関数のLayerなど)。
/// その場合、set_params(),params()やcleargrad()の関数は不要なので、LayerErrorのNoParameterErrorを用いる。
/// Maxpool2dを参照。
pub trait Layer: Debug {
    fn set_params(&mut self, param: &RcVariable) -> FrameResult<()>;

    fn get_input(&self) -> RcVariable;
    fn get_output(&self) -> RcVariable;
    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable>;
    fn get_generation(&self) -> i32;
    fn get_id(&self) -> usize;
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>>;
    fn cleargrad(&mut self);
    fn has_params(&self) -> bool;
}

///線形変換(Linear)を処理するLayer構造体
///
/// new()よる呼び出しでResult型で返す
///
/// # Examples
///
///     let mut model = BaseModel::new();
///     model.stack(L::Linear::new(10, true, None)?);
///
#[derive(Debug, Clone)]
pub struct Linear {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    out_size: u32,
    w_id: Option<usize>,
    b_id: Option<usize>,
    params: FxHashMap<usize, RcVariable>,
    generation: i32,
    id: usize,
}

impl Layer for Linear {
    fn set_params(&mut self, param: &RcVariable) -> FrameResult<()> {
        self.params.insert(param.id(), param.clone());
        Ok(())
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Ok(&mut self.params)
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }

    fn has_params(&self) -> bool {
        true
    }
}

impl Linear {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        if let None = &self.w_id {
            let i = x.data().shape().dims()[1];
            let o = self.out_size as usize;
            let i_f32 = i as f32;

            let w_data = Tensor::standard_normal(vec![i, o])? * ((1.0f32 / i_f32).sqrt()).ts();
            /*
            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
                &Array::random((i, o), StandardNormal) * ((1.0f32 / i_f32).sqrt());
            */
            let w = w_data?.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone())?;
        }

        // フィールドでパラメータのidを保持しているので、idでパラメータを呼び出す
        let w_id = self.w_id.unwrap();
        let w = self.params.get(&w_id).unwrap();

        //bはoption型なので、場合分け
        let b;
        if let Some(b_id_data) = self.b_id {
            b = self.params.get(&b_id_data).cloned();
        } else {
            b = None;
        }

        let y = linear_simple(&x, &w, &b)?;

        Ok(y)
    }

    pub fn new(out_size: u32, biased: bool, opt_in_size: Option<usize>) -> FrameResult<Self> {
        let mut linear = Self {
            input: None,
            output: None,
            out_size: out_size,
            w_id: None,
            b_id: None,
            params: FxHashMap::default(),
            generation: 0,
            id: id_generator(),
        };

        //in_sizeが設定されていたら、ここでWを作成
        //されていない場合は後で作成
        if let Some(in_size) = opt_in_size {
            let i = in_size as usize;
            let o = out_size as usize;

            let i_f32 = in_size as f32;

            let w_data = Tensor::standard_normal(vec![i, o])? * ((1.0f32 / i_f32).sqrt()).ts();

            let w = w_data?.rv();

            linear.w_id = Some(w.id());
            linear.set_params(&w.clone())?;
        }

        if biased == true {
            let b = Tensor::zeros(vec![1, out_size as usize])?.rv();
            linear.b_id = Some(b.id());
            linear.set_params(&b.clone())?;
        }

        Ok(linear)
    }
}

///線形変換(Linear)と活性化関数をまとめて計算するLayer構造体
///
/// new()よる呼び出しでResult型で返す
///
/// # Examples
///
///     let mut model = BaseModel::new();
///     model.stack(L::Dense::new(1000, true, None, Activation::Sigmoid)?);
///
#[derive(Debug, Clone)]
pub struct Dense {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    out_size: u32,
    w_id: Option<usize>,
    b_id: Option<usize>,
    activation: Activation,
    params: FxHashMap<usize, RcVariable>,
    generation: i32,
    id: usize,
}

impl Layer for Dense {
    fn set_params(&mut self, param: &RcVariable) -> FrameResult<()> {
        self.params.insert(param.id(), param.clone());
        Ok(())
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputsを覚える
        self.input = Some(input.downgrade());

        self.generation = input.generation();

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Ok(&mut self.params)
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }

    fn has_params(&self) -> bool {
        true
    }
}

impl Dense {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        if let None = &self.w_id {
            let i = x.data().shape().dims()[1];
            let o = self.out_size as usize;
            let i_f32 = i as f32;

            let w_data = Tensor::standard_normal(vec![i, o])? * ((1.0f32 / i_f32).sqrt()).ts();

            let w = w_data?.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone())?;
        }

        // フィールドでパラメータのidを保持しているので、idでパラメータを呼び出す
        let w_id = self.w_id.unwrap();
        let w = self.params.get(&w_id).unwrap();

        //bはoption型なので、場合分け
        let b;
        if let Some(b_id_data) = self.b_id {
            b = self.params.get(&b_id_data).cloned();
        } else {
            b = None;
        }

        let t = linear_simple(&x, &w, &b)?;

        let y = match self.activation {
            Activation::Sigmoid => sigmoid_simple(&t)?,
            Activation::Relu => relu(&t)?,
            Activation::Tanh => tanh(&t)?,
        };

        Ok(y)
    }

    pub fn new(
        out_size: u32,
        biased: bool,
        opt_in_size: Option<u32>,
        activation: Activation,
    ) -> FrameResult<Self> {
        let mut dense = Self {
            input: None,
            output: None,
            out_size: out_size,
            w_id: None,
            b_id: None,
            activation: activation,
            params: FxHashMap::default(),
            generation: 0,
            id: id_generator(),
        };

        //in_sizeが設定されていたら、ここでWを作成
        //されていない場合は後で作成
        if let Some(in_size) = opt_in_size {
            let i = in_size as usize;
            let o = out_size as usize;

            let i_f32 = in_size as f32;

            let w_data = Tensor::standard_normal(vec![i, o])? * ((1.0f32 / i_f32).sqrt()).ts();

            let w = w_data?.rv();

            dense.w_id = Some(w.id());
            dense.set_params(&w.clone())?;
        }

        if biased == true {
            let b = Tensor::zeros(vec![1, out_size as usize])?.rv();
            dense.b_id = Some(b.id());
            dense.set_params(&b.clone())?;
        }

        Ok(dense)
    }
}
/// Conv2d関数を処理するLayer構造体
///
/// new()よる呼び出しでResult型で返す
///
/// # Examples
///
///     let out_channels = 4;
///     let kernel_size = (3, 3);
///     let stride_size = (1, 1);
///     let pad_size = (0, 0);
///     let biased = false;
///     let mut model = BaseModel::new();
///     model.stack(L::Conv2d::new(out_channels, kernel_size, stride_size, pad_size, biased)?);
///
///conv2d_simple関数でbiasの処理が未実装なので、このLayerでもbiasはまだ使えない。
#[derive(Debug, Clone)]
pub struct Conv2d {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    w_id: Option<usize>,
    b_id: Option<usize>,
    params: FxHashMap<usize, RcVariable>,
    generation: i32,
    id: usize,
}

impl Layer for Conv2d {
    fn set_params(&mut self, param: &RcVariable) -> FrameResult<()> {
        self.params.insert(param.id(), param.clone());
        Ok(())
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Ok(&mut self.params)
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }

    fn has_params(&self) -> bool {
        true
    }
}

impl Conv2d {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        if let None = &self.w_id {
            let c = x.data().shape().dims()[1];
            let oc = self.out_channels;
            let (kh, kw) = self.kernel_size;

            let scale = (1.0f32 / ((c * kh * kw) as f32)).sqrt();

            let w_data = Tensor::standard_normal(vec![oc, c, kh, kw])? * scale.ts();
            /*
            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
                &Array::random((oc, c, kh, kw), StandardNormal) * scale;
            */
            let w = w_data?.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone())?;
        }

        // フィールドでパラメータのidを保持しているので、idでパラメータを呼び出す
        let w_id = self.w_id.unwrap();
        let w = self.params.get(&w_id).unwrap();

        //bはoption型なので、場合分け
        let b;
        if let Some(b_id_data) = self.b_id {
            b = self.params.get(&b_id_data).cloned();
        } else {
            b = None;
        }

        let y = conv2d_simple(x, w, b, self.stride_size, self.pad_size)?;

        Ok(y)
    }

    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
        biased: bool,
    ) -> FrameResult<Self> {
        let mut conv2d = Self {
            input: None,
            output: None,
            out_channels: out_channels,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            w_id: None,
            b_id: None,
            params: FxHashMap::default(),
            generation: 0,
            id: id_generator(),
        };

        if biased == true {
            let b = Tensor::zeros(vec![out_channels as usize])?.rv();
            conv2d.b_id = Some(b.id());
            conv2d.set_params(&b.clone())?;
        }

        Ok(conv2d)
    }
}

/// Maxpool2dを処理するLayer構造体
///
/// # Examples
///
///     let kernel_size = (2, 2);
///     let stride_size = (1, 1);
///     let pad_size = (0, 0);
///     let mut model = BaseModel::new();
///     model.stack(L::Maxpool2d::new(kernel_size, stride_size, pad_size));
///
#[derive(Debug, Clone)]
pub struct Maxpool2d {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
    generation: i32,
    id: usize,
}

impl Layer for Maxpool2d {
    fn set_params(&mut self, _param: &RcVariable) -> FrameResult<()> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Maxpool2d"),
        })) //Maxpool2dはparamsを持たないので
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Maxpool2d"),
        }))
    }

    fn cleargrad(&mut self) {}

    fn has_params(&self) -> bool {
        false
    }
}

impl Maxpool2d {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        let y = max_pool2d_simple(x, self.kernel_size, self.stride_size, self.pad_size)?;

        Ok(y)
    }

    pub fn new(
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
    ) -> Self {
        let maxpool2d = Self {
            input: None,
            output: None,
            kernel_size: kernel_size,
            stride_size: stride_size,
            pad_size: pad_size,
            generation: 0,
            id: id_generator(),
        };

        maxpool2d
    }
}

/// Dropoutを処理するLayer構造体
///
/// # Examples
///
///     let input_array = array![[4.0f32, 1.0, 5.0, 3.0], [1.0, 5.0, 3.0, 9.0]];
///     let ratio = 0.5f32;
///     let mut model = BaseModel::new();
///     model.stack(L::Dropout::new(ratio));
///
#[derive(Debug, Clone)]
pub struct Dropout {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    ratio: f32,
    generation: i32,
    id: usize,
}

impl Layer for Dropout {
    fn set_params(&mut self, _param: &RcVariable) -> FrameResult<()> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Dropout"),
        })) //Dropoutはparamsを持たないので
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Dropout"),
        }))
    }

    fn cleargrad(&mut self) {}

    fn has_params(&self) -> bool {
        false
    }
}

impl Dropout {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        let y = dropout(x, self.ratio)?;

        Ok(y)
    }

    pub fn new(ratio: f32) -> Self {
        let dropout = Self {
            input: None,
            output: None,
            ratio: ratio,
            generation: 0,
            id: id_generator(),
        };

        dropout
    }
}

#[derive(Debug, Clone)]
pub struct ActivationLayer {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    activation: Activation,
    generation: i32,
    id: usize,
}

impl Layer for ActivationLayer {
    fn set_params(&mut self, _param: &RcVariable) -> FrameResult<()> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("ActivationLayer"),
        }))
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("ActivationLayer"),
        }))
    }

    fn cleargrad(&mut self) {}

    fn has_params(&self) -> bool {
        false
    }
}

impl ActivationLayer {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        let y = match self.activation {
            Activation::Sigmoid => sigmoid_simple(&x)?,
            Activation::Relu => relu(&x)?,
            Activation::Tanh => tanh(&x)?,
        };

        Ok(y)
    }

    pub fn new(activation: Activation) -> Self {
        let activations_layer = Self {
            input: None,
            output: None,
            activation: activation,
            generation: 0,
            id: id_generator(),
        };

        activations_layer
    }
}

#[derive(Debug, Clone)]
pub struct Flatten {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    generation: i32,
    id: usize,
}

impl Layer for Flatten {
    fn set_params(&mut self, _param: &RcVariable) -> FrameResult<()> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Flatten"),
        }))
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Err(FrameError::LayerError(LayerError::NoParameterError {
            layer: ("Flatten"),
        }))
    }

    fn cleargrad(&mut self) {}

    fn has_params(&self) -> bool {
        false
    }
}

impl Flatten {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        let x_data = x.data();
        let x_shape = x_data.shape();
        let n = x_shape.dims()[0];
        let numel = x_shape.numel();

        let new_shape = Shape::new(vec![n, numel / n])?;

        let y = reshape(&x, &new_shape)?;

        Ok(y)
    }

    pub fn new() -> Self {
        let flatten = Self {
            input: None,
            output: None,
            generation: 0,
            id: id_generator(),
        };

        flatten
    }
}

#[derive(Debug, Clone)]
pub struct RNN {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    x2h: Linear,
    h2h: Linear,
    h: Option<RcVariable>,
    out_size: u32,
    w_id: Option<usize>,
    b_id: Option<usize>,
    activation: Activation,
    params: FxHashMap<usize, RcVariable>,
    generation: i32,
    id: usize,
}

impl Layer for RNN {
    fn set_params(&mut self, param: &RcVariable) -> FrameResult<()> {
        self.params.insert(param.id(), param.clone());
        Ok(())
    }
    fn get_input(&self) -> RcVariable {
        let input = self
            .input
            .as_ref()
            .unwrap()
            .upgrade()
            .as_ref()
            .unwrap()
            .clone();
        RcVariable(input)
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

    fn call(&mut self, input: &RcVariable) -> FrameResult<RcVariable> {
        // inputのvariableからdataを取り出す

        let output = self.forward(input)?;

        //ここから下の処理はbackwardするときだけ必要。

        //　inputsを覚える
        self.input = Some(input.downgrade());

        self.generation = input.generation();

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        Ok(output)
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> FrameResult<&mut FxHashMap<usize, RcVariable>> {
        Ok(&mut self.params)
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }

    fn has_params(&self) -> bool {
        true
    }
}

impl RNN {
    fn forward(&mut self, x: &RcVariable) -> FrameResult<RcVariable> {
        if let Some(h_rc) = &self.h {
            let _h_new = tanh(&(self.x2h.call(&x)? + self.h2h.call(h_rc)?))?;
        }
        if let None = &self.w_id {
            let i = x.data().shape().dims()[1];
            let o = self.out_size as usize;
            let i_f32 = i as f32;

            let w_data = Tensor::standard_normal(vec![i, o])? * ((1.0f32 / i_f32).sqrt()).ts();

            let w = w_data?.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone())?;
        }

        // フィールドでパラメータのidを保持しているので、idでパラメータを呼び出す
        let w_id = self.w_id.unwrap();
        let w = self.params.get(&w_id).unwrap();

        //bはoption型なので、場合分け
        let b;
        if let Some(b_id_data) = self.b_id {
            b = self.params.get(&b_id_data).cloned();
        } else {
            b = None;
        }

        let t = linear_simple(&x, &w, &b)?;

        let y = match self.activation {
            Activation::Sigmoid => sigmoid_simple(&t)?,
            Activation::Relu => relu(&t)?,
            Activation::Tanh => tanh(&t)?,
        };

        Ok(y)
    }

    pub fn new(
        out_size: u32,
        biased: bool,
        opt_in_size: Option<usize>,
        activation: Activation,
    ) -> FrameResult<Self> {
        let mut rnn = Self {
            input: None,
            output: None,
            x2h: Linear::new(out_size, biased, opt_in_size)?,
            h2h: Linear::new(out_size, false, opt_in_size)?,
            h: None,
            out_size: out_size,
            w_id: None,
            b_id: None,
            activation: activation,
            params: FxHashMap::default(),
            generation: 0,
            id: id_generator(),
        };

        //in_sizeが設定されていたら、ここでWを作成
        //されていない場合は後で作成
        if let Some(in_size) = opt_in_size {
            let i = in_size as usize;
            let o = out_size as usize;

            let i_f32 = in_size as f32;

            let w_data = Tensor::standard_normal(vec![i, o])? * ((1.0f32 / i_f32).sqrt()).ts();

            let w = w_data?.rv();

            rnn.w_id = Some(w.id());
            rnn.set_params(&w.clone())?;
        }

        if biased == true {
            let b = Tensor::zeros(vec![out_size as usize])?.rv();
            rnn.b_id = Some(b.id());
            rnn.set_params(&b.clone())?;
        }

        Ok(rnn)
    }
}

/// 実装している活性化関数をまとめた列挙型
///
/// 今後新たに活性化関数を実装したら、名前をここに追加
#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
}

#[derive(Debug, Error)]
pub enum LayerError {
    #[error("レイヤー:{layer}はパラメーターを保持していません。")]
    NoParameterError { layer: &'static str },
}
