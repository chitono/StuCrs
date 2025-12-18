use crate::config::id_generator;
use crate::core_new::ArrayDToRcVariable;
use crate::core_new::{RcVariable, Variable};
use crate::functions::activation_funcs::{relu, sigmoid_simple};
use crate::functions::math::tanh;
use crate::functions::neural_funcs::linear_simple;
use crate::functions_cnn::{conv2d_simple, max_pool2d_simple};

use fxhash::FxHashMap;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;

use std::fmt::Debug;

use std::rc::Weak;

pub trait Layer: Debug {
    fn set_params(&mut self, param: &RcVariable);

    fn get_input(&self) -> RcVariable;
    fn get_output(&self) -> RcVariable;
    fn call(&mut self, input: &RcVariable) -> RcVariable;
    fn get_generation(&self) -> i32;
    fn get_id(&self) -> usize;
    fn params(&mut self) -> &mut FxHashMap<usize, RcVariable>;
    fn cleargrad(&mut self);
}

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
    fn set_params(&mut self, param: &RcVariable) {
        self.params.insert(param.id(), param.clone());
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

    fn call(&mut self, input: &RcVariable) -> RcVariable {
        // inputのvariableからdataを取り出す

        let output = self.forward(input);

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> &mut FxHashMap<usize, RcVariable> {
        &mut self.params
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }
}

impl Linear {
    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        if let None = &self.w_id {
            let i = x.data().shape()[1];
            let o = self.out_size as usize;
            let i_f32 = i as f32;

            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
                &Array::random((i, o), StandardNormal) * ((1.0f32 / i_f32).sqrt());

            let w = w_data.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone());
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

        let y = linear_simple(&x, &w, &b);

        y
    }

    pub fn new(out_size: u32, biased: bool, opt_in_size: Option<u32>) -> Self {
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

            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
                &Array::random((i, o), StandardNormal) * ((1.0f32 / i_f32).sqrt());

            let w = w_data.rv();

            linear.w_id = Some(w.id());
            linear.set_params(&w.clone());
        }

        if biased == true {
            let b = Array::zeros(out_size as usize).rv();
            linear.b_id = Some(b.id());
            linear.set_params(&b.clone());
        }

        linear
    }

    pub fn update_params(&mut self, lr: f32) {
        for (_id, param) in self.params.iter() {
            let param_data = param.data();
            let current_grad = param.grad().as_ref().unwrap().data();
            param.0.borrow_mut().data = param_data - lr * current_grad;
        }
    }
}

///線形変換(Linear)と活性化関数をまとめて計算するLayer構造体.
/// new()で呼び出す際、activationのところはenumのActivationから選び、渡す。
/// 例...Dense::new(1000, true, None, Activation::Sigmoid)
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
    fn set_params(&mut self, param: &RcVariable) {
        self.params.insert(param.id(), param.clone());
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

    fn call(&mut self, input: &RcVariable) -> RcVariable {
        // inputのvariableからdataを取り出す

        let output = self.forward(input);

        //ここから下の処理はbackwardするときだけ必要。

        //　inputsを覚える
        self.input = Some(input.downgrade());

        self.generation = input.generation();

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> &mut FxHashMap<usize, RcVariable> {
        &mut self.params
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }
}

impl Dense {
    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        if let None = &self.w_id {
            let i = x.data().shape()[1];
            let o = self.out_size as usize;
            let i_f32 = i as f32;

            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
                &Array::random((i, o), StandardNormal) * ((1.0f32 / i_f32).sqrt());

            let w = w_data.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone());
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

        let t = linear_simple(&x, &w, &b);

        let y = match self.activation {
            Activation::Sigmoid => sigmoid_simple(&t),
            Activation::Relu => relu(&t),
            Activation::Tanh => tanh(&t),
        };

        y
    }

    pub fn new(
        out_size: u32,
        biased: bool,
        opt_in_size: Option<u32>,
        activation: Activation,
    ) -> Self {
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

            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> =
                &Array::random((i, o), StandardNormal) * ((1.0f32 / i_f32).sqrt());

            let w = w_data.rv();

            dense.w_id = Some(w.id());
            dense.set_params(&w.clone());
        }

        if biased == true {
            let b = Array::zeros(out_size as usize).rv();
            dense.b_id = Some(b.id());
            dense.set_params(&b.clone());
        }

        dense
    }

    pub fn update_params(&mut self, lr: f32) {
        for (_id, param) in self.params.iter() {
            let param_data = param.data();
            let current_grad = param.grad().as_ref().unwrap().data();
            param.0.borrow_mut().data = param_data - lr * current_grad;
        }
    }
}
/// Conv2d関数を処理するLayer構造体
///
/// ## 実装例
///     let kernel_size = (2, 2);
///     let stride_size = (1, 1);
///     let pad_size = (0, 0);
///     let mut model = BaseModel::new();
///     model.stack(L::Maxpool2d::new(kernel_size, stride_size, pad_size));
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
    fn set_params(&mut self, param: &RcVariable) {
        self.params.insert(param.id(), param.clone());
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

    fn call(&mut self, input: &RcVariable) -> RcVariable {
        // inputのvariableからdataを取り出す

        let output = self.forward(input);

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> &mut FxHashMap<usize, RcVariable> {
        &mut self.params
    }

    fn cleargrad(&mut self) {
        for (_id, param) in self.params.iter_mut() {
            param.cleargrad();
        }
    }
}

impl Conv2d {
    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        if let None = &self.w_id {
            let c = x.data().shape()[1];
            let oc = self.out_channels;
            let (kh, kw) = self.kernel_size;

            let scale = (1.0f32 / ((c * kh * kw) as f32)).sqrt();

            let w_data: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> =
                &Array::random((oc, c, kh, kw), StandardNormal) * scale;

            let w = w_data.rv();

            self.w_id = Some(w.id());
            self.set_params(&w.clone());
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

        let y = conv2d_simple(x, w, b, self.stride_size, self.pad_size);

        y
    }

    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride_size: (usize, usize),
        pad_size: (usize, usize),
        biased: bool,
    ) -> Self {
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
            let b = Array::zeros(out_channels as usize).rv();
            conv2d.b_id = Some(b.id());
            conv2d.set_params(&b.clone());
        }

        conv2d
    }
}

/// Maxpool2dを処理するLayer構造体
///
/// ## 実装例
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
    fn set_params(&mut self, _param: &RcVariable) {
        unimplemented!("Maxpool2dはパラメータを保持していません。") //Maxpool2dはparamsを持たないので
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

    fn call(&mut self, input: &RcVariable) -> RcVariable {
        // inputのvariableからdataを取り出す

        let output = self.forward(input);

        //ここから下の処理はbackwardするときだけ必要。

        //　inputを弱参照で覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        output
    }

    fn get_generation(&self) -> i32 {
        self.generation
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn params(&mut self) -> &mut FxHashMap<usize, RcVariable> {
        unimplemented!("Maxpool2dはパラメータを保持していません。")
    }

    fn cleargrad(&mut self) {
        unimplemented!("Maxpool2dはパラメータを保持していません。")
    }
}

impl Maxpool2d {
    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        let y = max_pool2d_simple(x, self.kernel_size, self.stride_size, self.pad_size);

        y
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

/// 実装している活性化関数をまとめた列挙型
///
/// 今後新たに活性化関数を実装したら、名前をここに追加
#[derive(Debug, Clone)]
pub enum Activation {
    Sigmoid,
    Relu,
    Tanh,
}
