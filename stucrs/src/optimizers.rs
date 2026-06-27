use fxhash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;
use thiserror::Error;

use crate::core::{F32ToTensor, RcVariable};
use crate::error::{FrameError, FrameResult};
use crate::layers::Layer;
use crate::models::Model;
use crate::tensor::tensor::Tensor;

pub trait Optimizer {
    fn setup(&mut self, target_model: &impl Model);
    fn update(&mut self) -> FrameResult<()>;
    fn update_param(&mut self, param: &RcVariable) -> FrameResult<Tensor>;
    fn set_hooks(&mut self);
}

pub struct SGD {
    lr: Tensor,
    layers: Option<Rc<RefCell<Vec<Box<dyn Layer + 'static>>>>>,
}

impl Optimizer for SGD {
    fn setup(&mut self, target_model: &impl Model) {
        self.layers = Some(target_model.layers().clone());
    }

    fn update(&mut self) -> FrameResult<()> {
        let params = {
            let mut layers = self
                .layers
                .as_ref()
                .ok_or(FrameError::OptimizerError(OptimizerError::MissingModel {
                    optimizer: "SGD",
                }))?
                .borrow_mut();

            let mut params_vec: Vec<RcVariable> = Vec::new();

            for layer in layers.iter_mut().filter(|l| l.has_params()) {
                params_vec.extend(layer.params()?.values().cloned());
            }
            params_vec
        };

        for param in params {
            let new_param = self.update_param(&param)?;
            param.0.borrow_mut().data = new_param;
        }
        Ok(())
    }
    fn update_param(&mut self, param: &RcVariable) -> FrameResult<Tensor> {
        let current_param_data = param.data();
        let param_grad = param
            .grad()
            .as_ref()
            .ok_or(FrameError::OptimizerError(OptimizerError::NoGrad {
                optimizer: "SGD",
            }))?
            .data();

        let new_param_data = (current_param_data - (self.lr.clone() * param_grad)?)?;
        Ok(new_param_data)
    }
    fn set_hooks(&mut self) {}
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr: lr.ts(),
            layers: None,
        }
    }
}

pub struct MomentumSGD {
    lr: Tensor,
    momentum: Tensor,
    vs: FxHashMap<usize, Tensor>,
    layers: Option<Rc<RefCell<Vec<Box<dyn Layer + 'static>>>>>,
}

impl Optimizer for MomentumSGD {
    fn setup(&mut self, target_model: &impl Model) {
        self.layers = Some(target_model.layers().clone());
    }

    fn update(&mut self) -> FrameResult<()> {
        let params = {
            let mut layers = self
                .layers
                .as_ref()
                .ok_or(FrameError::OptimizerError(OptimizerError::MissingModel {
                    optimizer: "MomentumSGD",
                }))?
                .borrow_mut();

            let mut params_vec: Vec<RcVariable> = Vec::new();

            for layer in layers.iter_mut().filter(|l| l.has_params()) {
                params_vec.extend(layer.params()?.values().cloned());
            }
            params_vec
        };

        for param in params {
            let new_param = self.update_param(&param)?;
            param.0.borrow_mut().data = new_param;
        }

        Ok(())
    }
    fn update_param(&mut self, param: &RcVariable) -> FrameResult<Tensor> {
        let current_param_data = param.data();
        let param_grad = param
            .grad()
            .as_ref()
            .ok_or(FrameError::OptimizerError(OptimizerError::NoGrad {
                optimizer: "MomentumSGD",
            }))?
            .data();

        let param_id = param.id();
        if self.vs.contains_key(&param_id) == false {
            let new_v = Tensor::zeros(current_param_data.shape().dims.clone())?;
            self.vs.insert(param_id, new_v);
        }
        let mut v = self.vs.get(&param_id).unwrap().clone();
        v = (v * self.momentum.clone())?;
        v = (v - (self.lr.clone() * param_grad)?)?;
        let new_param_data = (current_param_data + v)?;

        Ok(new_param_data)
    }
    fn set_hooks(&mut self) {}
}

impl MomentumSGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr: lr.ts(),
            momentum: momentum.ts(),
            vs: FxHashMap::default(),
            layers: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("オプティマイザー:{optimizer}にモデルが設定されていません。")]
    MissingModel { optimizer: &'static str },

    #[error(
        "オプティマイザー:{optimizer}の更新中にgradを保持しないパラメーターが見つかりました。"
    )]
    NoGrad { optimizer: &'static str },
}
