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
    fn update(&self) -> FrameResult<()>;
    fn update_param(&self, param: &RcVariable) -> FrameResult<Tensor>;
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

    fn update(&self) -> FrameResult<()> {
        for layer in self
            .layers
            .as_ref()
            .ok_or(FrameError::OptimizerError(OptimizerError::MissingModel {
                optimizer: "SGD",
            }))?
            .borrow_mut()
            .iter_mut()
            .filter(|layer| layer.has_params())
        {
            for (_id, param) in layer.params()? {
                let new_param = self.update_param(&param)?;
                param.0.borrow_mut().data = new_param;
            }
        }
        Ok(())
    }
    fn update_param(&self, param: &RcVariable) -> FrameResult<Tensor> {
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

#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("オプティマイザー:{optimizer}にモデルが設定されていません。")]
    MissingModel { optimizer: &'static str },

    #[error(
        "オプティマイザー:{optimizer}の更新中にgradを保持しないパラメーターが見つかりました。"
    )]
    NoGrad { optimizer: &'static str },
}
