use std::cell::RefCell;
use std::rc::Rc;

use ndarray::ArrayD;

use crate::core_new::RcVariable;
use crate::layers::Layer;
use crate::models::Model;

pub trait Optimizer {
    fn setup(&mut self, target_model: &impl Model);
    fn update(&self);
    fn update_param(&self, param: &RcVariable) -> ArrayD<f32>;
    fn set_hooks(&mut self);
}

pub struct SGD {
    lr: f32,
    layers: Option<Rc<RefCell<Vec<Box<dyn Layer + 'static>>>>>,
}

impl Optimizer for SGD {
    fn setup(&mut self, target_model: &impl Model) {
        self.layers = Some(target_model.layers().clone());
    }

    fn update(&self) {
        for layer in self
            .layers
            .as_ref()
            .expect("SGDにModelが設定されていません")
            .borrow_mut()
            .iter_mut()
        {
            for (_id, param) in layer.params() {
                let new_param = self.update_param(&param);
                param.0.borrow_mut().data = new_param;
            }
        }
    }
    fn update_param(&self, param: &RcVariable) -> ArrayD<f32> {
        let current_param_data = param.data();
        let param_grad = param
            .grad()
            .as_ref()
            .expect("SGDで更新中のパラメータにgradがありません")
            .data();

        let new_param_data = current_param_data - self.lr * param_grad;
        new_param_data
    }
    fn set_hooks(&mut self) {}
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr: lr,
            layers: None,
        }
    }
}
