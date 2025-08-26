use crate::core_new::{RcVariable, Variable};
use crate::layers::{self, Layer};
use fxhash::FxHashMap;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub trait Model {
    fn stack(&mut self, layer: impl Layer + 'static);
    fn forward(&mut self, x: &RcVariable) -> RcVariable;
    fn layers(&mut self) -> &mut Vec<Box<dyn Layer + 'static>>;
    fn cleargrad(&mut self);
}

pub struct BaseModel {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    layers: Vec<Box<dyn Layer + 'static>>,
}

impl Model for BaseModel {
    fn stack(&mut self, layer: impl Layer + 'static) {
        let add_layer = Box::new(layer);
        self.layers.push(add_layer);
    }

    fn cleargrad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.cleargrad();
        }
    }
    fn layers(&mut self) -> &mut Vec<Box<dyn Layer + 'static>> {
        &mut self.layers
    }

    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        let mut y = x.clone();

        for layer in self.layers.iter_mut() {
            let t = y;
            y = layer.call(&t);
        }

        y
    }
}

impl BaseModel {
    pub fn new() -> Self {
        BaseModel {
            input: None,
            output: None,
            layers: Vec::new(),
        }
    }

    pub fn call(&mut self, input: &RcVariable) -> RcVariable {
        // inputのvariableからdataを取り出す

        let output = self.forward(input);

        //　inputsを覚える
        self.input = Some(input.downgrade());

        //  outputを弱参照(downgrade)で覚える
        self.output = Some(output.downgrade());

        output
    }
}
