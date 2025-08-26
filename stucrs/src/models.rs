use crate::core_new::{RcVariable, Variable};
use crate::layers::{self, Layer};
use fxhash::FxHashMap;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub trait Model {
    fn stack(&mut self, layer: impl Layer + 'static);
    fn forward(&mut self, x: &RcVariable) -> RcVariable;
    fn layers(&self) -> Rc<RefCell<Vec<Box<dyn Layer + 'static>>>>;
    fn layers_mut(&mut self) -> Rc<RefCell<Vec<Box<dyn Layer + 'static>>>>;
    fn cleargrad(&mut self);
}

#[derive(Debug, Clone)]
pub struct BaseModel {
    input: Option<Weak<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
    layers: Rc<RefCell<Vec<Box<dyn Layer + 'static>>>>,
}

impl Model for BaseModel {
    fn stack(&mut self, layer: impl Layer + 'static) {
        let add_layer = Box::new(layer);
        self.layers.borrow_mut().push(add_layer);
    }

    fn cleargrad(&mut self) {
        for layer in self.layers.borrow_mut().iter_mut() {
            layer.cleargrad();
        }
    }
    fn layers(&self) -> Rc<RefCell<Vec<Box<dyn Layer + 'static>>>> {
        self.layers.clone()
    }

    fn layers_mut(&mut self) -> Rc<RefCell<Vec<Box<dyn Layer + 'static>>>> {
        self.layers.clone()
    }

    fn forward(&mut self, x: &RcVariable) -> RcVariable {
        let mut y = x.clone();

        for layer in self.layers.borrow_mut().iter_mut() {
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
            layers: Rc::new(RefCell::new(Vec::new())),
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
