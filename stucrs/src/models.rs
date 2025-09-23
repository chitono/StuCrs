use crate::core_new::RcVariable;
use crate::core_new::Variable;
use crate::layers::Layer;
//use crate::optimizers::SGD;

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

//loss: Loss, optimizer: Optimizer, learning_rate: f32

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

    /*

    pub fn train(
        &mut self,
        train_data: ArrayViewD<f32>,
        test_data: ArrayViewD<f32>,
        batch_size: usize,
        epochs: i32,
    ) {
        let data_size = train_data.shape()[0];
        let mut optimizer = SGD::new(lr);
        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..train_data.shape()[0]).collect();
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
            let mut sum_loss = array![0.0f32];

            for chunk_indices in indices.chunks(batch_size) {
                let x_batch = train_data.select(Axis(0), chunk_indices).to_owned().rv();
                let t_batch = test_data.select(Axis(0), chunk_indices).to_owned().rv();

                //println!("x_batch = {:?}, t_batch = {:?}", x_batch, t_batch);

                let y = self.call(&x_batch);
                let mut loss = F::softmax_cross_entropy_simple(&y, &t_batch);
                self.cleargrad();
                loss.backward(false);
                optimizer.update();

                //ここでt_batch.lenはu32からf32に変換、さらに暗黙的にndarray型に変換されて、計算される。
                //また、sum_lossは静的次元なので、epoch_lossを動的次元から静的次元に変換して足せるようにする。

                let epoch_loss: Array1<f32> = (&loss.data() * (t_batch.len() as f32))
                    .into_dimensionality()
                    .unwrap();

                sum_loss = &sum_loss + &epoch_loss;
            }

            let average_loss = &sum_loss / (data_size as f32);

            println!("epoch = {:?}, loss = {:?}", epoch + 1, average_loss);
        }
    }*/
}

#[derive(Debug, Clone)]
pub enum Loss {
    MeanSquaredError,
    SoftmaxCrossEntropySimple,
}

#[derive(Debug, Clone)]
pub enum Optimizer {
    SGD,
}
