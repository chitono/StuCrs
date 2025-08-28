use ndarray::{array, s, Array};
use std::array;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::functions_new as F;
use stucrs::layers::{self as L, Activation, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};
use stucrs::optimizers::{Optimizer, SGD};

use std::f32::consts::PI;

use stucrs::core_new::ArrayDToRcVariable;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

fn main() {
    let x = array![[0.2f32, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]].rv();
    let t = array![
        [0.0f32, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ]
    .rv();

    let mut model = BaseModel::new();
    model.stack(L::Dense::new(10, false, None, Activation::Sigmoid));
    model.stack(L::Dense::new(3, false, None, Activation::Sigmoid));

    let y = model.call(&x);

    let loss = F::softmax_cross_entropy_simple(&y, &t);

    println!("loss = {:?}", loss.clone().data());
}
