use ndarray::{array, Array};
use std::array;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::functions_new as F;
use stucrs::layers::{self as L, Dense, Layer, Linear};
use stucrs::models::{BaseModel, Model};

use std::f32::consts::PI;

use stucrs::core_new::ArrayDToRcVariable;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

fn f(x: &RcVariable) -> RcVariable {
    let y = x.pow(4.0) - 2.0.rv() * x.pow(2.0);
    y
}

fn main() {
    let x_data = Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();

    let y_data =
        x_data.mapv(|x| (2.0 * PI * x).sin()) + Array::random((100, 1), Uniform::new(0.0f32, 1.0));

    let x = x_data.rv();

    let y = y_data.rv();

    let l1 = L::Dense::new(10, false, None, L::Activation::Sigmoid);
    let l2 = L::Linear::new(10, false, None);

    let mut model = BaseModel::new();
    model.stack(l1);
    model.stack(l2);

    let start = Instant::now();
    //set_grad_false();
    let lr = 0.1;
    let iters = 10000;
    for i in 0..iters {
        let y_pred = model.call(&x);

        //println!("y_pred = {:?}\n", y_pred.clone().data());
        //println!("w2 = {:?}\n", w.clone());

        let mut loss = F::mean_squared_error(&y, &y_pred);

        model.cleargrad();
        loss.backward(false);

        for layer in model.layers() {
            for (_id, param) in layer.params() {
                let param_data = param.data();
                let current_grad = param.grad().as_ref().unwrap().data();
                param.0.borrow_mut().data = param_data - lr * current_grad;
            }
        }

        //loss.clear_grad_backward();

        //println!("w1 = {:?}\n", W1.clone().data());
        // y.backward();
        if i % 1000 == 0 {
            println!("i = {:?}", i);
            println!("loss = {:?}\n", loss.data());
        }
        //println!("w1 = {:?}\n", W1.clone().data());
        //println!("b1= {:?}\n", B1.clone().data());
        //println!("w2 = {:?}\n", W2.clone().data());
        //println!("b2 = {:?}\n", B2.clone().data());
        //println!("x1_grad = {:?}\n", x1.clone().grad());

        //println!("x2_grad={:?}\n", x2.grad())
    }

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}
