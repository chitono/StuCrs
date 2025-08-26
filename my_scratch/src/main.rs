use ndarray::{array, Array};
use std::array;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::functions_new as F;
use stucrs::Layers::{self as L, Dense, Layer, Linear};

use std::f32::consts::PI;

use stucrs::core_new::ArrayDToRcVariable;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

fn f(x: &RcVariable) -> RcVariable {
    let y = x.pow(4.0) - 2.0.rv() * x.pow(2.0);
    y
}

fn main() {
    let mut x = array![1.0f32].rv();
    let mut y = F::sin(&x);
    y.backward(true);

    let start = Instant::now();
    let iters = 3;

    for _i in 0..iters {
        let opt_gx = x.grad();
        let mut gx = opt_gx.unwrap();
        x.cleargrad();

        gx.backward(true);
        println!("x = {:?}", x.grad().as_ref().unwrap().data());
    }
    //set_grad_false();

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}
