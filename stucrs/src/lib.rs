//use ndarray::{array, ArrayBase, Dimension, OwnedRepr};

use core_new::RcVariable;
use core_new::{add, div, mul, neg, sub};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU32, Ordering};

//演算子のオーバーロード

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[Some(self.clone()), Some(rhs.clone())]);
        add_y
    }
}

impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[Some(self.clone()), Some(rhs.clone())]);
        mul_y
    }
}

impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[Some(self.clone()), Some(rhs.clone())]);
        sub_y
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[Some(self.clone()), Some(rhs.clone())]);
        div_y
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[Some(self.clone()), None]);
        neg_y
    }
}

pub mod core_new;
//pub mod core_hdv;
pub mod Layers;
pub mod functions_new;
pub mod models;
pub mod optimizers;
//pub mod functions_hdv;
pub use core_new::set_grad_false;
pub use core_new::set_grad_true;

#[cfg(test)]
mod tests {
    use crate::{
        core_new::F32ToRcVariable,
        functions_new::{sin, tanh},
    };

    use super::*;

    #[test]
    fn it_works() {
        let x = 1.0f32.rv();
        let mut y = tanh(&x);
        y.backward(false);
        println!("x={:?}", x.clone());
    }
}
