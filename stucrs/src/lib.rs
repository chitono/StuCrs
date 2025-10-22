//use ndarray::{array, ArrayBase, Dimension, OwnedRepr};

use core_new::RcVariable;
use core_new::{add, div, mul, neg, sub};
use std::ops::{Add, Div, Mul, Neg, Sub};
//use std::sync::atomic::{AtomicU32, Ordering};

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
pub mod config;
pub mod dataloaders;
pub mod datasets;
pub mod functions_new;
pub mod layers;
pub mod models;
pub mod optimizers;
//pub mod functions_hdv;

#[cfg(test)]
mod tests {

    use ndarray::array;

    use crate::config::set_test_flag_true;

    use super::*;

    #[test]
    fn dropout_test() {

        use crate::{
        core_new::ArrayDToRcVariable,
        functions_new::{dropout},
        };
        // Create a 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    


        let a = array![1.0,1.0,1.0,1.0,1.0].rv();

        let b = dropout(&a, 0.5);

        // Sum along axis 0 (columns): should give [5, 7, 9] with shape [3]

        println!("b = {}", b.data());

    

        set_test_flag_true();

        let c = dropout(&a, 0.5);

        println!("c = {}", c.data());

    }
}
