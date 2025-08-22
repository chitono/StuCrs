use ndarray::{array, ArrayBase, Dimension, OwnedRepr};

use core::RcVariable;
use core::{add, div, mul, neg, sub};
use std::ops::{Add, Div, Mul, Neg, Sub};

//演算子のオーバーロード

impl Add for RcVariable {
    type Output = RcVariable;
    fn add(self, rhs: RcVariable) -> Self::Output {
        // add_op関数はRc<RefCell<Variable>>を扱う
        let add_y = add(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(add_y.clone())
    }
}



impl Mul for RcVariable {
    type Output = RcVariable;
    fn mul(self, rhs: RcVariable) -> Self::Output {
        let mul_y = mul(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(mul_y.clone())
    }
}


impl Sub for RcVariable {
    type Output = RcVariable;
    fn sub(self, rhs: RcVariable) -> Self::Output {
        let sub_y = sub(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(sub_y.clone())
    }
}

impl Div for RcVariable {
    type Output = RcVariable;
    fn div(self, rhs: RcVariable) -> Self::Output {
        let div_y = div(&[Some(self.0.clone()), Some(rhs.0.clone())]);
        RcVariable(div_y.clone())
    }
}

impl Neg for RcVariable {
    type Output = RcVariable;
    fn neg(self) -> Self::Output {
        let neg_y = neg(&[Some(self.0.clone()), None]);
        RcVariable(neg_y.clone())
    }
}

//array型からRcVariable型を生成
pub trait ArrayDToRcVariable {
    fn rv(&self) -> RcVariable;
}
//arrayは任意の次元に対応
impl<D: Dimension> ArrayDToRcVariable for ArrayBase<OwnedRepr<f32>, D> {
    fn rv(&self) -> RcVariable {
        RcVariable::new(self.view().into_dyn())
    }
}

pub trait F32ToRcVariable {
    fn rv(&self) -> RcVariable;
}

//rustの数値のデフォルトがf64なので、f32に変換する
//f32からarray型に変換し、rv()でRcVariableを生成
impl F32ToRcVariable for f32 {
    fn rv(&self) -> RcVariable {
        let array = array![*self];
        array.rv()
    }
}

pub mod core;
//pub mod core_hdv;
pub mod functions;
//pub mod functions_hdv;
pub use core::set_grad_false;
pub use core::set_grad_true;

#[cfg(test)]
mod tests {
    //use super::*;

    #[test]
    fn it_works() {}
}
