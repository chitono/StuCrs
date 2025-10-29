use std::cell::RefCell;
//use std::clone;
use std::collections::HashSet;
use std::fmt::Debug;

use ndarray::{array, ArrayBase, ArrayD, ArrayViewD, Dimension, IxDyn, OwnedRepr};
use std::rc::{Rc, Weak};
use std::{usize, vec};

use crate::config::{get_grad_status, id_generator, set_grad_false, set_grad_true};
use crate::functions_new::*;

pub fn get_conv_outsize(
    input_size: (usize, usize),
    kernel_size: (usize, usize),
    stride_size: (usize, usize),
    pad_size: (usize, usize),
) -> (usize, usize) {
    let oh = (input_size.0 + pad_size.0 * 2 - kernel_size.0) / stride_size.0 + 1;
    let ow = (input_size.1 + pad_size.1 * 2 - kernel_size.1) / stride_size.1 + 1;

    (oh, ow)
}
