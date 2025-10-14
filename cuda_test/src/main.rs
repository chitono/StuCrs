use stucrs_gpu::core_new::RcVariable;
use stucrs_gpu::core_new::TensorToRcVariable;
use stucrs_gpu::functions_new::clamp;
use tensor_frame::{Shape, Tensor, TensorOps};

fn main() {
    let a = Tensor::from_vec(vec![1.0, -2.0, -3.0, 0.5, 0.6, 0.0], vec![3, 2]).unwrap();
    let b = a.rows_slice(&[0, 2]).unwrap();

    println!("a = {}, shape = {:?}", b, b.shape().dims());
}
