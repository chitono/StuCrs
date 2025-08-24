use ndarray::{array, Array, IxDyn};
use std::array;
use std::time::Instant;
use stucrs::core_new::{F32ToRcVariable, RcVariable};
use stucrs::functions_new::{linear_simple, matmul, reshape, sigmoid_simple, sum_to};

use std::f32::consts::PI;

use stucrs::core_new::ArrayDToRcVariable;
use stucrs::functions_new::mean_squared_error;

use ndarray_rand::rand_distr::{StandardNormal, Uniform};
use ndarray_rand::RandomExt;

fn main() {
    let x_data = Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();

    let y_data =
        x_data.mapv(|x| (2.0 * PI * x).sin()) + Array::random((100, 1), Uniform::new(0.0f32, 1.0));

    let x = x_data.rv();

    let y = y_data.rv();

    let I = 1;
    let H = 10;
    let O = 1;

    let mut W1 = (0.01f32 * Array::random((I, H), StandardNormal)).rv();

    let mut B1 = Array::zeros(H).rv();

    let mut W2 = (0.01f32 * Array::random((H, O), StandardNormal)).rv();

    let mut B2 = Array::zeros(O).rv();

    fn predict(
        x: &RcVariable,
        w1: &RcVariable,
        b1: &Option<RcVariable>,
        w2: &RcVariable,
        b2: &Option<RcVariable>,
    ) -> RcVariable {
        let t0 = linear_simple(&x, &w1, &b1);

        //println!("t0 = {:?}\n", t0.clone().data());
        let t1 = sigmoid_simple(&t0);
        //println!("t1 = {:?}\n", t1.clone().data());
        let y = linear_simple(&t1, &w2, &b2);

        y
    }

    //let y_data = 2.0*x_data.clone() +5.0 + Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    //println!("y_data = {:?}",y_data.clone());

    //let x=x_data.rv();

    //let y=y_data.rv();

    //let mut w = Array::zeros((1,1)).rv();
    //let mut b = Array::zeros(1).rv();

    let start = Instant::now();
    //set_grad_false();
    let lr = 0.2;
    let iters = 10000;
    for i in 0..iters {
        let y_pred = predict(&x, &W1, &Some(B1.clone()), &W2, &Some(B2.clone()));

        //println!("y_pred = {:?}\n", y_pred.clone().data());
        //println!("w2 = {:?}\n", w.clone());

        let mut loss = mean_squared_error(&y, &y_pred);

        //println!("loss = {:?}\n", loss.0.borrow().creator);

        W1.cleargrad();
        W2.cleargrad();
        B1.cleargrad();
        B2.cleargrad();

        loss.backward(false);
        //loss.clear_grad_backward();

        //let w_data=lr*w.grad();
        //let b_data=lr*b.grad();

        let w1_data = W1.data();
        //println!("w1_data = {:?}\n", w1_data.clone());
        let b1_data = B1.data();
        let w2_data = W2.data();
        let b2_data = B2.data();
        //let w.data=w_data-lr*w.grad();

        let current_grad_w1 = W1.grad().unwrap().data();
        let current_grad_b1 = B1.grad().unwrap().data();
        let current_grad_w2 = W2.grad().unwrap().data();
        let current_grad_b2 = B2.grad().unwrap().data();

        // w =  (w_data - lr*current_grad_w).rv();
        // b = (b_data - lr*current_grad_b).rv();

        W1.0.borrow_mut().data = w1_data - lr * current_grad_w1;
        B1.0.borrow_mut().data = b1_data - lr * current_grad_b1;
        W2.0.borrow_mut().data = w2_data - lr * current_grad_w2;
        B2.0.borrow_mut().data = b2_data - lr * current_grad_b2;
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
    /*

    let lr = 0.001;
    let iters = 1000;






    for i in 0..iters {



        println!("{:?}, {:?}",x0.data() ,x1.data());


        let mut y = rosenbrock(&x0, &x1);


        x0.cleargrad();
        x1.cleargrad();
        y.backward();




        let current_data_0 = x0.data();
        let current_data_1 = x1.data();

        let current_grad_0 =x0.grad().unwrap();
        let current_grad_1 =x1.grad().unwrap();

        x0.0.borrow_mut().data =current_data_0- lr*current_grad_0;
        x1.0.borrow_mut().data = current_data_1- lr*current_grad_1;




    }*/
    //println!("(x0,x1)=({:?},{:?})", x0.0.borrow().data,x1.0.borrow().data);

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}
