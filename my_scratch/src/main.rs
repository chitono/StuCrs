use ndarray::{array,Array, Data};
use stucrs::core::RcVariable;
use stucrs::functions::sum;
use std::array;
use std::time::Instant;

use stucrs::functions::matmul;
use stucrs::{ArrayDToRcVariable, F32ToRcVariable};

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn predict(x:&RcVariable,w:&RcVariable,b:&RcVariable) -> RcVariable {
        let y = matmul(&x, &w) + b.clone();
        y
    }

fn mean_squared_error(x0:&RcVariable,x1:&RcVariable) ->RcVariable{
    let diff =x0.clone()-x1.clone();
    let len = diff.len() as f32;
    println!("len = {:?}",len);
    
    let error = sum(&diff.pow(2.0), None) /len.rv();
    
    error


}

fn main() {
    let start = Instant::now();
    
    let x_data = Array::random((100, 1), Uniform::new(0.0f32, 1.0));
        //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        
        
    let y_data = 2.0*x_data.clone() +5.0 + Array::random((100, 1), Uniform::new(0.0f32, 1.0));
    println!("y_data = {:?}",y_data.clone());

    let x=x_data.rv();

    let y=y_data.rv();
    
    let mut w = Array::zeros((1,1)).rv();
    let mut b = Array::zeros(1).rv();
    

    
    //set_grad_false();
    let lr =0.1;
    let iters = 10000;
    for _i in 0..iters {
        //println!("w1 = {:?}\n", w.clone());

       let y_pred=predict(&x, &w, &b);
       //println!("y_pred = {:?}\n",y_pred.clone().data().clone());
        //println!("w2 = {:?}\n", w.clone());

       let mut loss=mean_squared_error(&y, &y_pred);
        //println!("loss = {:?}\n", loss.0.borrow().creator);

       w.cleargrad();
       b.cleargrad();
       
       loss.backward();
       //println!("wwwwwwwwwwwwwwwwwwwwwwwwwwwww\n");

        

       //let w_data=lr*w.grad();
       //let b_data=lr*b.grad();


       let w_data=w.data().clone();
       let b_data=b.data().clone();

       //let w.data=w_data-lr*w.grad(); 

  

        let current_grad_w =w.grad().unwrap();
        let current_grad_b =b.grad().unwrap();

       // w =  (w_data - lr*current_grad_w).rv();
       // b = (b_data - lr*current_grad_b).rv();

        w.0.borrow_mut().data =w_data- lr*current_grad_w;
        b.0.borrow_mut().data = b_data- lr*current_grad_b;

    

    // y.backward();

        println!("w = {:?}\n", w.clone().data());

        println!("b= {:?}\n", b.clone().data());
        //println!("loss = {:?}\n", loss.clone());
        //println!("x1_grad = {:?}\n", x1.clone().grad());

        //println!("x2_grad={:?}\n", x2.grad());
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
    println!("処理時間{:?}", duration / iters);
}
