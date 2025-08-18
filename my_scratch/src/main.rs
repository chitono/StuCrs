use ndarray::array;
use std::time::Instant;
use stucrs::functions::sin;
use stucrs::ArrayDToRcVariable;
fn main() {
    let start = Instant::now();

    //set_grad_false();

    let iters = 1;
    for _i in 0..iters {
        let x = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].rv();
        //let x1 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        //let x2 = array![[11.0f32, 12.0, 13.0], [14.0, 15.0, 16.0]].rv();
        //let  shape_array = [1,6];

        // `&[usize; 2]`を`IxDyn`に変換
        //let dyn_shape = IxDyn(&shape_array);
        let mut y = sin(&x);

        y.backward();

        println!("y_data = {:?}\n", y.clone());

        println!("x_grad = {:?}\n", x.clone());

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
