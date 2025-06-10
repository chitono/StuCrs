use core::panic;
use std::cell::RefCell;

use std::fmt::Debug;
//use std::future;
//use std::process::Output;
use std::rc::{Rc, Weak};

//use std::thread;
//use std::time::Duration;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    /*
    let x1 = Variable::new(2.0);
    let x2 = Variable::new(3.0);

    let xs = [Some(x1.clone()), Some(x2.clone())];

    x.borrow_mut().name = Some("x".to_string());
    println!("x = {:?}", x);

    let xs = to_var(x.clone());

    */
    let x = [Some(Variable::new(3.0)), None];

    let y = add(&[x[0].clone(), x[0].clone()]);

    println!("y.data = {:?}", y[0].as_ref().unwrap().borrow().data);
    y[0].as_ref().unwrap().borrow_mut().backward();
    println!("y={:?}", y[0]);
    println!("x0={:?}", x[0]);

    /*
    println!("x1={:?}", x1.borrow());
    println!("x2={:?}", x2.borrow()); */

    //y.borrow_mut().backward();

    //println!("x = {:?}", x);

    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("処理時間{:?}", duration);
}

fn to_var(x: Rc<RefCell<Variable>>) -> [Option<Rc<RefCell<Variable>>>; 2] {
    let mut xs = [None, None];
    xs[0] = Some(x);
    xs
}

#[derive(Debug, Clone)]
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Rc<RefCell<dyn Function>>>,
    name: Option<String>,
}

impl Variable {
    fn new(data: f32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Variable {
            data,
            grad: None,
            creator: None,
            name: None,
        }))
    }

    fn set_creator(&mut self, func: Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::clone(&func));
    }

    fn backward(&self) {
        let mut funcs: Vec<Rc<RefCell<dyn Function>>> =
            vec![Rc::clone(self.creator.as_ref().unwrap())];

        //&selfで最初の変数はborrowされるので場合分け
        let mut last_variable = true;

        while let Some(f) = funcs.pop() {
            let xs = f.borrow().get_inputs();
            let mut y_grad: [Option<f32>; 2] = [None, None];

            if last_variable {
                y_grad = [Some(1.0), None];

                last_variable = false;
            } else {
                //関数の出力は一つだけなので、[1]は必要なし
                y_grad[0] = f.borrow().get_outputs()[0].as_ref().unwrap().borrow().grad;
            }

            let xs_grad = f.borrow().backward(y_grad);

            // gradを置き換えまたは足していくので、Noneか判別

            match xs[0].as_ref().unwrap().borrow().grad {
                Some(xs0_grad) => 
            }
            if let None = &xs[0].as_ref().unwrap().borrow().grad {
                xs[0].as_ref().unwrap().borrow_mut().grad = xs_grad[0];
            } else {

                //xs[0].as_ref().unwrap().borrow_mut().grad = xs_grad[0];
            }

            //xs[0]にcreatorがあるか確認
            if let Some(func_creator) = &xs[0].as_ref().unwrap().borrow().creator {
                funcs.push(Rc::clone(&func_creator));
            }

            //xs[1]はfが一変数関数の時、Noneなので確認が必要
            if let Some(xs_1) = xs[1].clone() {
                if let Some(mut xs1_grad) = xs_1.borrow().grad {
                    xs1_grad = xs1_grad.clone() + xs_grad[1].unwrap();
                } else {
                    xs_1.borrow_mut().grad = xs_grad[1];
                }

                //xs[0]にcreatorがあるか確認
                if let Some(func_creator) = &xs_1.borrow().creator {
                    funcs.push(Rc::clone(&func_creator));
                }
            }
        }
    }
}
#[cfg(test)]
impl Drop for Variable {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

trait Function: Debug {
    fn call(
        &mut self,
        input: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2];

    //  forward,backwardはVariableの数値のみを計算する
    fn forward(&self, x: [Option<f32>; 2]) -> [Option<f32>; 2];
    fn backward(&self, gy: [Option<f32>; 2]) -> [Option<f32>; 2];

    //　関数クラス.inputs, .outputsではvariableのbackwardの中でアクセスできないので、関数にして取得
    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2];
    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2];
}

#[derive(Debug, Clone)]
struct Square {
    //Square Class
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
}

impl Function for Square {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        if let None = &inputs[0] {
            panic!("Squareは一変数関数です。input[0]がNoneです")
        }
        if let Some(variable) = &inputs[1] {
            panic!("Squareは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_square: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2
        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_square.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];
        let y = xs[0].expect("数値がありません").powf(2.0);

        ys[0] = Some(y);

        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let x0_data = self.inputs[0].as_ref().unwrap().borrow().data;

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(2.0 * x0_data * gy0_data)
        }

        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }
}
impl Square {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
        }))
    }
}

#[cfg(test)]
impl Drop for Square {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

//fn square(input_x:&Rc<RefCell<Variable>>)

fn square(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    //xを一変数から配列に変換

    Square::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Exp {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
}

impl Function for Exp {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        if let None = &inputs[0] {
            panic!("Expは一変数関数です。input[0]がNoneです")
        }
        if let Some(variable) = &inputs[1] {
            panic!("Expは一変数関数です。input[1]がNoneではある必要があります")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new(ys_data[0].expect("数値が存在するはず")));

        //　inputsを覚える
        self.inputs = inputs.clone();

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_exp: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2
        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_exp.clone());

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];
        let y = xs[0].expect("数値がありません").exp();

        ys[0] = Some(y);

        /*
        if let Some(x1_data) = xs[1] {
            ys[1] = Some(x1_data.powf(2.0))
        } */

        /*
        for (i, x) in xs.iter().enumerate() {
            if let Some(x_data) = x {
                ys[i] = Some(x_data.powf(2.0));
            }
        */
        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let x0_data = self.inputs[0].as_ref().unwrap().borrow().data;

        if let Some(gy0_data) = gys[0] {
            gxs[0] = Some(x0_data.exp() * gy0_data)
        }

        /*
        if let Some(x1_data) = xs[1] {
            ys[1] = Some(x1_data.powf(2.0))
        } */

        /*
        for (i, x) in xs.iter().enumerate() {
            if let Some(x_data) = x {
                ys[i] = Some(x_data.powf(2.0));
            }
        */
        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();

        outputs
    }
}
impl Exp {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
        }))
    }
}

#[cfg(test)]
impl Drop for Exp {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

fn exp(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    //xを一変数から配列に変換

    Exp::new().borrow_mut().call(xs)
}

#[derive(Debug, Clone)]
struct Add {
    //Add Class
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
}

impl Function for Add {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        //ここで渡された値の配列がどっちも数値を持っているか確認
        if let None = &inputs[0] {
            panic!("Addは二変数関数です。input[0]がNoneです。")
        }

        if let None = &inputs[1] {
            panic!("Addは二変数関数です。input[1]がNoneです。")
        }

        let mut xs_data = [None, None];

        // inputのvariableからdataを取り出す
        xs_data[0] = Some(
            inputs[0]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );
        xs_data[1] = Some(
            inputs[1]
                .as_ref()
                .expect("数値が存在するはず")
                .borrow()
                .data,
        );

        /*
        if let Some(var) = &inputs[0] {
            xs_data[0] = Some(var.borrow().data)
        }
        if let Some(var) = &inputs[1] {
            xs_data[1] = Some(var.borrow().data)
        } */

        //配列だけど一つの値のみ
        let ys_data = self.forward(xs_data);

        let mut outputs = [None, None];

        //ys_dataは一変数なので、outputs[1]は必要なし
        outputs[0] = Some(Variable::new(ys_data[0].expect("数値が存在するはず")));

        /*  ys_dataは必ず二つとも数値を持っているので、if letにしなくてもよい
        if let Some(var) = ys_data[0] {
            outputs[0] = Some(Variable::new(var))
        }

        if let Some(var) = ys_data[1] {
            outputs[1] = Some(Variable::new(var))
        } */

        //　inputsを覚える
        self.inputs = inputs.clone();

        //  outputsを弱参照(downgrade)で覚える
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_add: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        //outputsに自分をcreatorとして覚えさせる　不変長　配列2

        outputs[0]
            .as_ref()
            .expect("outputs[0]がNoneになってる")
            .borrow_mut()
            .set_creator(self_add.clone());

        /* 　上のコードの可変長版
        for output_opt in &outputs {
            if let Some(output_rc) = output_opt {
                output_rc.borrow_mut().set_creator(&self_square);
            }
        } */

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];

        // y = xs[0] + xs[1]
        let y = xs[0].expect("数値が存在するはず in forward")
            + xs[1].expect("数値が存在するはず in forward");

        ys[0] = Some(y);

        /*
        if let Some(x1_data) = xs[1] {
            ys[1] = Some(x1_data.powf(2.0))
        } */

        /*
        for (i, x) in xs.iter().enumerate() {
            if let Some(x_data) = x {
                ys[i] = Some(x_data.powf(2.0));
            }
        */
        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let gy = gys[0].clone();
        let gxs = [gy, gy];

        /*
        if let Some(x1_data) = xs[1] {
            ys[1] = Some(x1_data.powf(2.0))
        } */

        /*
        for (i, x) in xs.iter().enumerate() {
            if let Some(x_data) = x {
                ys[i] = Some(x_data.powf(2.0));
            }
        */
        gxs
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        self.inputs.clone()
    }

    fn get_outputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut outputs = [None, None];
        outputs[0] = self.outputs[0].as_ref().unwrap().upgrade().clone();
        outputs[1] = self.outputs[1].as_ref().unwrap().upgrade().clone();

        outputs
    }
}
impl Add {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            inputs: [None, None],
            outputs: [None, None],
        }))
    }
}

#[cfg(test)]
impl Drop for Add {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

fn add(xs: &[Option<Rc<RefCell<Variable>>>; 2]) -> [Option<Rc<RefCell<Variable>>>; 2] {
    Add::new().borrow_mut().call(xs)
}

/*
#[derive(Debug, Clone)]
struct Exp {
    input: Rc<RefCell<Variable>>,
    output: Weak<RefCell<Variable>>,
}






impl Function for Exp {
    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let x = input.borrow().data;
        let y = self.forward(x);


        let output = Variable::new(y);


        self.input = Rc::clone(input);
        self.output = Rc::downgrade(&output);


        let self_exp: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));
        output.borrow_mut().set_creator(&self_exp);


        output
    }


    fn forward(&self, x: f32) -> f32 {
        x.exp()
    }


    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.borrow().data;
        x.exp() * gy // = gx
    }


    fn get_input(&self) -> Rc<RefCell<Variable>> {
        self.input.clone()
    }


    fn get_output(&self) -> Rc<RefCell<Variable>> {
        self.output.upgrade().as_ref().unwrap().clone()
    }
}


impl Exp {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            input: Variable::new(0.0),
            output: Rc::downgrade(&Variable::new(0.0)),
        }))
    }
}
#[cfg(test)]
impl Drop for Exp {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}


fn exp(x: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    Exp::new().borrow_mut().call(x)
} */

/*      input[1]はNoneなので必要なし
if let Some(var) = &inputs[1] {
    xs_data[1] = Some(var.borrow().data)
} */

/* 　可変長版
for (i, input) in inputs.iter().enumerate() {
    if let Some(var) = input {
        xs[i] = Some(var.borrow().data);
    }
}*/

/*
for (i, y) in ys.iter().enumerate() {
    if let Some(data) = y {
        outputs[i] = Some(Variable::new(*data));
    }
} */

/*
if let Some(var) = ys_data[1] {
    outputs[1] = Some(Variable::new(var))
} */
