use std::cell::RefCell;

use std::fmt::{Debug, LowerExp};
use std::future;
use std::process::Output;
use std::rc::{Rc, Weak};

fn main() {
    let x = Variable::new(0.5);
    x.borrow_mut().name = Some("x".to_string());
    println!("x = {:?}", x);

    let y = square(&exp(&square(&x)));

    println!("y.data = {:?}", y.borrow().data);

    y.borrow_mut().backward();

    println!("x = {:?}", x);
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

    fn set_creator(&mut self, func: &Rc<RefCell<dyn Function>>) {
        self.creator = Some(Rc::clone(func));
    }

    fn backward(&self) {
        let mut funcs: Vec<Rc<RefCell<dyn Function>>> =
            vec![Rc::clone(self.creator.as_ref().unwrap())];

        let mut last_variable = true;
        while let Some(f) = funcs.pop() {
            let x = f.borrow().get_input();

            if last_variable {
                let y_grad: f32 = 1.0;
                x.borrow_mut().grad = Some(f.borrow().backward(y_grad));
                last_variable = false;
            } else {
                let y = f.borrow().get_output();
                let y_grad = *y.borrow().grad.as_ref().unwrap();
                x.borrow_mut().grad = Some(f.borrow().backward(y_grad));
            }

            if x.borrow().creator.is_none() {
                break;
            };

            funcs.push(Rc::clone(x.borrow().creator.as_ref().unwrap()));
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
    fn forward(&self, x: [Option<f32>; 2]) -> [Option<f32>; 2]; // 引数f32
    fn backward(&self, gy: [Option<f32>; 2]) -> [Option<f32>; 2]; // 引数f32
    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2];
    fn get_outputs(&self) -> [Option<Weak<RefCell<Variable>>>; 2];
}

#[derive(Debug, Clone)]
struct Square {
    inputs: [Option<Rc<RefCell<Variable>>>; 2],
    outputs: [Option<Weak<RefCell<Variable>>>; 2],
}

impl Function for Square {
    fn call(
        &mut self,
        inputs: &[Option<Rc<RefCell<Variable>>>; 2],
    ) -> [Option<Rc<RefCell<Variable>>>; 2] {
        let mut xs = [None, None];

        for (i, input) in inputs.iter().enumerate() {
            if let Some(var) = input {
                xs[i] = Some(var.borrow().data);
            }
        }

        let ys = self.forward(xs);

        let mut outputs = [None, None];

        for (i, y) in ys.iter().enumerate() {
            if let Some(data) = y {
                outputs[i] = Some(Variable::new(*data));
            }
        }

        /*
        if let Some(var) = ys[0] {
            outputs[0] = Some(Variable::new(*ys[0].as_ref().unwrap()))
        }
        if let Some(var) = ys[1] {
            outputs[1] = Some(Variable::new(*ys[1].as_ref().unwrap()))
        }
        */
        self.inputs = inputs.clone();
        self.outputs = [
            outputs[0].as_ref().map(|rc_0| Rc::downgrade(rc_0)),
            outputs[1].as_ref().map(|rc_1| Rc::downgrade(rc_1)),
        ];

        let self_square: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));

        for output_opt in &outputs {
            if let Some(output_rc) = output_opt {
                output_rc.borrow_mut().set_creator(&self_square);
            }
        }

        outputs
    }

    fn forward(&self, xs: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut ys = [None, None];
        for (i, x) in xs.iter().enumerate() {
            if let Some(x_data) = x {
                ys[i] = Some(x_data.powf(2.0));
            }
        }
        ys
    }

    fn backward(&self, gys: [Option<f32>; 2]) -> [Option<f32>; 2] {
        let mut gxs = [None, None];

        let xs = [self.inputs[0].borrow().data, self.inputs];
        2.0 * x * gy // = gx
    }

    fn get_inputs(&self) -> [Option<Rc<RefCell<Variable>>>; 2] {
        Rc::clone(&self.input)
    }

    fn get_outputs(&self) -> [Option<Weak<RefCell<Variable>>>; 2] {
        Rc::clone(self.output.upgrade().as_ref().unwrap())
    }
}
impl Square {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            input: Variable::new(0.0),
            output: Rc::downgrade(&Variable::new(0.0)),
        }))
    }
}

#[cfg(test)]
impl Drop for Square {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

fn square(x: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    Square::new().borrow_mut().call(x)
}

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
}
