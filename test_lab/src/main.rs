use std::cell::RefCell;

use std::fmt::{Debug, LowerExp};
use std::future;
use std::rc::{Rc, Weak};

fn main() {
    let x = Variable::new(0.5);
    x.borrow_mut().name = Some("x".to_string());
    println!("x = {:?}", x);

    let A = Square::new();
    let B = Exp::new();
    let C = Square::new();

    let a = A.borrow_mut().call(&x);
    let b = B.borrow_mut().call(&a);
    let y = C.borrow_mut().call(&b);

    println!("y.data = {:?}", y.borrow().data);

    y.borrow_mut().grad = Some(1.0);
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
                let y_grad = *self.grad.as_ref().unwrap();
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
    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>>;
    fn forward(&self, x: f32) -> f32; // 引数f32
    fn backward(&self, gy: f32) -> f32; // 引数f32
    fn get_input(&self) -> Rc<RefCell<Variable>>;
    fn get_output(&self) -> Rc<RefCell<Variable>>;
}
#[derive(Debug, Clone)]
struct Square {
    input: Rc<RefCell<Variable>>,
    output: Weak<RefCell<Variable>>,
}

impl Function for Square {
    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let x = input.borrow().data;
        let y = self.forward(x);

        let output = Variable::new(y);

        self.input = Rc::clone(input);
        self.output = Rc::downgrade(&output);

        let self_square: Rc<RefCell<dyn Function>> = Rc::new(RefCell::new(self.clone()));
        output.borrow_mut().set_creator(&self_square);

        output
    }

    fn forward(&self, x: f32) -> f32 {
        x.powf(2.0)
    }

    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.borrow().data;
        2.0 * x * gy // = gx
    }

    fn get_input(&self) -> Rc<RefCell<Variable>> {
        Rc::clone(&self.input)
    }

    fn get_output(&self) -> Rc<RefCell<Variable>> {
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
/*
#[derive(Debug)]
enum Functions {
    Square(Square),
    Exp(Exp),
}

impl Function for Functions {
    fn new() -> Rc<RefCell<Self>> {
        unimplemented!("new() is not directly applicable to Functions enum");
    }

    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        match self {
            Functions::Square(square) => square.call(input),
            Functions::Exp(exp) => exp.call(input),
        }
    }

    fn forward(&self, x: f32) -> f32 {
        match self {
            Functions::Square(square) => square.forward(x),
            Functions::Exp(exp) => exp.forward(x),
        }
    }

    fn backward(&self, gy: f32) -> f32 {
        match self {
            Functions::Square(square) => square.forward(gy),
            Functions::Exp(exp) => exp.forward(gy),
        }
    }
}
*/
/*fn _numerical_diff<'a, T: Function<'a>>(f: &'a mut T, x: &'a Variable, eps: f32) -> f32 {
    let x0 = Variable::init(x.data - eps);
    let x1 = Variable::init(x.data + eps);

    let y0 = f.call(&x0);
    let y1 = f.call(&x1);

    (y1.data - y0.data) / (2.0 * eps)
}



if let Some(creator) = &y.borrow().creator {
        match &*creator.borrow() {
            Functions::Square(sq) => {
                println!("y.creator.input={:?}", sq.input);
            }
            Functions::Exp(ex) => {
                println!("y.creator.input={:?}", ex.input);
            }
        }
    };


fn cap_input(creator: &Rc<RefCell<Functions>>) -> Option<Rc<RefCell<Variable>>> {
    let captured_input = match &*creator.borrow() {
        Functions::Square(sqare) => sqare.input,
        Functions::Exp(exp) => exp.input,
    };
    captured_input
}
 */
