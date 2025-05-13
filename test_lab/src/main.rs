use std::cell::RefCell;

use std::fmt::{Debug, LowerExp};
use std::rc::{Rc, Weak};

fn main() {
    let x = Variable::new(2.0);
    x.borrow_mut().name = Some("x".to_string());
    println!("x = {:?}", x);

    let A = Square::new();
    let B = Exp::new();

    let a = A.borrow_mut().call(&x);
    let y = B.borrow_mut().call(&a);
    y.borrow_mut().name = Some("y".to_string());
    println!(
        "y.creator = {:?}",
        get_input(&y.borrow().creator.as_ref().unwrap())
            .borrow()
            .creator
    );
}

fn get_input(creator: &Rc<RefCell<Functions>>) -> Rc<RefCell<Variable>> {
    match &*creator.borrow() {
        Functions::Square(square) => square.input.as_ref().unwrap().clone(),
        Functions::Exp(exp) => exp.input.as_ref().unwrap().clone(),
    }
}

#[derive(Debug, Clone)]
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Rc<RefCell<Functions>>>,
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

    fn set_creator(&mut self, func: &Rc<RefCell<Functions>>) {
        self.creator = Some(Rc::clone(func));
    }
}
#[cfg(test)]
impl Drop for Variable {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

trait Function: Debug {
    fn new() -> Rc<RefCell<Self>>;
    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>>;
    fn forward(&self, x: f32) -> f32; // 引数f32
    fn backward(&self, gy: f32) -> f32; // 引数f32
}
#[derive(Debug, Clone)]
struct Square {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
}

impl Function for Square {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            input: None,
            output: None,
        }))
    }
    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let x = input.borrow().data;
        let y = self.forward(x);
        let output = Variable::new(y);
        self.input = Some(Rc::clone(input));
        self.output = Some(Rc::downgrade(&output));
        output
            .borrow_mut()
            .set_creator(&Rc::new(RefCell::new(Functions::Square(self.clone()))));
        output
    }

    fn forward(&self, x: f32) -> f32 {
        x.powf(2.0)
    }

    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.as_ref().unwrap().borrow().data;
        2.0 * x * gy // = gx
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
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Weak<RefCell<Variable>>>,
}

impl Function for Exp {
    fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            input: None,
            output: None,
        }))
    }

    fn call(&mut self, input: &Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let x = input.borrow().data;
        let y = self.forward(x);
        let output = Variable::new(y);
        self.input = Some(Rc::clone(input));
        self.output = Some(Rc::downgrade(&output));
        output
            .borrow_mut()
            .set_creator(&Rc::new(RefCell::new(Functions::Exp(self.clone()))));
        output
    }

    fn forward(&self, x: f32) -> f32 {
        x.exp()
    }

    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.as_ref().unwrap().borrow().data;
        x.exp() * gy // = gx
    }
}

#[cfg(test)]
impl Drop for Exp {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

#[derive(Debug)]
enum Functions {
    Square(Square),
    Exp(Exp),
}
/*
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
