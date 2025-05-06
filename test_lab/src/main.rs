use std::{clone, fmt::Debug, ops::Drop, process::Output};

fn main() {
    let mut x = Variable::new(2.0);
    x.name = Some("x".to_string());
    println!("x = {:?}", x);

    let mut A = Square::new();

    let mut y = A.call(&x);
    y.name = Some("y".to_string());

    println!("y = {:?}", y);

    println!("1");
    y.grad = Some(1.0);

    println!("2");
    x.grad = Some(A.backward(y.grad.unwrap()));

    println!("3");
    println!("x.grad= {:?}", x.grad);
}

#[derive(Debug, Clone)]
struct Variable<F: Function> {
    data: f32,
    grad: Option<f32>,
    creator: Option<F>,
    name: Option<String>,
}

impl<F: Function> Variable<F> {
    fn new(data: f32) -> Self {
        Variable {
            data,
            grad: None,
            creator: None,
            name: None,
        }
    }

    fn set_creator(&mut self, func: F) {
        self.creator = Some(func)
    }
}

impl<F: Function> Drop for Variable<F> {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

trait Function: Debug + Clone + 'static {
    type Fun_trait: Function;
    fn new() -> Self;

    fn call(&mut self, input: &Variable<Self::Fun_trait>) -> Variable<Self::Fun_trait>;

    fn forward(&mut self, x: f32) -> f32; // 引数f32
    fn backward(&mut self, gy: f32) -> f32; // 引数f32

    fn set_input(&mut self, input: &Variable<Self::Fun_trait>) {}
    fn set_output(&mut self, Output: &Variable<Self::Fun_trait>) {}
}

#[derive(Debug, Clone)]
struct Square<F: Function> {
    input: Option<Variable<F>>,
}

impl<F: Function> Function for Square<F> {
    fn new() -> Self {
        Self { input: None }
    }

    fn call(&mut self, input: &Variable) -> Variable {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::new(y);
        output.set_creator(self);
        self.set_input(input);
        self.set_output(&output);
        output
    }

    fn forward(&mut self, x: f32) -> f32 {
        x.powf(2.0)
    }

    fn backward(&mut self, gy: f32) -> f32 {
        let x = self.input.as_ref().unwrap().data;
        2.0 * x * gy // = gx
    }

    fn set_input(&mut self, input: &Variable) {
        self.input = Some(input.clone());
    }
}

impl Drop for Square {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

#[derive(Debug, Clone)]
struct Exp {
    input: Option<Variable>,
}

impl<F: Function> Function for Exp {
    fn new() -> Self {
        Self { input: None }
    }

    fn call(&mut self, input: &Variable<F>) -> Variable<F> {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::new(y);
        output.set_creator(self);
        self.set_input(input);
        self.set_output(&output);
        output
    }

    fn forward(&mut self, x: f32) -> f32 {
        x.exp()
    }

    fn backward(&mut self, gy: f32) -> f32 {
        let x = self.input.as_ref().unwrap().data;
        x.exp() * gy // = gx
    }

    fn set_input(&mut self, input: &Variable<F>) {
        self.input = Some(input.clone());
    }
}

impl Drop for Exp {
    fn drop(&mut self) {
        println!("Dropping : {:?}", self);
    }
}

/*#[derive(Debug, Clone)]
enum Functions {
    Square {},
    Exp {},
}
 */
/*fn _numerical_diff<'a, T: Function<'a>>(f: &'a mut T, x: &'a Variable, eps: f32) -> f32 {
    let x0 = Variable::init(x.data - eps);
    let x1 = Variable::init(x.data + eps);

    let y0 = f.call(&x0);
    let y1 = f.call(&x1);

    (y1.data - y0.data) / (2.0 * eps)
} */
