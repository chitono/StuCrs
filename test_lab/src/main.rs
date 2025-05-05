//use std::any::type_name;

fn main() {
    let mut x = Variable::new(0.5);
    println!("{:?}", x);

    let mut A = Square::new();
    let mut B = Exp::new();
    let mut C = Square::new();

    let mut a = A.call(&x);
    let mut b = B.call(&a);
    let mut y = C.call(&b);

    y.grad = Some(1.0);
    b.grad = Some(C.backward(y.grad.unwrap()));
    a.grad = Some(B.backward(b.grad.unwrap()));
    x.grad = Some(A.backward(a.grad.unwrap()));
    println!("{}", x.grad.unwrap());
}

#[derive(Debug)]
struct Variable {
    data: f32,
    grad: Option<f32>,
}

impl Variable {
    fn new(data: f32) -> Self {
        Variable { data, grad: None }
    }
}

trait Function<'a> {
    fn new() -> Self;

    fn call(&mut self, input: &'a Variable) -> Variable {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::new(y);
        self.set_input(input);
        output
    }

    fn forward(&self, x: f32) -> f32; // 引数f32
    fn backward(&self, gy: f32) -> f32; // 引数f32

    fn set_input(&mut self, input: &'a Variable) {}
}

struct Square<'a> {
    input: Option<&'a Variable>,
}

impl<'a> Function<'a> for Square<'a> {
    fn new() -> Self {
        Self { input: None }
    }
    fn forward(&self, x: f32) -> f32 {
        x.powf(2.0)
    }

    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.unwrap().data;
        2.0 * x * gy // = gx
    }

    fn set_input(&mut self, input: &'a Variable) {
        self.input = Some(input);
    }
}

#[allow(dead_code)]
struct Exp<'a> {
    input: Option<&'a Variable>,
}

impl<'a> Function<'a> for Exp<'a> {
    fn new() -> Self {
        Self { input: None }
    }

    fn forward(&self, x: f32) -> f32 {
        x.exp()
    }

    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.unwrap().data;
        x.exp() * gy // = gx
    }

    fn set_input(&mut self, input: &'a Variable) {
        self.input = Some(input);
    }
}

/*fn _numerical_diff<'a, T: Function<'a>>(f: &'a mut T, x: &'a Variable, eps: f32) -> f32 {
    let x0 = Variable::init(x.data - eps);
    let x1 = Variable::init(x.data + eps);

    let y0 = f.call(&x0);
    let y1 = f.call(&x1);

    (y1.data - y0.data) / (2.0 * eps)
} */
