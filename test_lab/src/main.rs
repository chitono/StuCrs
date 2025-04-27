//use std::any::type_name;

fn main() {
    let x = Variable::init(2.0);
    let eps = 0.0001;
    let f = Square {};

    let y = f.call(&x);

    let dy = numerical_diff(&f, &x, eps);

    println!("{}", dy);
}

struct Variable {
    data: f64,
}

impl Variable {
    fn init(data: f64) -> Self {
        Variable { data }
    }
}

trait Function {
    fn call(&self, input: &Variable) -> Variable {
        let x = input.data;
        let y = self.forward(x);
        let output = Variable::init(y);
        output
    }

    fn forward(&self, x: f64) -> f64;
}

struct Square {}

impl Function for Square {
    fn forward(&self, x: f64) -> f64 {
        x.powf(2.0)
    }
}

struct Exp {}

impl Function for Exp {
    fn forward(&self, x: f64) -> f64 {
        x.exp()
    }
}

fn numerical_diff<T: Function>(f: &T, x: &Variable, eps: f64) -> f64 {
    let x0 = Variable::init(x.data - eps);
    let x1 = Variable::init(x.data + eps);

    let y0 = f.call(&x0);
    let y1 = f.call(&x1);

    (y1.data - y0.data) / (2.0 * eps)
}
