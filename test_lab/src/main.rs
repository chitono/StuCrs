//use std::any::type_name;

fn main() {
    let x = Variable::init(0.5);

    let a_f = Square {};
    let b_f = Exp {};
    let c_f = Square {};

    let a = a_f.call(&x);
    let b = b_f.call(&a);
    let y = c_f.call(&b);
    println!("{}", y.data);
}

struct Variable {
    data: f32,
}

impl Variable {
    fn init(data: f32) -> Self {
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

    fn forward(&self, x: f32) -> f32;
}

struct Square {}

impl Function for Square {
    fn forward(&self, x: f32) -> f32 {
        x.powf(2.0)
    }
}

struct Exp {}

impl Function for Exp {
    fn forward(&self, x: f32) -> f32 {
        x.exp()
    }
}
