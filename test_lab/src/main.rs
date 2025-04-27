//use std::any::type_name;

fn main() {
    let x = Variable::init(2.0);
    println!("{}", x.data);
    let f = Square {};
    let y = f.call(&x);
    println!("{}", y.data);
    println!("{}", x.data);
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
