fn main() {
    let mut a = Variable::init(1.0);
    println!("{}", a.data);
    a.data = 5.0;
    println!("{}", a.data);
}

struct Variable {
    data: f32,
}

impl Variable {
    fn init(data: f32) -> Self {
        Variable { data }
    }
}
