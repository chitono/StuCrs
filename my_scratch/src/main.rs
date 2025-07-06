use std::ops::Add;

fn main() {
    let iti = Variable { data: 1.0 };
    let ni = Variable { data: 2.0 };
    println!("{}", iti + ni);
}

#[derive(Debug, Clone)]
struct Variable {
    data: f32,
}

impl Add for Variable {
    type Output = f32;
    fn add(self, rhs: Self) -> Self::Output {
        self.data + rhs.data
    }
}
