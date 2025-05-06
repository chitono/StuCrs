fn main() {
    let zahyo = Zahyo { x: 1.0, y: 2.0 };

    let x = Circle {
        r: 10.0,
        tyusin_zahyo: zahyo,
    };

    let x_cp = x.clone();
    x_cp.show();
}

#[derive(Debug, Clone)]
struct Circle {
    r: f32,
    tyusin_zahyo: Zahyo,
}

impl Circle {
    fn show(&self) {
        println!("{:?}", self);
    }
}

#[derive(Debug, Clone)]
struct Zahyo {
    x: f32,
    y: f32,
}
