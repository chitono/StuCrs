use std::rc::Rc;

fn main() {
    let mut soto = Rc::new(Variable {
        data: 10.0,
        grad: None,
        name: None,
    });

    println!("soto_count : {}", Rc::strong_count(&soto));
    let mut soto2 = soto.clone();
    println!("soto_count2 : {}", Rc::strong_count(&soto));
}

#[derive(Debug, Clone)]
struct Variable {
    data: f32,
    grad: Option<f32>,
    name: Option<String>,
}
