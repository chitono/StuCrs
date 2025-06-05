//use std::rc::Rc;

fn main() {
    let mut soto = 1.0;
    let flag = false;

    if flag == true {
        soto = 10.0;
    } else {
        soto = 20.0;
    }

    println!("{}", soto);
}
