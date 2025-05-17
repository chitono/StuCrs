use std::rc::Rc;

fn main() {
    let mut a = vec![1, 2, 3, 4, 5];
    let b = a.pop();
    println!("a={:?},b={:?}", a, b);
}
