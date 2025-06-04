//use std::rc::Rc;

fn main() {
    let a = [Some(5.0), Some(1.0)];
    if let Some(_variable) = &a[1] {
        panic!("Squareは一変数関数です。input[1]がNoneではありません")
    } else {
        println!("ok");
    }
}
