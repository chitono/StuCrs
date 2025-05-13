use std::rc::Rc;

fn main() {
    let foo1 = Rc::new(Foo {
        value: 1,
        child: None,
    });
    let foo2 = Rc::new(Foo {
        value: 2,
        child: Some(Rc::clone(&foo1)),
    });
    let foo3 = Rc::new(Foo {
        value: 3,
        child: Some(Rc::clone(&foo2)),
    });
    println!("foo3: {:?}", foo3);
    println!("foo2: {:?}", foo2);
    println!("foo1: {:?}", foo1);
}

#[derive(Debug)]
struct Foo {
    value: i32,
    child: Option<Rc<Foo>>,
}
