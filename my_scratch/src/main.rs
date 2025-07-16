use once_cell::sync::Lazy; // 1.3.1
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static GRAD_CONFIG: Mutex<bool> = Mutex::new(true);

fn set_grad_true() {
    let mut flag = GRAD_CONFIG.lock().unwrap();
    *flag = true;
}

fn set_grad_false() {
    let mut flag = GRAD_CONFIG.lock().unwrap();
    *flag = false;
}

fn get_grad_status() -> bool {
    let flag = GRAD_CONFIG.lock().unwrap();
    *flag
}

fn main() {
    println!("{}", get_grad_status());
    set_grad_false();
    println!("{}", get_grad_status());
    set_grad_true();
    println!("{}", get_grad_status());
}
