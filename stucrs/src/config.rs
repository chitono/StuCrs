use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
//pub static GRAD_CONFIG: AtomicBool = AtomicBool::new(true);

pub fn id_generator() -> usize {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}
