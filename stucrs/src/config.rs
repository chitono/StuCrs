use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
/// Variableや関数たちにidを付けるための値
pub static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// 微分するかしないかというフラグ
/// 推論するときなど、微分する必要がないときに切り替える
pub static GRAD_CONFIG: AtomicBool = AtomicBool::new(true);

/// バックエンドを示すもの
/// 今後複数のバックエンドに対応するために数字で判別
///
/// 0 --> "CPU"
///
/// 1 --> "CUDA"
///
/// その他 --> 未対応
///
pub static BACKEND: AtomicU8 = AtomicU8::new(0);

pub static TEST_FLAG: AtomicBool = AtomicBool::new(false);
/// idを生成する関数。構造体のコンストラクタを作成する際に、呼び出して、idを付ける
pub fn id_generator() -> usize {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

pub fn set_grad_true() {
    GRAD_CONFIG.store(true, Ordering::SeqCst);
}

pub fn set_grad_false() {
    GRAD_CONFIG.store(false, Ordering::SeqCst);
}

pub fn get_grad_status() -> bool {
    GRAD_CONFIG.load(Ordering::SeqCst)
}

/// test_flagをtrueに変更する関数。
pub fn set_test_flag_true() {
    TEST_FLAG.store(true, Ordering::SeqCst);
}

pub fn set_test_flag_false() {
    TEST_FLAG.store(false, Ordering::SeqCst);
}

pub fn get_test_flag_status() -> bool {
    TEST_FLAG.load(Ordering::SeqCst)
}
