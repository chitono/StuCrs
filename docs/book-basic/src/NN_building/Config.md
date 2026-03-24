# Configの実装

ここではフレームワーク全体の設定を管理するConfigファイルを実装します。

今までの同様にに **core.rs** ファイルと同じ階層に **config.rs** ファイルを追加し、**lib.rs**、**mod.rs** に名前を追加しておきます。

```rust
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
/// Variableや関数たちにidを付けるための値
pub static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// 微分するかしないかというフラグ
/// 推論するときなど、微分する必要がないときに切り替える
pub static GRAD_CONFIG: AtomicBool = AtomicBool::new(true);

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
```

はじめに **id** に関する設定を行います。[微分の実装（複雑な関数）](../Aut_dif_realnumber/bibun_jissou_hukuzatu.md)で実装した **id** を扱う **NEXT_ID**、 **id_generator()** をここに移します。また高階微分を行うかというフラグ **GRAD_CONFIG** の設定もここで行います。このフラグの状態、または変更する関数を実装します。

>高階微分に関しては補足のところで解説します。 
  
TODO:高階微分補足の場所で説明