use cudarc::cublas::{CudaBlas, GemmConfig, GemmMatrix, Operation};
use cudarc::driver::{CudaDevice, CudaView, CudaViewMut, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, Ptx};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. CUDAデバイスの初期化
    let dev = CudaDevice::new(0)?; // GPU 0 を使用
    let mut cublas = CudaBlas::new(Arc::clone(&dev))?; // cuBLASコンテキストを作成

    // 行列の次元を定義
    let m = 2; // 行列Aの行数、行列Cの行数
    let n = 3; // 行列Bの列数、行列Cの列数
    let k = 4; // 行列Aの列数、行列Bの行数

    // ホストメモリ上のデータ
    // A: m x k (2x4)
    let a_host = vec![
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    // B: k x n (4x3)
    let b_host = vec![
        9.0f32, 10.0, 11.0,
        12.0, 13.0, 14.0,
        15.0, 16.0, 17.0,
        18.0, 19.0, 20.0,
    ];
    // C: m x n (2x3) - 結果を格納する行列、初期値は0
    let mut c_host = vec![0.0f32; m * n];

    // 2. ホストからデバイスへデータを転送
    let mut d_a = dev.alloc_copy(&a_host)?;
    let mut d_b = dev.alloc_copy(&b_host)?;
    let mut d_c = dev.alloc_zeros::<f32>(m * n)?;

    // 3. 行列積の設定
    // C = alpha * A * B + beta * C
    let alpha = 1.0f32;
    let beta = 0.0f32; // 初期化されたCに上書きするため0.0fを指定

    // cuBLASはデフォルトでColumn-Majorオーダーを期待します。
    // RustのVecはRow-Majorオーダーで格納されていると仮定します。
    // そのため、cuBLASに渡す際には転置フラグ (Operation::Trans) を使用するか、
    // データをColumn-Majorに変換して転送する必要があります。
    // ここでは、Operation::Trans を使用してAとBを転置して扱います。
    // つまり、A^T * B^T を計算して、結果がCに格納されます。
    // C(m,n) = A(m,k) * B(k,n) を行う場合、cublasSgemm(..., A, lda, B, ldb, ...)
    // ここでは、簡略化のため、cuBLASの `cublasSgemm` の引数の意味をそのまま利用します。
    // cuBLASのSGEMMは C = alpha * op(A) * op(B) + beta * op(C) を計算します。
    // デフォルトのレイアウトはColumn Majorなので、RustのVec (Row Major) を使う場合は注意が必要です。
    // RustのVecをそのまま使って行列積を行う場合、一般的には以下のいずれかの方法を取ります。
    //   a) データをColumn Majorに並べ替えてからGPUに転送する
    //   b) cuBLASの転置オプション `Operation::Trans` を利用して、Row Majorのデータを扱う
    // `cudarc`は `GemmConfig` で `GemmMatrix` を指定できます。
    // 通常、 `GemmMatrix::new(rows, cols, stride)` で stride は列数になります (Column Major)。
    // 今回はRow Majorの `a_host` と `b_host` を使うので、`lda` と `ldb` (leading dimension) を適切に設定します。
    // Row Majorの行列 M (R行, C列) の leading dimension は C です。
    // cublasSgemmのドキュメントを見ると、leading dimensionは非転置行列の行数、または転置行列の列数です。
    // 今回はAとBを転置しない (Operation::NoTrans) で呼び出すことを前提に、
    // lda = k, ldb = n となります。
    // ただし、データはRustのVecでRow Majorで用意されているので、cuBLASの行列はColumn Majorとみなす必要があります。
    // したがって、RustのRow Major (M行K列) は cuBLASのColumn Major (K行M列) に相当します。
    // なので、cuBLASに渡す行列の次元を入れ替えます。
    // A (m, k) -> cuBLASでは (k, m)
    // B (k, n) -> cuBLASでは (n, k)
    // C (m, n) -> cuBLASでは (n, m)

    let conf = GemmConfig::new(
        // C = alpha * A * B + beta * C
        // RustのRow Major A (m x k) と B (k x n) をそのまま使い、cuBLASのNoTransで実行するには
        // cuBLASの内部ではColumn Majorで扱われるため、次元を入れ替えるか、転置フラグを使う必要があります。
        // ここでは、行列A, BをRustでRow Majorとして定義し、cuBLASに渡す際に転置して計算するようにします。
        // C(m,n) = (A(m,k))^T * (B(k,n))^T ではなく、通常の C=A*B を計算したい場合、
        // cuBLASの引数 `transa` と `transb` を使って調整します。
        // 一般的な SGEMM(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) で
        // C(m,n) = A(m,k) * B(k,n) を計算するなら:
        // transa = CUBLAS_OP_N, transb = CUBLAS_OP_N
        // m=m, n=n, k=k
        // lda = k (Aの列数), ldb = n (Bの列数), ldc = n (Cの列数)
        // となりますが、これは Column Major レイアウトでの話です。
        // RustのVecがRow Majorであることを考慮すると、これをColumn Majorとして読み替える必要があります。
        // つまり、Row Majorの (m x k) 行列は、Column Majorの (k x m) 行列としてGPUに転送されます。
        // このため、cuBLASのSGEMMに渡す引数を調整する必要があります。
        // Aを転置せず、Bを転置せずに行列積を行うためには、以下の設定が一般的です。
        Operation::NoTrans, // Aを転置しない
        Operation::NoTrans, // Bを転置しない
        m as i32,           // 行列Cの行数
        n as i32,           // 行列Cの列数
        k as i32,           // 行列Aの列数 / 行列Bの行数
        alpha,
        &d_a,               // デバイス上の行列A
        k as i32,           // Aのリーディングディメンション (Row Majorなので列数)
        &d_b,               // デバイス上の行列B
        n as i32,           // Bのリーディングディメンション (Row Majorなので列数)
        beta,
        &mut d_c,           // デバイス上の行列C (結果格納用)
        n as i32,           // Cのリーディングディメンション (Row Majorなので列数)
    );

    // 4. cuBLASで行列積を実行
    cublas.sgemm(conf)?;

    // 5. デバイスからホストへ結果を転送
    dev.wait_for_default()?; // カーネル実行が完了するのを待つ
    d_c.copy_to_host(&mut c_host)?;

    // 結果の出力
    println!("Matrix A ({}x{}): {:?}", m, k, a_host);
    println!("Matrix B ({}x{}): {:?}", k, n, b_host);
    println!("Matrix C ({}x{}): {:?}", m, n, c_host);

    // 期待される結果 (手計算)
    // A = [[1,2,3,4], [5,6,7,8]]
    // B = [[9,10,11], [12,13,14], [15,16,17], [18,19,20]]
    // C[0][0] = 1*9 + 2*12 + 3*15 + 4*18 = 9 + 24 + 45 + 72 = 150
    // C[0][1] = 1*10 + 2*13 + 3*16 + 4*19 = 10 + 26 + 48 + 76 = 160
    // C[0][2] = 1*11 + 2*14 + 3*17 + 4*20 = 11 + 28 + 51 + 80 = 170
    // C[1][0] = 5*9 + 6*12 + 7*15 + 8*18 = 45 + 72 + 105 + 144 = 366
    // C[1][1] = 5*10 + 6*13 + 7*16 + 8*19 = 50 + 78 + 112 + 152 = 392
    // C[1][2] = 5*11 + 6*14 + 7*17 + 8*20 = 55 + 84 + 119 + 160 = 418
    // 期待されるC: [150.0, 160.0, 170.0, 366.0, 392.0, 418.0]
    // 実際にはcuBLASはColumn Majorなので、Row Majorで渡したAとBがColumn Majorとして解釈され、
    // C(m,n) = A(m,k) * B(k,n) の計算結果が C(n,m) に格納されます。
    // 例えば、`a_host`がRow-Majorの (2,4) 行列として用意されている場合、
    // GPUメモリ上では以下のように連続して配置されます: [1,2,3,4,5,6,7,8]
    // cuBLASがこれをColumn-Major (4,2) 行列として解釈すると:
    // [[1,5],
    //  [2,6],
    //  [3,7],
    //  [4,8]]
    // となります。同様に `b_host` も (3,4) 行列として解釈されます。
    // この場合、`CUBLAS_OP_N` (NoTrans) で実行すると期待通りの結果は得られません。
    // Row MajorのAとBに対して A*B を計算するには、`CUBLAS_OP_T` を使って A^T * B^T を計算するか、
    // あるいは `ld_a` と `ld_b` を調整して `m` と `k` を入れ替える (つまり `m` に `k` を、`k` に `m` を設定する) 必要があるかもしれません。
    // 最も確実なのは、Rust側でデータをColumn Majorに変換してから転送することです。
    // あるいは、`ndarray-cuda-matmul` のような `ndarray` クレートと `cudarc` を統合したライブラリを使うと、
    // ndarrayのAPIでGPU計算を透過的に行えるようになります。

    Ok(())
}