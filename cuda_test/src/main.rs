use cudarc::cublas::{self, CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::{Array, Array2, ArrayD, ArrayViewMut, ArrayViewMut2, array};

fn main() {
    let Vec = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let mut a = Array::from_shape_vec((2, 4), Vec).unwrap();

    mut_arr(a.view_mut());
    println!("{:?}", a);
}

fn mut_arr(mut av: ArrayViewMut2<i32>) {
    av[[0, 0]] = 0;
}

/*


const PTX_SRC: &str = "
extern \"C\" __global__ void matpow(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;


    float tmpSum = 0;


    if (ROW < N && COL < N) {
        // each thread computes one element of the block pow-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}
";


*/

/*
fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();


    let kernel_file_path = "src/kernels/matpow.cu";
    let ptx_src = std::fs::read_to_string(kernel_file_path).unwrap();


    let ptx = compile_ptx(ptx_src).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());


    let dev = CudaDevice::new(0)?;


    println!("Built in {:?}", start.elapsed());


    dev.load_ptx(ptx, "matpow", &["matpow"])?;
    let f = dev.get_func("matpow", "matpow").unwrap();
    println!("Loaded in {:?}", start.elapsed());


    let a_host = [1.0f32, 2.0, 3.0, 4.0];
    let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let mut c_host = [0.0f32; 4];


    let a_dev = dev.htod_sync_copy(&a_host)?;
    let b_dev = dev.htod_sync_copy(&b_host)?;
    let mut c_dev = dev.htod_sync_copy(&c_host)?;


    println!("Copied in {:?}", start.elapsed());


    let cfg = LaunchConfig {
        block_dim: (2, 2, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, 2i32)) }?;


    dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}




fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();


    println!("Compilation succeeded in {:?}", start.elapsed());


    let dev = CudaDevice::new(0)?;
    let cublas = CudaBlas::new(dev.clone()).unwrap();


    println!("Built in {:?}", start.elapsed());


    println!("Loaded in {:?}", start.elapsed());


    println!("Loaded in {:?}", start.elapsed());


    let row = 200;
    let col = 500;
    let k = 300;


    let a_host = vec![2.0f32; row*k];
    let b_host = vec![5.0f32; k*col];
    let mut c_host = vec![0.0f32; row*col];


    let a_dev = dev.htod_copy(a_host)?;
    let b_dev = dev.htod_sync_copy(&b_host)?;
    let mut c_dev = dev.htod_sync_copy(&c_host)?;


    println!("Copied in {:?}", start.elapsed());


    let alpha = 1.0f32;
    let beta = 1.0f32;


    let cfg = GemmConfig {
        transa: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: row as i32,
        n: col as i32,
        k: k as i32,
        alpha: alpha,
        lda: row as i32, // 行列Aの(a×b)のaの数
        ldb: k as i32,
        beta: beta,
        ldc: row as i32,
    };


    //let cfg2 = GemvConfig{}


    unsafe {
        let _ = cublas.gemm(cfg, &a_dev, &b_dev, &mut c_dev);
    };


    dev.dtoh_sync_copy_into(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
















fn main()  {
    let start = std::time::Instant::now();


    let row = 200;
    let col = 400;
    let k = 300;


    let data_a: Vec<f32> = (0..row*k).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..k*col).map(|i| i as f32).collect();


    // Vecから100x100の行列を作成
    let matrix_a: Array2<f32> = Array2::from_shape_vec((row, k), data_a).unwrap();
    let matrix_b: Array2<f32> = Array2::from_shape_vec((k, col), data_b).unwrap();


    let c = matrix_a.dot(&matrix_b);



    println!("\n行列の積C:\n{:?}", c);
    println!("in {:?}", start.elapsed());
    /*




    // 2x3の行列Aを作成
    let a: Array2<f64> = array![[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]];


    // 3x2の行列Bを作成
    let b: Array2<f64> = array![[7.0, 8.0],
                               [9.0, 10.0],
                               [11.0, 12.0]];


    // 行列の積 C = A * B
    let c = a.dot(&b);


    println!("行列A:\n{:?}", a);
    println!("\n行列B:\n{:?}", b);
    println!("\n行列の積C:\n{:?}", c);


    // 要素ごとの乗算
    let d = a * 2.0;
    println!("\n要素ごとの乗算D:\n{:?}", d);


    */





}


*/

/*



extern "C" __global__ void add(const float* A, const float* B, float* C,int row,int col) {
    //int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // 2次元インデックスを1次元インデックスに変換
    int idx = y * col + x;

    if (idx < row*col) {
        C[idx] = A[idx] + B[idx];
    }



}

 */
