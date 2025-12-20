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