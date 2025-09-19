extern "C" __global__ void sin_kernel(float *out, const float *inp, int row, int col) {
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2次元インデックスを1次元インデックスに変換
    int idx = y * col + x;

    if (idx < row*col) {
        out[idx] = cos(inp[idx]);
    }
}