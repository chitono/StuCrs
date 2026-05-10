// Transform operations kernels
extern "C" {

// 2D transpose kernel
__global__ void transpose_2d_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        int input_idx = idy * cols + idx;
        int output_idx = idx * rows + idy;
        output[output_idx] = input[input_idx];
    }
}

} // extern "C"