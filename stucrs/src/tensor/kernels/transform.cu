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


// permute_axes kernel
__global__ void permute_kernel(const float* input, float* output,
                                const int* in_strides, const int* out_strides,
                                const int* axes, int ndim,int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numel) return;

    int coords[6] = {0};
    int tmp = idx;

    for (int i = 0; i < ndim; ++i) {
        coords[i] = tmp / out_strides[i];
        tmp %= out_strides[i];

    }


    int in_idx = 0;

    for (int i = 0; i< numel; ++i) {
        in_idx += coords[i] * in_strides[axes[i]];
    }

    output[idx] = input[in_idx];

}

} // extern "C"