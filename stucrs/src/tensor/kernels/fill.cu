// Fill operations kernels
extern "C" {

// Fill array with ones
__global__ void fill_ones_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

} // extern "C"