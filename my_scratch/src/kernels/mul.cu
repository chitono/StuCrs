extern "C" __global__ void mul(const float* A, const float* B, float* C, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = A[idx] * B[idx];
    }

}