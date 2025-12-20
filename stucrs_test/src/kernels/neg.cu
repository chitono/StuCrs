extern "C" __global__ void neg(const float* A,float* C, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = -A[idx];
    }

    
}