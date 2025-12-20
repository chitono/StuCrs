extern "C" __global__ void matpow(const float* A,  float* C,int c, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = powf(A[idx],c);
    }

    
}