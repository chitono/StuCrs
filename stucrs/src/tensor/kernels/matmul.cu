// Matrix multiplication kernels
extern "C" {

// Basic matrix multiplication kernel (C = A * B)
// A: M x K matrix, B: K x N matrix, C: M x N matrix
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized tiled matrix multiplication kernel
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, 
                                    int M, int K, int N) {
    const int TILE_SIZE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Batched matrix multiplication kernel
// A: batch_size x M x K, B: batch_size x K x N, C: batch_size x M x N
__global__ void bmm_kernel(const float* A, const float* B, float* C,
                           int batch_size, int M, int K, int N) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && row < M && col < N) {
        int A_offset = batch_idx * M * K;
        int B_offset = batch_idx * K * N;
        int C_offset = batch_idx * M * N;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[A_offset + row * K + k] * B[B_offset + k * N + col];
        }
        C[C_offset + row * N + col] = sum;
    }
}

} // extern "C"