extern "C" __global__ void add_kernel(const float* A, const float* B, float* C,int row,int col) {
    //int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // 2次元インデックスを1次元インデックスに変換
    int idx = y * col + x;
   
    if (idx < row*col) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" __global__ void sub_kernel(const float* A, const float* B, float* C, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = A[idx] - B[idx];
    }

    
}

extern "C" __global__ void mul_kernel(const float* A, const float* B, float* C, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = A[idx] * B[idx];
    }

}

extern "C" __global__ void div_kernel(const float* A, const float* B, float* C, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = A[idx] / B[idx];
    }

    
}

extern "C" __global__ void pow_kernel(const float* A,  float* C,int c, int numel) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    
    if (idx < numel) {
        C[idx] = powf(A[idx],c);
    }

    
}

extern "C" __global__ void sin_kernel(float *out, const float *inp, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}


extern "C" __global__ void cos_kernel(float *out, const float *inp, int row, int col) {
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2次元インデックスを1次元インデックスに変換
    int idx = y * col + x;

    if (idx < row*col) {
        out[idx] = cos(inp[idx]);
    }
}


extern "C" __global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}