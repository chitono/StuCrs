// Reduction operations kernels
extern "C" {

// Sum reduction kernel
__global__ void sum_kernel(const float* data, float* result, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? data[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void sum_axis0_kernel(const float* input, float* output,
                  int in_rows, int in_cols) {
    extern __shared__ float sdata[];

    int col_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (col_idx >= in_cols) {
        return;
    }

    float row_sum = 0.0f;

    for (int row = tid; row<in_rows;row += blockDim.x) {
        row_sum += input[row*in_cols+col_idx];
    }
    sdata[tid] = row_sum;

    __syncthreads();

    for (unsigned int s = blockDim.x/2;s>0;s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid +s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[col_idx] = sdata[0];
    }


    
}

__global__ void sum_axis1_kernel(const float* input, float* output,
                  int in_rows, int in_cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_rows) return;


    float sum = 0.0f;

    for (int col = 0; col<in_cols; ++col) {
        sum += input[i*in_cols+col];
    }
    output[i] = sum;
}


// Mean kernel (uses sum and divides by size)
__global__ void mean_kernel(const float* data, float* result, int size) {
    // This is a simplified version - in practice you'd use the sum kernel
    // and then divide by size on the host
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? data[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(result, sdata[0] / size);
    }
}



__global__ void broadcast_to_kernel(const float* input, float* output,
                  int in_rows, int in_cols,
                  int out_rows, int out_cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < out_rows && j < out_cols) {
        int src_i = in_rows == 1 ? 0 : i;
        int src_j = in_cols == 1 ? 0 : j;

        output[i * out_cols + j] = input[src_i * in_cols + src_j];
    }
}

__global__ void rows_slice_kernel(const float* input, float* output,
                  int* indices, int num_indices,int cols) {

    
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < num_indices && col < cols) {
        int src_row = indices[row_idx];
        int src_idx = src_row*cols+col;
        int dst_idx = row_idx*cols+col;

        output[dst_idx] = input[src_idx];
    }
}


__global__ void argmax_axis0_2d_kernel(const float* input, float* output,
                  int in_rows, int in_cols) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (col_idx>= in_cols) {
        return;
    }
     


    float max_val = input[col_idx];
    int max_row_idx = 0;



    for (int row = 1; row<in_rows; ++row) {
        float current_val = input[row*in_cols+col_idx];

        //現在の値が前の最大値より大きいか判別
        if (current_val>max_val){
            max_val=current_val;   //最大値を更新
            max_row_idx=row;  // その時のインデックスに更新
        }

    output[col_idx] = (float)max_row_idx;
    
    }
    
    
    

    
}

__global__ void argmax_axis1_2d_kernel(const float* input, float* output,
                  int in_rows, int in_cols) {

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (row_idx>= in_rows) {
        return;
    }
     

    int row_start_idx = row_idx * in_cols;

    float max_val = input[row_start_idx];
    int max_col_idx = 0;



    for (int c = 1; c<in_cols; ++c) {
        float current_val = input[row_start_idx+c];

        //現在の値が前の最大値より大きいか判別
        if (current_val>max_val){
            max_val=current_val;   //最大値を更新
            max_col_idx=c;  // その時のインデックスに更新
        }

    output[row_idx] = (float)max_col_idx;
    
    }



}



__global__ void one_hot_encode_kernel(const float* input, float* output,
                  int N, int num_class) {

    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (row_idx>= N) {
        return;
    }
     
    int label = input[row_idx];

    if (label >=0 && label <num_class){

        output[row_idx*num_class+label]=1.0f;
    }

    

}





} // extern "C"