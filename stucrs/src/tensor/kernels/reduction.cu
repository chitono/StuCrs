


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


__global__ void sum_axis_kernel(const float* input, float* output,
                    const int* in_shape, const int* out_shape,
                    const int* in_strides, const int* out_strides,
                    int in_ndim, int out_ndim,
                    int in_n, int out_n, int axis) {

    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_n) {
        return;
    }


    int output_coords[6] = {0};
    int input_coords[6] = {0};

    int tmp = idx;

    for (int i = out_ndim -1; i >= 0; --i) {
        output_coords[i] = tmp % out_shape[i];
        tmp /= out_shape[i];
    }

    int current_out_dim = 0;
    for (int i = 0; i < in_ndim; ++i) {
        if (i == axis) {
            input_coords[i] = 0;
        }else{
            if (current_out_dim < out_ndim) {
                input_coords[i] = output_coords[current_out_dim];
                current_out_dim++;
            }else{
                input_coords[i] = 0;
            }
        }
    }

    int in_idx_base = 0;
    for (int i = 0; i < in_ndim; ++i) {
        in_idx_base += input_coords[i]*in_strides[i];
    }

    int stride_for_axis = in_strides[axis];
    int reduction_size = in_shape[axis];

    float sum = 0.0f;
    for (int k = 0; k < reduction_size; ++k) {
        int in_idx = in_idx_base + k * stride_for_axis;
        if (in_idx >= 0 && in_idx < in_n) {
            sum += input[in_idx];
        }
    }

    output[idx] = sum;
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
                  const int* in_shape, const int* out_shape,
                  const int* in_strides, const int* out_strides,
                  int in_ndim, int out_ndim,
                  int in_n, int out_n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= out_n) return;
    int coords[6] = {0};
    int tmp = idx;

    for (int i = out_ndim -1; i >= 0; --i) {
        coords[i] = tmp % out_shape[i];
        tmp /= out_shape[i];

    }


    int in_idx = 0;

    if (in_n == 1){
        in_idx = 0;
    }else{
        for (int i = 0; i < out_ndim; ++i) {
            int coord = (in_shape[i] ==1) ? 0 : coords[i];
            in_idx += coord * out_strides[i];
        }
            
    }

    output[idx] = input[in_idx];
    
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




__global__ void argmax_axis_kernel(const float* input, float* output,
                    const int* in_shape, const int* out_shape,
                    const int* in_strides, const int* out_strides,
                    int in_ndim, int out_ndim,
                    int in_n, int out_n, int axis) {

    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_n) {
        return;
    }


    int output_coords[6] = {0};
    int input_coords[6] = {0};

    int tmp = idx;

    for (int i = out_ndim -1; i >= 0; --i) {
        output_coords[i] = tmp % out_shape[i];
        tmp /= out_shape[i];
    }

    int current_out_dim = 0;
    for (int i = 0; i < in_ndim; ++i) {
        if (i == axis) {
            input_coords[i] = 0;
        }else{
            if (current_out_dim < out_ndim) {
                input_coords[i] = output_coords[current_out_dim];
                current_out_dim++;
            }else{
                input_coords[i] = 0;
            }
        }
    }

    int in_idx_base = 0;
    for (int i = 0; i < in_ndim; ++i) {
        in_idx_base += input_coords[i]*in_strides[i];
    }

    int stride_for_axis = in_strides[axis];
    int reduction_size = in_shape[axis];



    float max_value = -10000.0;
    int max_idx = 0;
    for (int k = 0; k < reduction_size; ++k) {
        int in_idx = in_idx_base + k * stride_for_axis;
        if (in_idx >= 0 && in_idx < in_n) {
            float current_value = input[in_idx];
            if (current_value > max_value) {
                max_value = current_value;
                max_idx = k;
            }
        }
    }

    output[idx] = (float)max_idx;
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