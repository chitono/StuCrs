// Transform operations kernels
extern "C" {

// 2D transpose kernel
__global__ void im2col_kernel(const float* input, float* output,
                            int n, int c, int h, int w,
                            int kh, int kw, int oh, int ow,
                            int stride_h, int stride_w, 
                            int pad_h, int pad_w) {
    int ohw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kernel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (ohw_idx >= oh * ow || kernel_idx >= c * kh *kw || batch_idx >= n) {
        return;
    }

    int oh_idx = ohw_idx / ow;
    int ow_idx = ohw_idx % ow;

    int c_idx = kernel_idx / (kh*kw);
    int kernel_mod_idx = kernel_idx % (kh*kw);
    int kh_idx = kernel_mod_idx / kw;
    int kw_idx = kernel_mod_idx % kw;

    int ih_idx = oh_idx * stride_h - pad_h + kh_idx;
    int iw_idx = ow_idx * stride_w - pad_w + kw_idx;

    int output_idx = kernel_idx * (oh * ow) + ohw_idx;

    if (ih_idx >= 0 && ih_idx < h && iw_idx >= 0 && iw_idx < w) {
        int input_idx = (batch_idx * c + c_idx) * h * w + ih_idx * w + iw_idx;
        output[output_idx] = input[input_idx];
    }else{
        output[output_idx] = 0.0f;
    }
}


// permute_axes kernel
__global__ void col2im_kernel(const float* input, float* output,
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