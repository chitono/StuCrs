// Transform operations kernels
extern "C" {

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

    int output_idx = batch_idx * (c * kh *kw *oh *ow) + kernel_idx * (oh * ow) + ohw_idx;

    if (ih_idx >= 0 && ih_idx < h && iw_idx >= 0 && iw_idx < w) {
        int input_idx = (batch_idx * c + c_idx) * h * w + ih_idx * w + iw_idx;
        output[output_idx] = input[input_idx];
    }else{
        output[output_idx] = 0.0f;
    }
}


__global__ void col2im_kernel(const float* input, float* output,
                            int n, int c, int h, int w,
                            int kh, int kw, int oh, int ow,
                            int stride_h, int stride_w, 
                            int pad_h, int pad_w) {
    int iw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ih_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_c_idx = blockIdx.z;
    
    if (ih_idx >= h || iw_idx >= w || batch_c_idx >= n * c) {
        return;
    }

    int batch_idx = batch_c_idx / c;
    int c_idx = batch_c_idx % c;

    int output_idx = (batch_idx * c + c_idx) * h * w + ih_idx * w + iw_idx;

    float sum = 0.0f;
    
    for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
        for (int kw_idx = 0; kw_idx < kw; kw_idx++) {

            int h_pad = ih_idx + pad_h - kh_idx;
            int w_pad = iw_idx + pad_w - kw_idx;

            if (h_pad < 0 || w_pad < 0)
                continue;

            if (h_pad % stride_h != 0 || w_pad % stride_w != 0)
                continue;

            int oh_idx = h_pad / stride_h;
            int ow_idx = w_pad / stride_w;

            if (oh_idx >= oh || ow_idx >= ow)
                continue;
            
            int kernel_idx = c_idx * kh * kw + kh_idx * kw + kw_idx;

            int col_idx = oh_idx * ow + ow_idx;

            int input_idx = batch_idx * (c * kh * kw) * (oh * ow) + kernel_idx * (oh * ow) + col_idx;

            sum += input[input_idx];
            

            
        }
    }
    output[output_idx] = sum;

    
}

} // extern "C"