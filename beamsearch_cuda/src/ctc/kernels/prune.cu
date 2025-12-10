#include "prune.cuh"
#include <cfloat>

__global__ void select_top_k(
    const float* log_probs,
    int* selected_tokens,
    int num_classes,
    int k,
    int batch_size,
    int blank_id,
    float token_min_logp
) {
    extern __shared__ float s_mem[];
    float* s_probs = s_mem;
    int* s_indices = (int*)&s_probs[num_classes];

    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* batch_probs = log_probs + (long long)batch_idx * num_classes;
    int* batch_output = selected_tokens + (long long)batch_idx * k;

    for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
        s_probs[i] = batch_probs[i];
        s_indices[i] = i;
    }
    __syncthreads();

    for (int iter = 0; iter < k; ++iter) {
        float my_max_val = -FLT_MAX;
        int my_max_idx = -1;
        
        for (int i = threadIdx.x; i < num_classes; i += blockDim.x) {
            float val = s_probs[i];
            if (val > my_max_val) {
                my_max_val = val;
                my_max_idx = i;
            }
        }
        
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, my_max_val, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, my_max_idx, offset);
            
            if (other_val > my_max_val) {
                my_max_val = other_val;
                my_max_idx = other_idx;
            }
        }
        
        if (threadIdx.x == 0) {
            if (my_max_idx != -1 && my_max_val >= token_min_logp) {
                batch_output[iter] = my_max_idx;
                s_probs[my_max_idx] = -FLT_MAX; 
            } else {
                batch_output[iter] = -1; 
            }
        }
        __syncthreads();
    }
}

