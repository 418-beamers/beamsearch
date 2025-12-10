#include "compact.cuh"
#include "../utils.cuh"

__global__ void compact_and_score_kernel(
    int num_unique,
    const int* unique_indices,
    const int* cand_token,
    const int* cand_last_token,
    const int* cand_parent_idx,
    const float* unique_score_blank,
    const float* unique_score_non_blank,
    int* unique_token,
    int* unique_last_token,
    int* unique_parent_idx,
    float* unique_score_total,
    int* unique_indices_out,
    unsigned long long* unique_sort_keys,
    const unsigned int* unique_keys,
    int hash_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;

    int gather_idx = unique_indices[idx];
    
    unique_token[idx] = cand_token[gather_idx];
    unique_last_token[idx] = cand_last_token[gather_idx];
    unique_parent_idx[idx] = cand_parent_idx[gather_idx];

    float score = log_add(unique_score_blank[idx], unique_score_non_blank[idx]);
    unique_score_total[idx] = score;
    
    unique_indices_out[idx] = idx;

    unsigned int key = unique_keys[idx];
    unsigned int batch_idx = key >> hash_bits;

    unique_sort_keys[idx] = ((unsigned long long)batch_idx << 32) | __float_as_uint(score);
}
