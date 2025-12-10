#ifndef COMPACT_CUH
#define COMPACT_CUH

#include "../beam_search.cuh"

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
);

#endif
