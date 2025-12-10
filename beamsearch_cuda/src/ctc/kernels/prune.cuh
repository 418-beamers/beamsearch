#ifndef PRUNE_CUH
#define PRUNE_CUH

#include "../beam_search.cuh"

__global__ void select_top_k(
    const float* log_probs,
    int* selected_tokens,
    int num_classes,
    int k,
    int batch_size,
    int blank_id,
    float token_min_logp
);

#endif

