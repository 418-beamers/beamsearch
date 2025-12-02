#ifndef EXPAND_CUH
#define EXPAND_CUH

#include "../beam_search.cuh"
#include <cuda_runtime.h>

__global__ void expand(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    const float* log_probs, 
    const int* input_lengths,
    int time_step
);

#endif

