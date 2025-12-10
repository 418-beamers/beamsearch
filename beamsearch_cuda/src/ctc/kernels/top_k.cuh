#ifndef TOP_K_CUH
#define TOP_K_CUH

#include "../beam_search.cuh"
#include <cuda_runtime.h>

__global__ void top_k(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    int num_unique,
    int time_step,
    int beam_width,
    const int* input_lengths
);

#endif

