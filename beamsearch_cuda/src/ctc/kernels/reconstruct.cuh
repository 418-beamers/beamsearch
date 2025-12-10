#ifndef RECONSTRUCT_CUH
#define RECONSTRUCT_CUH

#include "../beam_search.cuh"
#include <cuda_runtime.h>

__global__ void reconstruct(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    const int* input_lengths
);

#endif

