#ifndef INITIALIZE_CUH
#define INITIALIZE_CUH

#include "../beam_search.cuh"
#include <cuda_runtime.h>

__global__ void initialize(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config
);

#endif

