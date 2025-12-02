#include "initialize.cuh"
#include "../utils.cuh"

__global__ void initialize(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.batch_size * config.beam_width) return;

    int beamIdx = idx % config.beam_width;

    if (beamIdx == 0) {
        state.prob_blank[idx] = 0.0f; 
        state.prob_non_blank[idx] = NEG_INF;
        state.prob_total[idx] = 0.0f;
        state.prefix_hashes[idx] = 0; 
        state.current_lengths[idx] = 0;
        state.last_tokens[idx] = -1; 
    } else {
        state.prob_blank[idx] = NEG_INF;
        state.prob_non_blank[idx] = NEG_INF;
        state.prob_total[idx] = NEG_INF;
        state.prefix_hashes[idx] = 0;
        state.current_lengths[idx] = 0;
        state.last_tokens[idx] = -1;
    }
}

