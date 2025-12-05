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
        state.beam.prob_blank[idx] = 0.0f; 
        state.beam.prob_non_blank[idx] = NEG_INF;
        state.beam.prob_total[idx] = 0.0f;
        state.beam.prefix_hashes[idx] = 0; 
        state.beam.current_lengths[idx] = 0;
        state.beam.last_tokens[idx] = -1; 
    } else {
        state.beam.prob_blank[idx] = NEG_INF;
        state.beam.prob_non_blank[idx] = NEG_INF;
        state.beam.prob_total[idx] = NEG_INF;
        state.beam.prefix_hashes[idx] = 0;
        state.beam.current_lengths[idx] = 0;
        state.beam.last_tokens[idx] = -1;
    }
}

