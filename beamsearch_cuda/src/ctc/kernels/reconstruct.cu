#include "reconstruct.cuh"
#include "../utils.cuh"

__global__ void reconstruct(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.batch_size * config.beam_width) return;

    int beamIdx = idx % config.beam_width;
    int batchIdx = idx / config.beam_width;
    
    int currentBeam = beamIdx;
    int len = 0;
    int* myOutput = state.output.sequences + (batchIdx * config.beam_width + beamIdx) * config.max_output_length;
    
    // reconstructing sequence from history tokens and parents
    for (int t = config.max_time - 1; t >= 0; t--) {
        int histIdx = t * config.batch_size * config.beam_width + batchIdx * config.beam_width + currentBeam;
        int token = state.beam.history_tokens[histIdx];
        int parent = state.beam.history_parents[histIdx];
        
        if (token != -1 && token != config.blank_id) { 
             if (len < config.max_output_length) {
                 myOutput[len++] = token;
             }
        }
        currentBeam = parent;
        if (currentBeam < 0) break; 
    }
    
    for (int i = 0; i < len / 2; i++) {
        int tmp = myOutput[i];
        myOutput[i] = myOutput[len - 1 - i];
        myOutput[len - 1 - i] = tmp;
    }
    
    for (int i = len; i < config.max_output_length; i++) {
        myOutput[i] = -1;
    }
    
    state.output.lengths[idx] = len;
}

