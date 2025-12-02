#include "top_k.cuh"
#include "../utils.cuh"

__global__ void top_k(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    int num_unique,
    int time_step
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= config.batch_size) return;

    int left = 0;
    int right = num_unique;
    int batchStart = -1;

    while (left < right) {
        int mid = left + (right - left) / 2;
        int actualIdx = state.unique_indices[mid];
        int midBatch = static_cast<int>(state.unique_keys[actualIdx] >> 16);
        
        if (midBatch < batchIdx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    batchStart = left;

    bool batchFound = false;
    if (batchStart < num_unique) {
        int actualIdx = state.unique_indices[batchStart];
        if (static_cast<int>(state.unique_keys[actualIdx] >> 16) == batchIdx) {
            batchFound = true;
        }
    }

    for (int k = threadIdx.x; k < config.beam_width; k += blockDim.x) {
        int globalBeamIdx = batchIdx * config.beam_width + k;
        
        bool isValid = batchFound && (batchStart + k < num_unique);
        if (isValid) {
             int sortedIdx = batchStart + k;
             int uniqueIdx = state.unique_indices[sortedIdx];
             
             if (static_cast<int>(state.unique_keys[uniqueIdx] >> 16) == batchIdx) {
                state.prob_blank[globalBeamIdx] = state.unique_prob_blank[uniqueIdx];
                state.prob_non_blank[globalBeamIdx] = state.unique_prob_non_blank[uniqueIdx];
                state.prob_total[globalBeamIdx] = state.unique_prob_total[uniqueIdx];

                unsigned int key = state.unique_keys[uniqueIdx];
                state.prefix_hashes[globalBeamIdx] = key & 0xFFFF;
                state.last_tokens[globalBeamIdx] = state.unique_last_token[uniqueIdx];

                int parent = state.unique_parent_idx[uniqueIdx];
                int token = state.unique_token[uniqueIdx];

                int histIdx = time_step * config.batch_size * config.beam_width + globalBeamIdx;
                state.history_parents[histIdx] = parent;
                state.history_tokens[histIdx] = token;
             } else {
                 isValid = false;
             }
        }
        
        if (!isValid) {
            state.prob_blank[globalBeamIdx] = NEG_INF;
            state.prob_non_blank[globalBeamIdx] = NEG_INF;
            state.prob_total[globalBeamIdx] = NEG_INF;
            int histIdx = time_step * config.batch_size * config.beam_width + globalBeamIdx;
            state.history_parents[histIdx] = -1;
            state.history_tokens[histIdx] = -1; 
        }
    }
}

