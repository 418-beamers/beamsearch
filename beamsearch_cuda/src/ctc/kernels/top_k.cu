#include "top_k.cuh"
#include "../utils.cuh"

__global__ void top_k(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    int num_unique,
    int time_step,
    int beam_width
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= config.batch_size) return;

    __shared__ int shared_batchStart;
    __shared__ bool shared_batchFound;
    __shared__ float shared_max_batch_score;

    if (threadIdx.x == 0) {
        // finding the start index for hypotheses in the current batch
        int left = 0;
        int right = num_unique;
        int batchStart = -1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            int actualIdx = state.unique.indices[mid];
            int midBatch = static_cast<int>(state.unique.keys[actualIdx] >> config.hash_bits);
            
            if (midBatch < batchIdx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        batchStart = left;
        shared_batchStart = batchStart;

        bool batchFound = false;
        if (batchStart < num_unique) {
            int actualIdx = state.unique.indices[batchStart];
            if (static_cast<int>(state.unique.keys[actualIdx] >> config.hash_bits) == batchIdx) {
                batchFound = true;
            }
        }
        shared_batchFound = batchFound;

        float max_batch_score = NEG_INF;
        if (batchFound) {
            int bestIdx = state.unique.indices[batchStart];
            if (static_cast<int>(state.unique.keys[bestIdx] >> config.hash_bits) == batchIdx) {
                max_batch_score = state.unique.score_total[bestIdx];
            }
        }
        shared_max_batch_score = max_batch_score;
    }
    __syncthreads();

    int batchStart = shared_batchStart;
    bool batchFound = shared_batchFound;
    float max_batch_score = shared_max_batch_score;

    unsigned int mask = (1u << config.hash_bits) - 1;

    for (int k = threadIdx.x; k < config.beam_width; k += blockDim.x) {
        int globalBeamIdx = batchIdx * config.beam_width + k;
        
        bool isValid = batchFound && (batchStart + k < num_unique) && (k < beam_width);
        if (isValid) {
            int sortedIdx = batchStart + k;
            int uniqueIdx = state.unique.indices[sortedIdx];
            
            if (static_cast<int>(state.unique.keys[uniqueIdx] >> config.hash_bits) == batchIdx) {
                
                float total_score = state.unique.score_total[uniqueIdx];
                
                if (total_score < (max_batch_score - config.beam_threshold)) {
                    isValid = false;
                } else {
                    state.beam.score_blank[globalBeamIdx] = state.unique.score_blank[uniqueIdx];
                    state.beam.score_non_blank[globalBeamIdx] = state.unique.score_non_blank[uniqueIdx];
                    state.beam.score_total[globalBeamIdx] = total_score;

                    unsigned int key = state.unique.keys[uniqueIdx];
                    state.beam.prefix_hashes[globalBeamIdx] = key & mask;
                    state.beam.last_tokens[globalBeamIdx] = state.unique.last_token[uniqueIdx];

                    int parent = state.unique.parent_idx[uniqueIdx];
                    int token = state.unique.token[uniqueIdx];

                    int histIdx = time_step * config.batch_size * config.beam_width + globalBeamIdx;
                    state.beam.history_parents[histIdx] = parent;
                    state.beam.history_tokens[histIdx] = token;
                }
            } else {
                isValid = false;
            }
        }
        
        if (!isValid) {
            state.beam.score_blank[globalBeamIdx] = NEG_INF;
            state.beam.score_non_blank[globalBeamIdx] = NEG_INF;
            state.beam.score_total[globalBeamIdx] = NEG_INF;
            int histIdx = time_step * config.batch_size * config.beam_width + globalBeamIdx;
            state.beam.history_parents[histIdx] = -1;
            state.beam.history_tokens[histIdx] = -1; 
        }
    }
}

