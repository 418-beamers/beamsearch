#include "expand.cuh"
#include "../utils.cuh"

__global__ void expand(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    const float* log_probs, 
    const int* input_lengths,
    int time_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCandidates = config.batch_size * config.beam_width * config.num_classes;
    
    if (idx >= totalCandidates) return;

    int c = idx % config.num_classes;
    int temp = idx / config.num_classes;
    int beamIdx = temp % config.beam_width;
    int batchIdx = temp / config.beam_width;
    int flatBeamIdx = batchIdx * config.beam_width + beamIdx;
    
    float pBlank = state.prob_blank[flatBeamIdx];
    float pNonBlank = state.prob_non_blank[flatBeamIdx];
    
    if (pBlank <= NEG_INF && pNonBlank <= NEG_INF) {
        state.cand_keys[idx] = UINT_MAX; 
        state.cand_prob_blank[idx] = NEG_INF;
        state.cand_prob_non_blank[idx] = NEG_INF;
        return;
    }

    bool finished = (input_lengths != nullptr && time_step >= input_lengths[batchIdx]);

    if (finished) {
        if (c == config.blank_id) {
             unsigned int hash = state.prefix_hashes[flatBeamIdx];
             // Combine batchIdx (16 bits) and hash (16 bits)
             state.cand_keys[idx] = ((unsigned int)batchIdx << 16) | (hash & 0xFFFF);
             state.cand_prob_blank[idx] = pBlank;
             state.cand_prob_non_blank[idx] = pNonBlank;
             state.cand_parent_idx[idx] = beamIdx; 
             state.cand_token[idx] = -1; 
             state.cand_last_token[idx] = state.last_tokens[flatBeamIdx];
        } else {
            state.cand_keys[idx] = UINT_MAX;
            state.cand_prob_blank[idx] = NEG_INF;
            state.cand_prob_non_blank[idx] = NEG_INF;
        }
        return;
    }

    float logProb = log_probs[(batchIdx * config.num_classes) + c]; 
    int prevLastToken = state.last_tokens[flatBeamIdx];
    unsigned int oldHash = state.prefix_hashes[flatBeamIdx];

    if (c == config.blank_id) {
        state.cand_keys[idx] = ((unsigned int)batchIdx << 16) | (oldHash & 0xFFFF);
        state.cand_prob_blank[idx] = log_add_helper(pBlank, pNonBlank) + logProb;
        
        if (prevLastToken != -1) {
             float logProbPrev = log_probs[(batchIdx * config.num_classes) + prevLastToken];
             state.cand_prob_non_blank[idx] = pNonBlank + logProbPrev; 
        } else {
             state.cand_prob_non_blank[idx] = NEG_INF;
        }

        state.cand_parent_idx[idx] = beamIdx;
        state.cand_token[idx] = -1; 
        state.cand_last_token[idx] = prevLastToken;
    } else {
        unsigned int newHash = oldHash * 33 + (c + 1);
        
        if (c == prevLastToken) {
             state.cand_keys[idx] = ((unsigned int)batchIdx << 16) | (newHash & 0xFFFF);
             state.cand_prob_blank[idx] = NEG_INF;
             state.cand_prob_non_blank[idx] = pBlank + logProb;
             state.cand_parent_idx[idx] = beamIdx;
             state.cand_token[idx] = c;
             state.cand_last_token[idx] = c;
        } else {
            state.cand_keys[idx] = ((unsigned int)batchIdx << 16) | (newHash & 0xFFFF);
            state.cand_prob_blank[idx] = NEG_INF;
            state.cand_prob_non_blank[idx] = log_add_helper(pBlank, pNonBlank) + logProb;
            state.cand_parent_idx[idx] = beamIdx;
            state.cand_token[idx] = c;
            state.cand_last_token[idx] = c;
        }
    }
}

