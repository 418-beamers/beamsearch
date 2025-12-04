#include "expand.cuh"
#include "../utils.cuh"

__global__ void expand(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    const float* log_probs, 
    const int* input_lengths,
    int time_step,
    int current_beam_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCandidates = config.batch_size * current_beam_width * config.num_classes;
    
    if (idx >= totalCandidates) return;

    int c = idx % config.num_classes;
    int temp = idx / config.num_classes;
    int beamIdx = temp % current_beam_width;
    int batchIdx = temp / current_beam_width;
    
    int flatBeamIdx = batchIdx * config.beam_width + beamIdx;
    
    float pBlank = state.beam.prob_blank[flatBeamIdx];
    float pNonBlank = state.beam.prob_non_blank[flatBeamIdx];
    
    if (pBlank <= NEG_INF && pNonBlank <= NEG_INF) {
        state.cand.keys[idx] = UINT_MAX; 
        state.cand.prob_blank[idx] = NEG_INF;
        state.cand.prob_non_blank[idx] = NEG_INF;
        return;
    }

    bool finished = (input_lengths != nullptr && time_step >= input_lengths[batchIdx]);

    if (finished) {
        if (c == config.blank_id) {
             unsigned int hash = state.beam.prefix_hashes[flatBeamIdx];
             unsigned int mask = (1u << config.hash_bits) - 1;
             state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (hash & mask);
             state.cand.prob_blank[idx] = pBlank;
             state.cand.prob_non_blank[idx] = pNonBlank;
             state.cand.parent_idx[idx] = beamIdx; 
             state.cand.token[idx] = -1; 
             state.cand.last_token[idx] = state.beam.last_tokens[flatBeamIdx];
        } else {
            state.cand.keys[idx] = UINT_MAX;
            state.cand.prob_blank[idx] = NEG_INF;
            state.cand.prob_non_blank[idx] = NEG_INF;
        }
        return;
    }

    float logProb = log_probs[(batchIdx * config.num_classes) + c]; 
    int prevLastToken = state.beam.last_tokens[flatBeamIdx];
    unsigned int oldHash = state.beam.prefix_hashes[flatBeamIdx];
    unsigned int mask = (1u << config.hash_bits) - 1;

    if (c == config.blank_id) {
        state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (oldHash & mask);
        state.cand.prob_blank[idx] = log_add_helper(pBlank, pNonBlank) + logProb;
        
        if (prevLastToken != -1) {
             float logProbPrev = log_probs[(batchIdx * config.num_classes) + prevLastToken];
             state.cand.prob_non_blank[idx] = pNonBlank + logProbPrev; 
        } else {
             state.cand.prob_non_blank[idx] = NEG_INF;
        }

        state.cand.parent_idx[idx] = beamIdx;
        state.cand.token[idx] = -1; 
        state.cand.last_token[idx] = prevLastToken;
    } else {
        unsigned int newHash = oldHash * 33 + (c + 1);
        
        if (c == prevLastToken) {
             state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (newHash & mask);
             state.cand.prob_blank[idx] = NEG_INF;
             state.cand.prob_non_blank[idx] = pBlank + logProb;
             state.cand.parent_idx[idx] = beamIdx;
             state.cand.token[idx] = c;
             state.cand.last_token[idx] = c;
        } else {
            state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (newHash & mask);
            state.cand.prob_blank[idx] = NEG_INF;
            state.cand.prob_non_blank[idx] = log_add_helper(pBlank, pNonBlank) + logProb;
            state.cand.parent_idx[idx] = beamIdx;
            state.cand.token[idx] = c;
            state.cand.last_token[idx] = c;
        }
    }
}

