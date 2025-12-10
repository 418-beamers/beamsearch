#include "expand.cuh"
#include "../utils.cuh"

__global__ void expand(
    CTCBeamSearchState state,
    CTCBeamSearchConfig config,
    const float* log_probs, 
    const int* input_lengths,
    int time_step,
    int current_beam_width,
    int expansion_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool use_pruning = (config.prob_top_k < config.num_classes);

    int totalCandidates = config.batch_size * current_beam_width * expansion_factor;
    
    if (idx >= totalCandidates) return;

    int innerIdx = idx % expansion_factor;
    int globalBeamIdx = idx / expansion_factor;

    int beamIdx = globalBeamIdx % current_beam_width;
    int batchIdx = globalBeamIdx / current_beam_width;
    
    int flatBeamIdx = batchIdx * config.beam_width + beamIdx;

    int candTokenIdx;
    if (use_pruning) {
        if (innerIdx == 0) {
            candTokenIdx = config.blank_id;
        } else {
            int token = state.beam.top_k_tokens[batchIdx * config.prob_top_k + (innerIdx - 1)];
            if (token == -1) {
                state.cand.keys[idx] = UINT_MAX;
                state.cand.score_blank[idx] = NEG_INF;
                state.cand.score_non_blank[idx] = NEG_INF;
                return;
            }
            candTokenIdx = token;
        }
    } else {
        candTokenIdx = innerIdx;
    }
    
    float sBlank = state.beam.score_blank[flatBeamIdx];
    float sNonBlank = state.beam.score_non_blank[flatBeamIdx];

    if (sBlank <= NEG_INF && sNonBlank <= NEG_INF) {
        state.cand.keys[idx] = UINT_MAX; 
        state.cand.score_blank[idx] = NEG_INF;
        state.cand.score_non_blank[idx] = NEG_INF;
        return;
    }

    bool finished = (input_lengths != nullptr && time_step >= input_lengths[batchIdx]);

    if (finished) {
        if (candTokenIdx == config.blank_id) {
            unsigned int hash = state.beam.prefix_hashes[flatBeamIdx];
            unsigned int mask = (1u << config.hash_bits) - 1;
            state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (hash & mask);
            state.cand.score_blank[idx] = sBlank;
            state.cand.score_non_blank[idx] = sNonBlank;
            state.cand.parent_idx[idx] = beamIdx; 
            state.cand.token[idx] = -1; 
            state.cand.last_token[idx] = state.beam.last_tokens[flatBeamIdx];
        } else {
            state.cand.keys[idx] = UINT_MAX;
            state.cand.score_blank[idx] = NEG_INF;
            state.cand.score_non_blank[idx] = NEG_INF; 
        }
        return;
    }
    
    float logProb = log_probs[(batchIdx * config.num_classes) + candTokenIdx]; 
    int prevLastToken = state.beam.last_tokens[flatBeamIdx];
    unsigned int oldHash = state.beam.prefix_hashes[flatBeamIdx];
    unsigned int mask = (1u << config.hash_bits) - 1;

    if (candTokenIdx == config.blank_id) {
        state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (oldHash & mask);
        state.cand.score_blank[idx] = log_add(sBlank, sNonBlank) + logProb;
        
        if (prevLastToken != -1) {
            float logProbPrev = log_probs[(batchIdx * config.num_classes) + prevLastToken];
            state.cand.score_non_blank[idx] = sNonBlank + logProbPrev; 
        } else {
            state.cand.score_non_blank[idx] = NEG_INF;
        }

        state.cand.parent_idx[idx] = beamIdx;
        state.cand.token[idx] = -1; 
        state.cand.last_token[idx] = prevLastToken;
    } else {    
        unsigned int newHash = oldHash * 33 + (candTokenIdx + 1);
        
        if (candTokenIdx == prevLastToken) {
            state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (newHash & mask);
            state.cand.score_blank[idx] = NEG_INF;
            state.cand.score_non_blank[idx] = sBlank + logProb;
            state.cand.parent_idx[idx] = beamIdx;
            state.cand.token[idx] = candTokenIdx;
            state.cand.last_token[idx] = candTokenIdx;
        } else {
            state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (newHash & mask);
            state.cand.score_blank[idx] = NEG_INF;
            state.cand.score_non_blank[idx] = log_add(sBlank, sNonBlank) + logProb;
            state.cand.parent_idx[idx] = beamIdx;
            state.cand.token[idx] = candTokenIdx;
            state.cand.last_token[idx] = candTokenIdx;
        }
    }
}
