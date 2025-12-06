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

    int candTokenIdx = idx % config.num_classes;

    int globalBeamIdx = idx / config.num_classes;

    int beamIdx = globalBeamIdx % current_beam_width;
    int batchIdx = globalBeamIdx / current_beam_width;
    
    int flatBeamIdx = batchIdx * config.beam_width + beamIdx;
    
    float sBlank = state.beam.score_blank[flatBeamIdx];
    float sNonBlank = state.beam.score_non_blank[flatBeamIdx];

    // checking that beam is "alive" (dead beams were pruned away/unitialized)
    if (sBlank <= NEG_INF && sNonBlank <= NEG_INF) {
        state.cand.keys[idx] = UINT_MAX; 
        state.cand.score_blank[idx] = NEG_INF;
        state.cand.score_non_blank[idx] = NEG_INF;
        return;
    }

    bool finished = (input_lengths != nullptr && time_step >= input_lengths[batchIdx]);

    if (finished) {
        if (candTokenIdx == config.blank_id) {
            // keep valid beam alive for selection
            unsigned int hash = state.beam.prefix_hashes[flatBeamIdx];
            unsigned int mask = (1u << config.hash_bits) - 1;
            state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (hash & mask);
            state.cand.score_blank[idx] = sBlank;
            state.cand.score_non_blank[idx] = sNonBlank;
            state.cand.parent_idx[idx] = beamIdx; 
            state.cand.token[idx] = -1; 
            state.cand.last_token[idx] = state.beam.last_tokens[flatBeamIdx];
        } else {
            // filter out invalid beams (emission after end of input)
            state.cand.keys[idx] = UINT_MAX;
            state.cand.score_blank[idx] = NEG_INF;
            state.cand.score_non_blank[idx] = NEG_INF; 
        }
        return;
    }
    
    // if not finished, we can expand the beam
    float logProb = log_probs[(batchIdx * config.num_classes) + candTokenIdx]; 
    int prevLastToken = state.beam.last_tokens[flatBeamIdx];
    unsigned int oldHash = state.beam.prefix_hashes[flatBeamIdx];
    unsigned int mask = (1u << config.hash_bits) - 1;

    if (candTokenIdx == config.blank_id) {
        state.cand.keys[idx] = ((unsigned int)batchIdx << config.hash_bits) | (oldHash & mask);
        state.cand.score_blank[idx] = log_add_helper(sBlank, sNonBlank) + logProb;
        
        // only possible if we have a previous token to repeat 
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
        // handling non-blank token extension
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
            state.cand.score_non_blank[idx] = log_add_helper(sBlank, sNonBlank) + logProb;
            state.cand.parent_idx[idx] = beamIdx;
            state.cand.token[idx] = candTokenIdx;
            state.cand.last_token[idx] = candTokenIdx;
        }
    }
}

