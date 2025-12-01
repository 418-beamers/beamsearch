#ifndef BEAM_SEARCH_CUDA_KERNEL_CUH
#define BEAM_SEARCH_CUDA_KERNEL_CUH

#include <cuda_runtime.h>

struct BeamSearchConfig {
    int batchSize;       
    int beamWidth;       
    int vocabSize;       // output vocabulary size
    int maxSeqLength;    
    int eosTokenId;      // EOS token identifier
};

struct BeamSearchState {
    float* beamScores;      // [batchSize, beamWidth] - cumulative scores
    int* beamSequences;     // [batchSize, beamWidth, maxSeqLength] - token sequences
    int* beamLengths;       // [batchSize, beamWidth] - current sequence lengths
    bool* beamFinished;     // [batchSize, beamWidth] - EOS completion flags
    int* parentIndices;     // [batchSize, beamWidth] - parent beam for backtracking

    // temporary buffers
    float* candidateScores; // [batchSize, beamWidth * vocabSize] - all candidates
    int* candidateTokens;   // [batchSize, beamWidth * vocabSize] - candidate tokens
    int* candidateParents;  // [batchSize, beamWidth * vocabSize] - candidate parents
};

/**
 * Initialize beam search state for decoding start
 * Only first beam is active with start token, others inactive
 */
void launchInitializeBeamSearch(
    BeamSearchState& state,
    const BeamSearchConfig& config,
    int startToken,
    cudaStream_t stream = 0
);

/**
 * Execute one decoding step: expand beams and select top-k candidates
 *
 * @param state          Beam state (updated in-place)
 * @param config         Search configuration
 * @param decoderProbs   [batchSize, beamWidth, vocabSize] - probabilities from RNN
 * @param currentStep    Current decoding timestep (0-indexed)
 */
void launchBeamSearchStep(
    BeamSearchState& state,
    const BeamSearchConfig& config,
    const float* decoderProbs,
    int currentStep,
    cudaStream_t stream = 0
);

/**
 * Allocate device memory for beam search state
 */
cudaError_t allocateBeamSearchState(
    BeamSearchState& state,
    const BeamSearchConfig& config
);

/**
 * Free device memory for beam search state
 */
void freeBeamSearchState(BeamSearchState& state);

/**
 * Check if all beams have finished (returns host boolean)
 */
bool checkAllBeamsFinished(
    const BeamSearchState& state,
    const BeamSearchConfig& config
);

#endif // BEAM_SEARCH_CUDA_KERNEL_CUH
