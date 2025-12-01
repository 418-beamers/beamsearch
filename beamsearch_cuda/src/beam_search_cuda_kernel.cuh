#ifndef BEAM_SEARCH_CUDA_KERNEL_CUH
#define BEAM_SEARCH_CUDA_KERNEL_CUH

#include <cuda_runtime.h>

struct CTCBeamSearchConfig {
    int batchSize;
    int beamWidth;
    int numClasses;      // vocabulary size (including blank)
    int maxTime;         // maximum number of time steps
    int maxOutputLength; // maximum output sequence length
    int blankId;         // blank token id (typically 0)
};

struct CTCBeamSearchState {
    // Prefix sequences and scores
    int* prefixes;          // [batchSize, beamWidth, maxOutputLength] - output sequences
    int* prefixLengths;     // [batchSize, beamWidth] - length of each prefix
    float* probBlank;       // [batchSize, beamWidth] - P(prefix ends in blank)
    float* probNonBlank;    // [batchSize, beamWidth] - P(prefix ends in non-blank)
    float* probTotal;       // [batchSize, beamWidth] - total probability (Pb + Pnb)

    // Temporary buffers for beam expansion
    float* nextProbBlank;   // [batchSize, beamWidth * (numClasses + 1)]
    float* nextProbNonBlank;// [batchSize, beamWidth * (numClasses + 1)]
    float* nextProbTotal;   // [batchSize, beamWidth * (numClasses + 1)]
    int* nextPrefixes;      // [batchSize, beamWidth * (numClasses + 1), maxOutputLength]
    int* nextPrefixLengths; // [batchSize, beamWidth * (numClasses + 1)]
    int* nextLabels;        // [batchSize, beamWidth * (numClasses + 1)] - last character

    // For sorting and selecting top-k
    int* sortedIndices;     // [batchSize, beamWidth * (numClasses + 1)]
};

/**
 * Initialize CTC beam search state
 * All beams start with empty prefix
 */
void launchInitializeCTCBeamSearch(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config,
    cudaStream_t stream = 0
);

/**
 * Execute CTC beam search on a batch of input sequences
 *
 * @param state          Beam state (updated in-place)
 * @param config         Search configuration
 * @param logProbs       [batchSize, maxTime, numClasses] - log probabilities from CTC
 * @param inputLengths   [batchSize] - actual length of each sequence (can be NULL for full length)
 */
void launchCTCBeamSearch(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config,
    const float* logProbs,
    const int* inputLengths,
    cudaStream_t stream = 0
);

/**
 * Allocate device memory for CTC beam search state
 */
cudaError_t allocateCTCBeamSearchState(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config
);

/**
 * Free device memory for CTC beam search state
 */
void freeCTCBeamSearchState(CTCBeamSearchState& state);

#endif // BEAM_SEARCH_CUDA_KERNEL_CUH
