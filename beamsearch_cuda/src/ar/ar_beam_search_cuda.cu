#include "beam_search_cuda_kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <cub/cub.cuh>

// ============================================================================
// HELPER FUNCTIONS AND MACROS
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define NEG_INF -1e20f

// Log-space addition: log(exp(a) + exp(b))
__device__ __forceinline__ float logAdd(float a, float b) {
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float maxVal = fmaxf(a, b);
    return maxVal + log1pf(expf(-fabsf(a - b)));
}

// ============================================================================
// INITIALIZATION KERNEL
// ============================================================================

__global__ void initializeCTCBeamSearchKernel(
    int* prefixes,
    int* prefixLengths,
    float* probBlank,
    float* probNonBlank,
    float* probTotal,
    int batchSize,
    int beamWidth,
    int maxOutputLength
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize) return;

    for (int k = 0; k < beamWidth; k++) {
        int beamIdx = batchIdx * beamWidth + k;

        // All beams start with empty prefix
        prefixLengths[beamIdx] = 0;

        // First beam has probability 1.0 (log(1) = 0), others have 0 (log(0) = -inf)
        if (k == 0) {
            probBlank[beamIdx] = 0.0f;      // log(1) = 0
            probNonBlank[beamIdx] = NEG_INF; // log(0) = -inf
            probTotal[beamIdx] = 0.0f;
        } else {
            probBlank[beamIdx] = NEG_INF;
            probNonBlank[beamIdx] = NEG_INF;
            probTotal[beamIdx] = NEG_INF;
        }

        // Initialize prefix to empty
        for (int i = 0; i < maxOutputLength; i++) {
            prefixes[beamIdx * maxOutputLength + i] = -1;
        }
    }
}

// ============================================================================
// BEAM EXPANSION KERNEL
// ============================================================================

__global__ void expandCTCBeamsKernel(
    const int* prefixes,
    const int* prefixLengths,
    const float* probBlank,
    const float* probNonBlank,
    const float* logProbs,
    int* nextPrefixes,
    int* nextPrefixLengths,
    int* nextLabels,
    float* nextProbBlank,
    float* nextProbNonBlank,
    float* nextProbTotal,
    int batchSize,
    int beamWidth,
    int numClasses,
    int maxOutputLength,
    int blankId,
    int timeStep,
    int maxTime
) {
    // Each thread handles one beam expansion
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalBeams = batchSize * beamWidth;

    if (idx >= totalBeams) return;

    int batchIdx = idx / beamWidth;
    int beamIdx = idx % beamWidth;
    int inputBeamIdx = batchIdx * beamWidth + beamIdx;

    // Get log probabilities for this time step
    int logProbOffset = (batchIdx * maxTime + timeStep) * numClasses;

    float prefixProbBlank = probBlank[inputBeamIdx];
    float prefixProbNonBlank = probNonBlank[inputBeamIdx];
    int prefixLen = prefixLengths[inputBeamIdx];

    // Skip if this beam has no probability mass
    if (prefixProbBlank == NEG_INF && prefixProbNonBlank == NEG_INF) {
        return;
    }

    // Get last character of prefix (-1 if empty)
    int lastChar = (prefixLen > 0) ?
        prefixes[inputBeamIdx * maxOutputLength + prefixLen - 1] : -1;

    // Calculate output indices for this beam's expansions
    int outputBaseIdx = batchIdx * beamWidth * (numClasses + 1) + beamIdx * (numClasses + 1);

    // Extension 1: Blank token (doesn't extend prefix)
    {
        int outIdx = outputBaseIdx;
        float logProbBlankToken = logProbs[logProbOffset + blankId];

        nextProbBlank[outIdx] = logAdd(prefixProbBlank, prefixProbNonBlank) + logProbBlankToken;
        nextProbNonBlank[outIdx] = NEG_INF;
        nextProbTotal[outIdx] = nextProbBlank[outIdx];
        nextPrefixLengths[outIdx] = prefixLen;
        nextLabels[outIdx] = -1; // -1 indicates blank extension

        // Copy prefix
        for (int i = 0; i < prefixLen && i < maxOutputLength; i++) {
            nextPrefixes[outIdx * maxOutputLength + i] =
                prefixes[inputBeamIdx * maxOutputLength + i];
        }
    }

    // Extensions 2-N: Non-blank tokens
    for (int c = 0; c < numClasses; c++) {
        if (c == blankId) continue;

        int outIdx = outputBaseIdx + 1 + ((c < blankId) ? c : c - 1);
        float logProbC = logProbs[logProbOffset + c];

        // Copy prefix first
        for (int i = 0; i < prefixLen && i < maxOutputLength; i++) {
            nextPrefixes[outIdx * maxOutputLength + i] =
                prefixes[inputBeamIdx * maxOutputLength + i];
        }

        if (c == lastChar) {
            // Same character as last: can only extend from blank path
            nextProbBlank[outIdx] = NEG_INF;
            nextProbNonBlank[outIdx] = prefixProbBlank + logProbC;
            nextProbTotal[outIdx] = nextProbNonBlank[outIdx];

            // Extend prefix if there's room
            if (prefixLen < maxOutputLength) {
                nextPrefixes[outIdx * maxOutputLength + prefixLen] = c;
                nextPrefixLengths[outIdx] = prefixLen + 1;
            } else {
                nextPrefixLengths[outIdx] = prefixLen;
            }
            nextLabels[outIdx] = c;

        } else {
            // Different character: can extend from both paths
            nextProbBlank[outIdx] = NEG_INF;
            nextProbNonBlank[outIdx] = logAdd(prefixProbBlank, prefixProbNonBlank) + logProbC;
            nextProbTotal[outIdx] = nextProbNonBlank[outIdx];

            // Extend prefix if there's room
            if (prefixLen < maxOutputLength) {
                nextPrefixes[outIdx * maxOutputLength + prefixLen] = c;
                nextPrefixLengths[outIdx] = prefixLen + 1;
            } else {
                nextPrefixLengths[outIdx] = prefixLen;
            }
            nextLabels[outIdx] = c;
        }
    }
}

// ============================================================================
// PREFIX MERGING KERNEL
// ============================================================================

__global__ void mergePrefixesKernel(
    const int* nextPrefixes,
    const int* nextPrefixLengths,
    const float* nextProbBlank,
    const float* nextProbNonBlank,
    const float* nextProbTotal,
    int* mergedPrefixes,
    int* mergedPrefixLengths,
    float* mergedProbBlank,
    float* mergedProbNonBlank,
    float* mergedProbTotal,
    int batchSize,
    int beamWidth,
    int numClasses,
    int maxOutputLength
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    int numCandidates = beamWidth * (numClasses + 1);
    int candidateOffset = batchIdx * numCandidates;

    extern __shared__ char sharedMem[];
    int* processedMask = (int*)sharedMem;

    // Initialize processed mask
    for (int i = threadIdx.x; i < numCandidates; i += blockDim.x) {
        processedMask[i] = 0;
    }
    __syncthreads();

    // Each thread processes one candidate
    for (int candIdx = threadIdx.x; candIdx < numCandidates; candIdx += blockDim.x) {
        int globalCandIdx = candidateOffset + candIdx;

        // Check if already processed
        if (atomicCAS(&processedMask[candIdx], 0, 1) != 0) {
            continue;
        }

        // This candidate becomes the representative
        int prefixLen = nextPrefixLengths[globalCandIdx];
        float mergedPb = nextProbBlank[globalCandIdx];
        float mergedPnb = nextProbNonBlank[globalCandIdx];

        // Look for duplicates
        for (int otherIdx = candIdx + 1; otherIdx < numCandidates; otherIdx++) {
            if (processedMask[otherIdx] != 0) continue;

            int globalOtherIdx = candidateOffset + otherIdx;
            int otherLen = nextPrefixLengths[globalOtherIdx];

            // Check if lengths match
            if (otherLen != prefixLen) continue;

            // Check if prefixes match
            bool match = true;
            for (int i = 0; i < prefixLen; i++) {
                if (nextPrefixes[globalCandIdx * maxOutputLength + i] !=
                    nextPrefixes[globalOtherIdx * maxOutputLength + i]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                // Merge probabilities
                mergedPb = logAdd(mergedPb, nextProbBlank[globalOtherIdx]);
                mergedPnb = logAdd(mergedPnb, nextProbNonBlank[globalOtherIdx]);
                atomicExch(&processedMask[otherIdx], 1);
            }
        }

        // Write merged result
        mergedProbBlank[globalCandIdx] = mergedPb;
        mergedProbNonBlank[globalCandIdx] = mergedPnb;
        mergedProbTotal[globalCandIdx] = logAdd(mergedPb, mergedPnb);
        mergedPrefixLengths[globalCandIdx] = prefixLen;

        // Copy prefix
        for (int i = 0; i < prefixLen; i++) {
            mergedPrefixes[globalCandIdx * maxOutputLength + i] =
                nextPrefixes[globalCandIdx * maxOutputLength + i];
        }
    }
}

// ============================================================================
// TOP-K SELECTION KERNEL
// ============================================================================

__global__ void selectTopKBeamsKernel(
    const int* candidatePrefixes,
    const int* candidatePrefixLengths,
    const float* candidateProbBlank,
    const float* candidateProbNonBlank,
    const float* candidateProbTotal,
    int* outputPrefixes,
    int* outputPrefixLengths,
    float* outputProbBlank,
    float* outputProbNonBlank,
    float* outputProbTotal,
    int batchSize,
    int beamWidth,
    int numClasses,
    int maxOutputLength
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    int numCandidates = beamWidth * (numClasses + 1);
    int candidateOffset = batchIdx * numCandidates;

    extern __shared__ char sharedMem[];
    float* topScores = (float*)sharedMem;
    int* topIndices = (int*)&topScores[beamWidth];

    // Initialize top-k with first k candidates
    if (threadIdx.x < beamWidth) {
        int candIdx = candidateOffset + threadIdx.x;
        topScores[threadIdx.x] = candidateProbTotal[candIdx];
        topIndices[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    // Find top-k using selection
    for (int candIdx = beamWidth + threadIdx.x; candIdx < numCandidates; candIdx += blockDim.x) {
        int globalCandIdx = candidateOffset + candIdx;
        float score = candidateProbTotal[globalCandIdx];

        // Find minimum in top-k
        float minScore = topScores[0];
        int minPos = 0;
        for (int k = 1; k < beamWidth; k++) {
            if (topScores[k] < minScore) {
                minScore = topScores[k];
                minPos = k;
            }
        }

        // Replace if better
        if (score > minScore) {
            atomicExch((int*)&topScores[minPos], __float_as_int(score));
            atomicExch(&topIndices[minPos], candIdx - candidateOffset);
        }
    }
    __syncthreads();

    // Write output
    if (threadIdx.x < beamWidth) {
        int selectedIdx = candidateOffset + topIndices[threadIdx.x];
        int outputIdx = batchIdx * beamWidth + threadIdx.x;

        outputProbBlank[outputIdx] = candidateProbBlank[selectedIdx];
        outputProbNonBlank[outputIdx] = candidateProbNonBlank[selectedIdx];
        outputProbTotal[outputIdx] = candidateProbTotal[selectedIdx];
        outputPrefixLengths[outputIdx] = candidatePrefixLengths[selectedIdx];

        // Copy prefix
        int prefixLen = candidatePrefixLengths[selectedIdx];
        for (int i = 0; i < prefixLen && i < maxOutputLength; i++) {
            outputPrefixes[outputIdx * maxOutputLength + i] =
                candidatePrefixes[selectedIdx * maxOutputLength + i];
        }
        // Clear rest
        for (int i = prefixLen; i < maxOutputLength; i++) {
            outputPrefixes[outputIdx * maxOutputLength + i] = -1;
        }
    }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

void launchInitializeCTCBeamSearch(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (config.batchSize + threadsPerBlock - 1) / threadsPerBlock;

    initializeCTCBeamSearchKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        state.prefixes,
        state.prefixLengths,
        state.probBlank,
        state.probNonBlank,
        state.probTotal,
        config.batchSize,
        config.beamWidth,
        config.maxOutputLength
    );

    CUDA_CHECK(cudaGetLastError());
}

void launchCTCBeamSearch(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config,
    const float* logProbs,
    const int* inputLengths,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;

    // Process each time step
    for (int t = 0; t < config.maxTime; t++) {
        // Step 1: Expand beams
        int totalBeams = config.batchSize * config.beamWidth;
        int numBlocks = (totalBeams + threadsPerBlock - 1) / threadsPerBlock;

        expandCTCBeamsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            state.prefixes,
            state.prefixLengths,
            state.probBlank,
            state.probNonBlank,
            logProbs,
            state.nextPrefixes,
            state.nextPrefixLengths,
            state.nextLabels,
            state.nextProbBlank,
            state.nextProbNonBlank,
            state.nextProbTotal,
            config.batchSize,
            config.beamWidth,
            config.numClasses,
            config.maxOutputLength,
            config.blankId,
            t,
            config.maxTime
        );
        CUDA_CHECK(cudaGetLastError());

        // Step 2: Merge duplicate prefixes
        int sharedMemSize = config.beamWidth * (config.numClasses + 1) * sizeof(int);

        mergePrefixesKernel<<<config.batchSize, threadsPerBlock, sharedMemSize, stream>>>(
            state.nextPrefixes,
            state.nextPrefixLengths,
            state.nextProbBlank,
            state.nextProbNonBlank,
            state.nextProbTotal,
            state.nextPrefixes,      // in-place
            state.nextPrefixLengths, // in-place
            state.nextProbBlank,     // in-place
            state.nextProbNonBlank,  // in-place
            state.nextProbTotal,     // in-place
            config.batchSize,
            config.beamWidth,
            config.numClasses,
            config.maxOutputLength
        );
        CUDA_CHECK(cudaGetLastError());

        // Step 3: Select top-k beams
        sharedMemSize = config.beamWidth * (sizeof(float) + sizeof(int));

        selectTopKBeamsKernel<<<config.batchSize, threadsPerBlock, sharedMemSize, stream>>>(
            state.nextPrefixes,
            state.nextPrefixLengths,
            state.nextProbBlank,
            state.nextProbNonBlank,
            state.nextProbTotal,
            state.prefixes,
            state.prefixLengths,
            state.probBlank,
            state.probNonBlank,
            state.probTotal,
            config.batchSize,
            config.beamWidth,
            config.numClasses,
            config.maxOutputLength
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

cudaError_t allocateCTCBeamSearchState(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config
) {
    cudaError_t err;

    int numBeams = config.batchSize * config.beamWidth;
    int numCandidates = config.batchSize * config.beamWidth * (config.numClasses + 1);

    // Allocate main beam state
    err = cudaMalloc(&state.prefixes, numBeams * config.maxOutputLength * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.prefixLengths, numBeams * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.probBlank, numBeams * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.probNonBlank, numBeams * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.probTotal, numBeams * sizeof(float));
    if (err != cudaSuccess) return err;

    // Allocate temporary buffers
    err = cudaMalloc(&state.nextPrefixes, numCandidates * config.maxOutputLength * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.nextPrefixLengths, numCandidates * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.nextLabels, numCandidates * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.nextProbBlank, numCandidates * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.nextProbNonBlank, numCandidates * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.nextProbTotal, numCandidates * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.sortedIndices, numCandidates * sizeof(int));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

void freeCTCBeamSearchState(CTCBeamSearchState& state) {
    cudaFree(state.prefixes);
    cudaFree(state.prefixLengths);
    cudaFree(state.probBlank);
    cudaFree(state.probNonBlank);
    cudaFree(state.probTotal);
    cudaFree(state.nextPrefixes);
    cudaFree(state.nextPrefixLengths);
    cudaFree(state.nextLabels);
    cudaFree(state.nextProbBlank);
    cudaFree(state.nextProbNonBlank);
    cudaFree(state.nextProbTotal);
    cudaFree(state.sortedIndices);
}
