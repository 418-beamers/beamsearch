#include "beam_search_cuda_kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <algorithm>

// ============================================================================
// HELPER FUNCTIONS
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


__global__ void initializeBeamSearchKernel(
    float* beamScores,
    int* beamSequences,
    int* beamLengths,
    bool* beamFinished,
    int batchSize,
    int beamWidth,
    int maxSeqLength,
    int startToken
) {
    int batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batchIdx >= batchSize) return;

    for (int k = 0; k < beamWidth; k++) {
        int beamIdx = batchIdx * beamWidth + k;
        beamScores[beamIdx] = (k == 0) ? 1.0f : 0.0f;
        beamLengths[beamIdx] = (k == 0) ? 1 : 0;
        beamFinished[beamIdx] = false;

        if (k == 0) {
            beamSequences[beamIdx * maxSeqLength] = startToken;
        }
    }
}


__global__ void expandBeamCandidatesKernel(
    const float* beamScores,
    const bool* beamFinished,
    const float* decoderProbs,
    float* candidateScores,
    int* candidateTokens,
    int* candidateParents,
    int batchSize,
    int beamWidth,
    int vocabSize
) {
    // Each thread handles one (batch, beam, token) candidate
    int candidateIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCandidates = batchSize * beamWidth * vocabSize;

    if (candidateIdx >= totalCandidates) return;

    int tokenId = candidateIdx % vocabSize;
    int temp = candidateIdx / vocabSize;
    int beamIdx = temp % beamWidth;
    int batchIdx = temp / beamWidth;

    int flatBeamIdx = batchIdx * beamWidth + beamIdx;

    if (beamFinished[flatBeamIdx]) {
        candidateScores[candidateIdx] = 0.0f;
    } else {
        int probIdx = (batchIdx * beamWidth + beamIdx) * vocabSize + tokenId;
        candidateScores[candidateIdx] = beamScores[flatBeamIdx] * decoderProbs[probIdx];
    }

    candidateTokens[candidateIdx] = tokenId;
    candidateParents[candidateIdx] = beamIdx;
}


__global__ void selectTopKCandidatesKernel(
    const float* candidateScores,
    const int* candidateTokens,
    const int* candidateParents,
    float* newBeamScores,
    int* newParentIndices,
    int* selectedTokens,
    int batchSize,
    int beamWidth,
    int vocabSize
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    int numCandidates = beamWidth * vocabSize;
    int candidateOffset = batchIdx * numCandidates;

    extern __shared__ int sharedIndices[];
    float* sharedScores = (float*)&sharedIndices[numCandidates];

    for (int i = threadIdx.x; i < numCandidates; i += blockDim.x) {
        sharedIndices[i] = i;
        sharedScores[i] = candidateScores[candidateOffset + i];
    }
    __syncthreads();


    if (threadIdx.x < beamWidth) {
        int bestIdx = threadIdx.x;
        float bestScore = sharedScores[bestIdx];

        for (int i = 0; i < numCandidates; i++) {
            if (sharedScores[i] > bestScore) {
                int count = 0;
                for (int j = 0; j < numCandidates; j++) {
                    if (sharedScores[j] > sharedScores[i]) count++;
                }
                if (count < threadIdx.x || (count == threadIdx.x && i < bestIdx)) {
                    bestIdx = i;
                    bestScore = sharedScores[i];
                }
            }
        }

        int outputIdx = batchIdx * beamWidth + threadIdx.x;
        newBeamScores[outputIdx] = bestScore;
        newParentIndices[outputIdx] = candidateParents[candidateOffset + bestIdx];
        selectedTokens[outputIdx] = candidateTokens[candidateOffset + bestIdx];
    }
}


__global__ void selectTopKCandidatesOptimizedKernel(
    const float* candidateScores,
    const int* candidateTokens,
    const int* candidateParents,
    float* newBeamScores,
    int* newParentIndices,
    int* selectedTokens,
    int batchSize,
    int beamWidth,
    int vocabSize
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    int numCandidates = beamWidth * vocabSize;
    int candidateOffset = batchIdx * numCandidates;

    extern __shared__ char sharedMem[];
    float* topScores = (float*)sharedMem;
    int* topIndices = (int*)&topScores[beamWidth];

    if (threadIdx.x < beamWidth) {
        topScores[threadIdx.x] = candidateScores[candidateOffset + threadIdx.x];
        topIndices[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();

    for (int candIdx = beamWidth + threadIdx.x; candIdx < numCandidates; candIdx += blockDim.x) {
        float score = candidateScores[candidateOffset + candIdx];

        float minScore = topScores[0];
        int minPos = 0;
        for (int k = 1; k < beamWidth; k++) {
            if (topScores[k] < minScore) {
                minScore = topScores[k];
                minPos = k;
            }
        }

        if (score > minScore) {
            atomicExch((int*)&topScores[minPos], __float_as_int(score));
            atomicExch(&topIndices[minPos], candIdx);
        }
    }
    __syncthreads();

    if (threadIdx.x < beamWidth) {
        int outputIdx = batchIdx * beamWidth + threadIdx.x;
        int selectedIdx = topIndices[threadIdx.x];
        newBeamScores[outputIdx] = topScores[threadIdx.x];
        newParentIndices[outputIdx] = candidateParents[candidateOffset + selectedIdx];
        selectedTokens[outputIdx] = candidateTokens[candidateOffset + selectedIdx];
    }
}


__global__ void updateBeamSequencesKernel(
    const int* oldSequences,
    const int* oldLengths,
    const bool* oldFinished,
    const int* parentIndices,
    const int* selectedTokens,
    int* newSequences,
    int* newLengths,
    bool* newFinished,
    int batchSize,
    int beamWidth,
    int maxSeqLength,
    int eosTokenId,
    int currentStep
) {
    int batchIdx = blockIdx.x;
    int beamIdx = threadIdx.x;

    if (batchIdx >= batchSize || beamIdx >= beamWidth) return;

    int oldBeamIdx = batchIdx * beamWidth + parentIndices[batchIdx * beamWidth + beamIdx];
    int newBeamIdx = batchIdx * beamWidth + beamIdx;

    for (int pos = 0; pos < maxSeqLength; pos++) {
        if (pos <= currentStep) {
            newSequences[newBeamIdx * maxSeqLength + pos] =
                oldSequences[oldBeamIdx * maxSeqLength + pos];
        }
    }

    int newToken = selectedTokens[newBeamIdx];
    if (currentStep + 1 < maxSeqLength) {
        newSequences[newBeamIdx * maxSeqLength + currentStep + 1] = newToken;
    }

    bool parentFinished = oldFinished[oldBeamIdx];
    newFinished[newBeamIdx] = parentFinished || (newToken == eosTokenId);
    newLengths[newBeamIdx] = parentFinished ? oldLengths[oldBeamIdx] : (currentStep + 2);
}


// host function required by header
void launchInitializeBeamSearch(
    BeamSearchState& state,
    const BeamSearchConfig& config,
    int startToken,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (config.batchSize + threadsPerBlock - 1) / threadsPerBlock;

    initializeBeamSearchKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        state.beamScores,
        state.beamSequences,
        state.beamLengths,
        state.beamFinished,
        config.batchSize,
        config.beamWidth,
        config.maxSeqLength,
        startToken
    );

    CUDA_CHECK(cudaGetLastError());
}

void launchBeamSearchStep(
    BeamSearchState& state,
    const BeamSearchConfig& config,
    const float* decoderProbs,
    int currentStep,
    cudaStream_t stream
) {
    int totalCandidates = config.batchSize * config.beamWidth * config.vocabSize;

    // Step 1: Expand all candidates
    int threadsPerBlock = 256;
    int numBlocks = (totalCandidates + threadsPerBlock - 1) / threadsPerBlock;

    expandBeamCandidatesKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        state.beamScores,
        state.beamFinished,
        decoderProbs,
        state.candidateScores,
        state.candidateTokens,
        state.candidateParents,
        config.batchSize,
        config.beamWidth,
        config.vocabSize
    );

    CUDA_CHECK(cudaGetLastError());

    // Step 2: Select top-k candidates per batch
    int sharedMemSize = config.beamWidth * (sizeof(float) + sizeof(int));

    selectTopKCandidatesOptimizedKernel<<<config.batchSize, 256, sharedMemSize, stream>>>(
        state.candidateScores,
        state.candidateTokens,
        state.candidateParents,
        state.beamScores,  
        state.parentIndices,
        state.candidateTokens, 
        config.batchSize,
        config.beamWidth,
        config.vocabSize
    );

    CUDA_CHECK(cudaGetLastError());

    // Step 3: Update sequences with selected tokens
    updateBeamSequencesKernel<<<config.batchSize, config.beamWidth, 0, stream>>>(
        state.beamSequences,
        state.beamLengths,
        state.beamFinished,
        state.parentIndices,
        state.candidateTokens,  
        state.beamSequences,    
        state.beamLengths,
        state.beamFinished,
        config.batchSize,
        config.beamWidth,
        config.maxSeqLength,
        config.eosTokenId,
        currentStep
    );

    CUDA_CHECK(cudaGetLastError());
}

cudaError_t allocateBeamSearchState(
    BeamSearchState& state,
    const BeamSearchConfig& config
) {
    cudaError_t err;

    int numBeams = config.batchSize * config.beamWidth;
    int numCandidates = numBeams * config.vocabSize;

    err = cudaMalloc(&state.beamScores, numBeams * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.beamSequences, numBeams * config.maxSeqLength * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.beamLengths, numBeams * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.beamFinished, numBeams * sizeof(bool));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.parentIndices, numBeams * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.candidateScores, numCandidates * sizeof(float));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.candidateTokens, numCandidates * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&state.candidateParents, numCandidates * sizeof(int));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

void freeBeamSearchState(BeamSearchState& state) {
    cudaFree(state.beamScores);
    cudaFree(state.beamSequences);
    cudaFree(state.beamLengths);
    cudaFree(state.beamFinished);
    cudaFree(state.parentIndices);
    cudaFree(state.candidateScores);
    cudaFree(state.candidateTokens);
    cudaFree(state.candidateParents);
}

bool checkAllBeamsFinished(
    const BeamSearchState& state,
    const BeamSearchConfig& config
) {
    int numBeams = config.batchSize * config.beamWidth;
    bool* hostFinished = new bool[numBeams];

    CUDA_CHECK(cudaMemcpy(hostFinished, state.beamFinished,
                          numBeams * sizeof(bool), cudaMemcpyDeviceToHost));

    bool allFinished = true;
    for (int i = 0; i < numBeams; i++) {
        if (!hostFinished[i]) {
            allFinished = false;
            break;
        }
    }

    delete[] hostFinished;
    return allFinished;
}
