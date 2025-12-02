#include "ctc_beam_search_cuda_kernel.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/gather.h>

#define NEG_INF -1e20f

__host__ __device__ __forceinline__ float logAdd(float a, float b) {
    if (a <= NEG_INF) return b;
    if (b <= NEG_INF) return a;
    float maxVal = fmaxf(a, b);
    return maxVal + log1pf(expf(-fabsf(a - b)));
}

struct ProbZipReduce {
    __host__ __device__
    thrust::tuple<float, float> operator()(const thrust::tuple<float, float>& a,
                                           const thrust::tuple<float, float>& b) const {
        float a_pb  = thrust::get<0>(a);
        float a_pnb = thrust::get<1>(a);
        float b_pb  = thrust::get<0>(b);
        float b_pnb = thrust::get<1>(b);
        return thrust::make_tuple(logAdd(a_pb, b_pb), logAdd(a_pnb, b_pnb));
    }
};

struct CalcTotalProb {
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        float pb  = thrust::get<0>(t);
        float pnb = thrust::get<1>(t);
        return logAdd(pb, pnb);
    }
};

__global__ void initializeCTCBeamSearchKernel(
    float* probBlank,
    float* probNonBlank,
    float* probTotal,
    unsigned long long* prefixHashes,
    int* currentLengths,
    int* lastTokens,
    int batchSize,
    int beamWidth,
    int blankId
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * beamWidth) return;

    int beamIdx = idx % beamWidth;

    if (beamIdx == 0) {
        probBlank[idx] = 0.0f; 
        probNonBlank[idx] = NEG_INF;
        probTotal[idx] = 0.0f;
        prefixHashes[idx] = 0; 
        currentLengths[idx] = 0;
        lastTokens[idx] = -1; 
    } else {
        probBlank[idx] = NEG_INF;
        probNonBlank[idx] = NEG_INF;
        probTotal[idx] = NEG_INF;
        prefixHashes[idx] = 0;
        currentLengths[idx] = 0;
        lastTokens[idx] = -1;
    }
}

// Host wrapper for initialization kernel
void launchInitializeCTCBeamSearch(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config,
    cudaStream_t stream
) {
    int numBeams = config.batchSize * config.beamWidth;
    int threads = 256;
    int blocks = (numBeams + threads - 1) / threads;

    initializeCTCBeamSearchKernel<<<blocks, threads, 0, stream>>>(
        state.probBlank,
        state.probNonBlank,
        state.probTotal,
        state.prefixHashes,
        state.currentLengths,
        state.lastTokens,
        config.batchSize,
        config.beamWidth,
        config.blankId
    );
}

__global__ void expandCTCBeamsParallelKernel(
    const float* probBlank,
    const float* probNonBlank,
    const unsigned long long* prefixHashes,
    const int* lastTokens,
    const float* logProbs, 
    const int* inputLengths,
    unsigned long long* candKeys,
    float* candProbBlank,
    float* candProbNonBlank,
    int* candParentIdx,
    int* candToken,
    int* candLastToken,
    int batchSize,
    int beamWidth,
    int numClasses,
    int blankId,
    int timeStep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCandidates = batchSize * beamWidth * numClasses;
    
    if (idx >= totalCandidates) return;

    int c = idx % numClasses;
    int temp = idx / numClasses;
    int beamIdx = temp % beamWidth;
    int batchIdx = temp / beamWidth;
    int flatBeamIdx = batchIdx * beamWidth + beamIdx;
    
    float pBlank = probBlank[flatBeamIdx];
    float pNonBlank = probNonBlank[flatBeamIdx];
    
    if (pBlank <= NEG_INF && pNonBlank <= NEG_INF) {
        candKeys[idx] = ULLONG_MAX; 
        candProbBlank[idx] = NEG_INF;
        candProbNonBlank[idx] = NEG_INF;
        return;
    }

    bool finished = (inputLengths != nullptr && timeStep >= inputLengths[batchIdx]);

    if (finished) {
        if (c == blankId) {
             unsigned long long hash = prefixHashes[flatBeamIdx];
             candKeys[idx] = ((unsigned long long)batchIdx << 32) | (hash & 0xFFFFFFFF);
             candProbBlank[idx] = pBlank;
             candProbNonBlank[idx] = pNonBlank;
             candParentIdx[idx] = beamIdx; 
             candToken[idx] = -1; // blank, do not emit
             candLastToken[idx] = lastTokens[flatBeamIdx];
        } else {
            candKeys[idx] = ULLONG_MAX;
            candProbBlank[idx] = NEG_INF;
            candProbNonBlank[idx] = NEG_INF;
        }
        return;
    }

    float logProb = logProbs[(batchIdx * numClasses) + c]; 
    int prevLastToken = lastTokens[flatBeamIdx];
    unsigned long long oldHash = prefixHashes[flatBeamIdx];

    if (c == blankId) {
        candKeys[idx] = ((unsigned long long)batchIdx << 32) | (oldHash & 0xFFFFFFFF);
        candProbBlank[idx] = logAdd(pBlank, pNonBlank) + logProb;
        
        if (prevLastToken != -1) {
             float logProbPrev = logProbs[(batchIdx * numClasses) + prevLastToken];
             candProbNonBlank[idx] = pNonBlank + logProbPrev; 
        } else {
             candProbNonBlank[idx] = NEG_INF;
        }

        candParentIdx[idx] = beamIdx;
        candToken[idx] = -1; // blank, do not emit
        candLastToken[idx] = prevLastToken;
    } else {
        unsigned long long newHash = oldHash * 33 + (c + 1);
        
        if (c == prevLastToken) {
             candKeys[idx] = ((unsigned long long)batchIdx << 32) | (newHash & 0xFFFFFFFF);
             candProbBlank[idx] = NEG_INF;
             candProbNonBlank[idx] = pBlank + logProb;
             candParentIdx[idx] = beamIdx;
             candToken[idx] = c;
             candLastToken[idx] = c;
        } else {
            candKeys[idx] = ((unsigned long long)batchIdx << 32) | (newHash & 0xFFFFFFFF);
            candProbBlank[idx] = NEG_INF;
            candProbNonBlank[idx] = logAdd(pBlank, pNonBlank) + logProb;
            candParentIdx[idx] = beamIdx;
            candToken[idx] = c;
            candLastToken[idx] = c;
        }
    }
}

__global__ void updateHistoryKernel(
    const int* uniqueParentIdx,
    const int* uniqueToken,
    const int* uniqueLastToken,
    const float* uniqueProbBlank,
    const float* uniqueProbNonBlank,
    const float* uniqueProbTotal,
    const unsigned long long* uniqueKeys,
    int numUnique,
    int* historyParents,
    int* historyTokens,
    float* probBlank,
    float* probNonBlank,
    float* probTotal,
    unsigned long long* prefixHashes,
    int* lastTokens,
    int batchSize,
    int beamWidth,
    int timeStep,
    int maxTime
) {
    int batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    extern __shared__ unsigned char smem[];
    float* topScores = reinterpret_cast<float*>(smem);
    int* topIndices = reinterpret_cast<int*>(topScores + beamWidth);

    if (threadIdx.x == 0) {
        for (int k = 0; k < beamWidth; ++k) {
            topScores[k] = NEG_INF;
            topIndices[k] = -1;
        }

        for (int i = 0; i < numUnique; ++i) {
            unsigned long long key = uniqueKeys[i];
            int candBatch = static_cast<int>(key >> 32);
            if (candBatch != batchIdx) continue;

            float score = uniqueProbTotal[i];

            // Find current worst in top-K
            int minPos = 0;
            float minScore = topScores[0];
            for (int k = 1; k < beamWidth; ++k) {
                if (topScores[k] < minScore) {
                    minScore = topScores[k];
                    minPos = k;
                }
            }

            if (score > minScore) {
                topScores[minPos] = score;
                topIndices[minPos] = i;
            }
        }

        for (int k = 0; k < beamWidth; ++k) {
            int sel = topIndices[k];
            int globalBeamIdx = batchIdx * beamWidth + k;

            if (sel >= 0) {
                probBlank[globalBeamIdx] = uniqueProbBlank[sel];
                probNonBlank[globalBeamIdx] = uniqueProbNonBlank[sel];
                probTotal[globalBeamIdx] = uniqueProbTotal[sel];

                unsigned long long key = uniqueKeys[sel];
                prefixHashes[globalBeamIdx] = key & 0xFFFFFFFFULL;
                lastTokens[globalBeamIdx] = uniqueLastToken[sel];

                int parent = uniqueParentIdx[sel];
                int token = uniqueToken[sel];

                int histIdx = timeStep * batchSize * beamWidth + globalBeamIdx;
                historyParents[histIdx] = parent;
                historyTokens[histIdx] = token;
            } else {
                probBlank[globalBeamIdx] = NEG_INF;
                probNonBlank[globalBeamIdx] = NEG_INF;
                probTotal[globalBeamIdx] = NEG_INF;
            }
        }
    }
}

void launchCTCBeamSearch(
    CTCBeamSearchState& state,
    const CTCBeamSearchConfig& config,
    const float* logProbs,
    const int* inputLengths,
    cudaStream_t stream
) {
    int numBeams = config.batchSize * config.beamWidth;
    int numCandidates = numBeams * config.numClasses;
    
    int threads = 256;
    int blocks = (numBeams + threads - 1) / threads;
    initializeCTCBeamSearchKernel<<<blocks, threads, 0, stream>>>(
        state.probBlank, state.probNonBlank, state.probTotal,
        state.prefixHashes, state.currentLengths, state.lastTokens,
        config.batchSize, config.beamWidth, config.blankId
    );

    for (int t = 0; t < config.maxTime; ++t) {
        int expBlocks = (numCandidates + threads - 1) / threads;
        expandCTCBeamsParallelKernel<<<expBlocks, threads, 0, stream>>>(
            state.probBlank, state.probNonBlank, state.prefixHashes, state.lastTokens,
            logProbs + (long long)t * config.batchSize * config.numClasses,
            inputLengths,
            state.candKeys, state.candProbBlank, state.candProbNonBlank,
            state.candParentIdx, state.candToken, state.candLastToken,
            config.batchSize, config.beamWidth, config.numClasses, config.blankId, t
        );

        thrust::counting_iterator<int> iter(0);
        thrust::copy(thrust::cuda::par.on(stream), iter, iter + numCandidates, state.candIndicesSorted);
        
        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned long long>(state.candKeys),
            thrust::device_ptr<unsigned long long>(state.candKeys + numCandidates),
            thrust::device_ptr<int>(state.candIndicesSorted)
        );

        auto prob_zip = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float>(state.candProbBlank),
            thrust::device_ptr<float>(state.candProbNonBlank)
        ));
        
        auto permuted_probs = thrust::make_permutation_iterator(
            prob_zip,
            thrust::device_ptr<int>(state.candIndicesSorted)
        );

        auto unique_prob_zip = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float>(state.uniqueProbBlank),
            thrust::device_ptr<float>(state.uniqueProbNonBlank)
        ));

        auto new_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned long long>(state.candKeys),
            thrust::device_ptr<unsigned long long>(state.candKeys + numCandidates),
            permuted_probs,
            thrust::device_ptr<unsigned long long>(state.uniqueKeys),
            unique_prob_zip,
            thrust::equal_to<unsigned long long>(),
            ProbZipReduce()
        );
        
        int numUnique = new_end.first - thrust::device_ptr<unsigned long long>(state.uniqueKeys);

        thrust::counting_iterator<int> idx_it(0);
        thrust::copy(thrust::cuda::par.on(stream),
                     idx_it,
                     idx_it + numCandidates,
                     state.candIndicesSorted);

        auto unique_end_idx = thrust::unique_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned long long>(state.candKeys),
            thrust::device_ptr<unsigned long long>(state.candKeys + numCandidates),
            thrust::device_ptr<int>(state.candIndicesSorted)
        );

        thrust::gather(
             thrust::cuda::par.on(stream),
             thrust::device_ptr<int>(state.candIndicesSorted),
             thrust::device_ptr<int>(state.candIndicesSorted) + numUnique,
             thrust::device_ptr<int>(state.candToken),
             thrust::device_ptr<int>(state.uniqueToken)
        );

        thrust::gather(
             thrust::cuda::par.on(stream),
             thrust::device_ptr<int>(state.candIndicesSorted),
             thrust::device_ptr<int>(state.candIndicesSorted) + numUnique,
             thrust::device_ptr<int>(state.candLastToken),
             thrust::device_ptr<int>(state.uniqueLastToken)
        );
        
        thrust::gather(
             thrust::cuda::par.on(stream),
             thrust::device_ptr<int>(state.candIndicesSorted),
             thrust::device_ptr<int>(state.candIndicesSorted) + numUnique,
             thrust::device_ptr<int>(state.candParentIdx),
             thrust::device_ptr<int>(state.uniqueParentIdx) 
        );

        thrust::transform(
            thrust::cuda::par.on(stream),
            unique_prob_zip,
            unique_prob_zip + numUnique,
            thrust::device_ptr<float>(state.uniqueProbTotal),
            CalcTotalProb()
        );

        size_t sharedMemSize = static_cast<size_t>(config.beamWidth) *
                               (sizeof(float) + sizeof(int));
        updateHistoryKernel<<<config.batchSize, 1, sharedMemSize, stream>>>(
            state.uniqueParentIdx, state.uniqueToken, state.uniqueLastToken,
            state.uniqueProbBlank, state.uniqueProbNonBlank, state.uniqueProbTotal,
            state.uniqueKeys, numUnique,
            state.historyParents, state.historyTokens,
            state.probBlank, state.probNonBlank, state.probTotal,
            state.prefixHashes, state.lastTokens,
            config.batchSize, config.beamWidth, t, config.maxTime
        );
    }

    cudaMemcpyAsync(state.outputScores, state.probTotal, 
                    numBeams * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

__global__ void reconstructSequencesKernel(
    const int* historyParents,
    const int* historyTokens,
    int* outputSequences,
    int* outputLengths,
    int batchSize,
    int beamWidth,
    int maxTime,
    int maxOutputLength,
    int blankId
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * beamWidth) return;

    int beamIdx = idx % beamWidth;
    int batchIdx = idx / beamWidth;
    
    int currentBeam = beamIdx;
    int len = 0;
    int* myOutput = outputSequences + (batchIdx * beamWidth + beamIdx) * maxOutputLength;
    
    for (int t = maxTime - 1; t >= 0; t--) {
        int histIdx = t * batchSize * beamWidth + batchIdx * beamWidth + currentBeam;
        int token = historyTokens[histIdx];
        int parent = historyParents[histIdx];
        
        if (token != -1 && token != blankId) { 
             if (len < maxOutputLength) {
                 myOutput[len++] = token;
             }
        }
        currentBeam = parent;
        if (currentBeam < 0) break; 
    }
    
    for (int i = 0; i < len / 2; i++) {
        int tmp = myOutput[i];
        myOutput[i] = myOutput[len - 1 - i];
        myOutput[len - 1 - i] = tmp;
    }
    
    for (int i = len; i < maxOutputLength; i++) {
        myOutput[i] = -1;
    }
    
    outputLengths[idx] = len;
}

void launchReconstructSequences(CTCBeamSearchState& state, const CTCBeamSearchConfig& config, cudaStream_t stream) {
    int numBeams = config.batchSize * config.beamWidth;
    reconstructSequencesKernel<<<(numBeams+255)/256, 256, 0, stream>>>(
        state.historyParents, state.historyTokens,
        state.outputSequences, state.outputLengths,
        config.batchSize, config.beamWidth, config.maxTime, config.maxOutputLength,
        config.blankId
    );
}

cudaError_t allocateCTCBeamSearchState(CTCBeamSearchState& state, const CTCBeamSearchConfig& config) {
    int numBeams = config.batchSize * config.beamWidth;
    int numCandidates = numBeams * config.numClasses;
    
    cudaMalloc(&state.probBlank, numBeams * sizeof(float));
    cudaMalloc(&state.probNonBlank, numBeams * sizeof(float));
    cudaMalloc(&state.probTotal, numBeams * sizeof(float));
    cudaMalloc(&state.prefixHashes, numBeams * sizeof(unsigned long long));
    cudaMalloc(&state.currentLengths, numBeams * sizeof(int));
    cudaMalloc(&state.lastTokens, numBeams * sizeof(int));
    
    cudaMalloc(&state.historyParents, config.maxTime * numBeams * sizeof(int));
    cudaMalloc(&state.historyTokens, config.maxTime * numBeams * sizeof(int));
    
    cudaMalloc(&state.candKeys, numCandidates * sizeof(unsigned long long));
    cudaMalloc(&state.candProbBlank, numCandidates * sizeof(float));
    cudaMalloc(&state.candProbNonBlank, numCandidates * sizeof(float));
    cudaMalloc(&state.candParentIdx, numCandidates * sizeof(int));
    cudaMalloc(&state.candToken, numCandidates * sizeof(int));
    cudaMalloc(&state.candLastToken, numCandidates * sizeof(int));
    
    cudaMalloc(&state.candKeysSorted, numCandidates * sizeof(unsigned long long));
    cudaMalloc(&state.candIndicesSorted, numCandidates * sizeof(int));
    
    cudaMalloc(&state.uniqueKeys, numCandidates * sizeof(unsigned long long));
    cudaMalloc(&state.uniqueProbBlank, numCandidates * sizeof(float));
    cudaMalloc(&state.uniqueProbNonBlank, numCandidates * sizeof(float));
    cudaMalloc(&state.uniqueProbTotal, numCandidates * sizeof(float));
    cudaMalloc(&state.uniqueParentIdx, numCandidates * sizeof(int));
    cudaMalloc(&state.uniqueToken, numCandidates * sizeof(int));
    cudaMalloc(&state.uniqueLastToken, numCandidates * sizeof(int));
    
    cudaMalloc(&state.outputSequences, numBeams * config.maxOutputLength * sizeof(int));
    cudaMalloc(&state.outputLengths, numBeams * sizeof(int));
    cudaMalloc(&state.outputScores, numBeams * sizeof(float));
    
    return cudaSuccess;
}

void freeCTCBeamSearchState(CTCBeamSearchState& state) {
    cudaFree(state.probBlank);
    cudaFree(state.probNonBlank);
    cudaFree(state.probTotal);
    cudaFree(state.prefixHashes);
    cudaFree(state.currentLengths);
    cudaFree(state.lastTokens);
    cudaFree(state.historyParents);
    cudaFree(state.historyTokens);
    cudaFree(state.candKeys);
    cudaFree(state.candProbBlank);
    cudaFree(state.candProbNonBlank);
    cudaFree(state.candParentIdx);
    cudaFree(state.candToken);
    cudaFree(state.candLastToken);
    cudaFree(state.candKeysSorted);
    cudaFree(state.candIndicesSorted);
    cudaFree(state.uniqueKeys);
    cudaFree(state.uniqueProbBlank);
    cudaFree(state.uniqueProbNonBlank);
    cudaFree(state.uniqueProbTotal);
    cudaFree(state.uniqueParentIdx);
    cudaFree(state.uniqueToken);
    cudaFree(state.uniqueLastToken);
    cudaFree(state.outputSequences);
    cudaFree(state.outputLengths);
    cudaFree(state.outputScores);
}
