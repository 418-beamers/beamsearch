#ifndef BEAM_SEARCH_CUDA_KERNEL_CUH
#define BEAM_SEARCH_CUDA_KERNEL_CUH

#include <cuda_runtime.h>

struct CTCBeamSearchConfig {
    int batchSize;
    int beamWidth;
    int numClasses;      
    int maxTime;         
    int maxOutputLength; 
    int blankId;         
};

struct CTCBeamSearchState {
    float* probBlank;       
    float* probNonBlank;    
    float* probTotal;       
    unsigned long long* prefixHashes; 
    int* currentLengths;    
    int* lastTokens;        

    int* historyParents;    
    int* historyTokens;     

    unsigned long long* candKeys;   
    float* candProbBlank;
    float* candProbNonBlank;
    int* candParentIdx;     
    int* candToken;         
    int* candLastToken;     
    
    unsigned long long* candKeysSorted;
    int* candIndicesSorted; 

    unsigned long long* uniqueKeys;
    float* uniqueProbBlank;
    float* uniqueProbNonBlank;
    float* uniqueProbTotal;
    int* uniqueParentIdx;
    int* uniqueToken;
    int* uniqueLastToken;
    
    int* outputSequences;   
    int* outputLengths;
    float* outputScores;
};

void launchInitializeCTCBeamSearch(CTCBeamSearchState& state, const CTCBeamSearchConfig& config, cudaStream_t stream = 0);

void launchCTCBeamSearch(CTCBeamSearchState& state, const CTCBeamSearchConfig& config, const float* logProbs, const int* inputLengths, cudaStream_t stream = 0);

cudaError_t allocateCTCBeamSearchState(CTCBeamSearchState& state, const CTCBeamSearchConfig& config);

void freeCTCBeamSearchState(CTCBeamSearchState& state);

void launchReconstructSequences(CTCBeamSearchState& state, const CTCBeamSearchConfig& config, cudaStream_t stream = 0);

#endif 
