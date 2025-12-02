#include "beam_search.cuh"
#include "utils.cuh"
#include "kernels/initialize.cuh"
#include "kernels/expand.cuh"
#include "kernels/top_k.cuh"
#include "kernels/reconstruct.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <stdexcept>
#include <string>

CTCBeamSearch::CTCBeamSearch(const CTCBeamSearchConfig& config) : config_(config) {
    cudaError_t err = allocate_state();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CTC beam search state: " + std::string(cudaGetErrorString(err)));
    }
}

CTCBeamSearch::~CTCBeamSearch() {
    free_state();
}

void CTCBeamSearch::decode(const float* log_probs, const int* input_lengths, cudaStream_t stream) {
    initialize(stream);
    launch(log_probs, input_lengths, stream);
    reconstruct(stream);
    cudaMemcpyAsync(state_.output_scores, state_.prob_total, 
                    config_.batch_size * config_.beam_width * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

int* CTCBeamSearch::get_sequences() const {
    return state_.output_sequences;
}

int* CTCBeamSearch::get_lengths() const {
    return state_.output_lengths;
}

float* CTCBeamSearch::get_scores() const {
    return state_.output_scores;
}

void CTCBeamSearch::initialize(cudaStream_t stream) {
    int num_beams = config_.batch_size * config_.beam_width;
    int threads = 256;
    int blocks = (num_beams + threads - 1) / threads;

    ::initialize<<<blocks, threads, 0, stream>>>(state_, config_);
}

void CTCBeamSearch::launch(const float* log_probs, const int* input_lengths, cudaStream_t stream) {
    int num_beams = config_.batch_size * config_.beam_width;
    int num_candidates = num_beams * config_.num_classes;

    for (int t = 0; t < config_.max_time; ++t) {
        int threads = 256;
        int expBlocks = (num_candidates + threads - 1) / threads;
        ::expand<<<expBlocks, threads, 0, stream>>>(
            state_, config_,
            log_probs + (long long)t * config_.batch_size * config_.num_classes,
            input_lengths,
            t
        );

        thrust::counting_iterator<int> iter(0);
        thrust::copy(thrust::cuda::par.on(stream), iter, iter + num_candidates, state_.cand_indices_sorted);
        
        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned int>(state_.cand_keys),
            thrust::device_ptr<unsigned int>(state_.cand_keys + num_candidates),
            thrust::device_ptr<int>(state_.cand_indices_sorted)
        );

        auto perm_pb = thrust::make_permutation_iterator(
            thrust::device_ptr<float>(state_.cand_prob_blank),
            thrust::device_ptr<int>(state_.cand_indices_sorted)
        );
        
        auto perm_pnb = thrust::make_permutation_iterator(
            thrust::device_ptr<float>(state_.cand_prob_non_blank),
            thrust::device_ptr<int>(state_.cand_indices_sorted)
        );

        auto values_in = thrust::make_zip_iterator(thrust::make_tuple(
            perm_pb,
            perm_pnb,
            thrust::device_ptr<int>(state_.cand_indices_sorted)
        ));

        auto values_out = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float>(state_.unique_prob_blank),
            thrust::device_ptr<float>(state_.unique_prob_non_blank),
            thrust::device_ptr<int>(state_.unique_indices)
        ));

        auto new_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned int>(state_.cand_keys),
            thrust::device_ptr<unsigned int>(state_.cand_keys + num_candidates),
            values_in,
            thrust::device_ptr<unsigned int>(state_.unique_keys),
            values_out,
            thrust::equal_to<unsigned int>(),
            ProbAndIndexReduce()
        );
        
        int num_unique = new_end.first - thrust::device_ptr<unsigned int>(state_.unique_keys);

        thrust::gather(
             thrust::cuda::par.on(stream),
             thrust::device_ptr<int>(state_.unique_indices),
             thrust::device_ptr<int>(state_.unique_indices) + num_unique,
             thrust::device_ptr<int>(state_.cand_token),
             thrust::device_ptr<int>(state_.unique_token)
        );

        thrust::gather(
             thrust::cuda::par.on(stream),
             thrust::device_ptr<int>(state_.unique_indices),
             thrust::device_ptr<int>(state_.unique_indices) + num_unique,
             thrust::device_ptr<int>(state_.cand_last_token),
             thrust::device_ptr<int>(state_.unique_last_token)
        );
        
        thrust::gather(
             thrust::cuda::par.on(stream),
             thrust::device_ptr<int>(state_.unique_indices),
             thrust::device_ptr<int>(state_.unique_indices) + num_unique,
             thrust::device_ptr<int>(state_.cand_parent_idx),
             thrust::device_ptr<int>(state_.unique_parent_idx) 
        );

        auto unique_prob_zip = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float>(state_.unique_prob_blank),
            thrust::device_ptr<float>(state_.unique_prob_non_blank)
        ));

        thrust::transform(
            thrust::cuda::par.on(stream),
            unique_prob_zip,
            unique_prob_zip + num_unique,
            thrust::device_ptr<float>(state_.unique_prob_total),
            CalcTotalProb()
        );

        thrust::sequence(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique_indices),
            thrust::device_ptr<int>(state_.unique_indices) + num_unique
        );

        thrust::sort(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique_indices),
            thrust::device_ptr<int>(state_.unique_indices) + num_unique,
            BatchScoreComp(state_.unique_keys, state_.unique_prob_total, config_.hash_bits)
        );

        ::top_k<<<config_.batch_size, 256, 0, stream>>>(
            state_, config_, num_unique, t
        );
    }
}

void CTCBeamSearch::reconstruct(cudaStream_t stream) {
    int num_beams = config_.batch_size * config_.beam_width;
    int threads = 256;
    int blocks = (num_beams + threads - 1) / threads;
    ::reconstruct<<<blocks, threads, 0, stream>>>(state_, config_);
}

cudaError_t CTCBeamSearch::allocate_state() {
    int num_beams = config_.batch_size * config_.beam_width;
    int num_candidates = num_beams * config_.num_classes;
    
    cudaMalloc(&state_.prob_blank, num_beams * sizeof(float));
    cudaMalloc(&state_.prob_non_blank, num_beams * sizeof(float));
    cudaMalloc(&state_.prob_total, num_beams * sizeof(float));
    cudaMalloc(&state_.prefix_hashes, num_beams * sizeof(unsigned int));
    cudaMalloc(&state_.current_lengths, num_beams * sizeof(int));
    cudaMalloc(&state_.last_tokens, num_beams * sizeof(int));
    
    cudaMalloc(&state_.history_parents, config_.max_time * num_beams * sizeof(int));
    cudaMalloc(&state_.history_tokens, config_.max_time * num_beams * sizeof(int));
    
    cudaMalloc(&state_.cand_keys, num_candidates * sizeof(unsigned int));
    cudaMalloc(&state_.cand_prob_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.cand_prob_non_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.cand_parent_idx, num_candidates * sizeof(int));
    cudaMalloc(&state_.cand_token, num_candidates * sizeof(int));
    cudaMalloc(&state_.cand_last_token, num_candidates * sizeof(int));
    
    cudaMalloc(&state_.cand_keys_sorted, num_candidates * sizeof(unsigned int));
    cudaMalloc(&state_.cand_indices_sorted, num_candidates * sizeof(int));
    
    cudaMalloc(&state_.unique_keys, num_candidates * sizeof(unsigned int));
    cudaMalloc(&state_.unique_prob_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.unique_prob_non_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.unique_prob_total, num_candidates * sizeof(float));
    cudaMalloc(&state_.unique_parent_idx, num_candidates * sizeof(int));
    cudaMalloc(&state_.unique_token, num_candidates * sizeof(int));
    cudaMalloc(&state_.unique_last_token, num_candidates * sizeof(int));
    cudaMalloc(&state_.unique_indices, num_candidates * sizeof(int));
    
    cudaMalloc(&state_.output_sequences, num_beams * config_.max_output_length * sizeof(int));
    cudaMalloc(&state_.output_lengths, num_beams * sizeof(int));
    cudaMalloc(&state_.output_scores, num_beams * sizeof(float));
    
    return cudaSuccess;
}

void CTCBeamSearch::free_state() {
    cudaFree(state_.prob_blank);
    cudaFree(state_.prob_non_blank);
    cudaFree(state_.prob_total);
    cudaFree(state_.prefix_hashes);
    cudaFree(state_.current_lengths);
    cudaFree(state_.last_tokens);
    cudaFree(state_.history_parents);
    cudaFree(state_.history_tokens);
    cudaFree(state_.cand_keys);
    cudaFree(state_.cand_prob_blank);
    cudaFree(state_.cand_prob_non_blank);
    cudaFree(state_.cand_parent_idx);
    cudaFree(state_.cand_token);
    cudaFree(state_.cand_last_token);
    cudaFree(state_.cand_keys_sorted);
    cudaFree(state_.cand_indices_sorted);
    cudaFree(state_.unique_keys);
    cudaFree(state_.unique_prob_blank);
    cudaFree(state_.unique_prob_non_blank);
    cudaFree(state_.unique_prob_total);
    cudaFree(state_.unique_parent_idx);
    cudaFree(state_.unique_token);
    cudaFree(state_.unique_last_token);
    cudaFree(state_.unique_indices);
    cudaFree(state_.output_sequences);
    cudaFree(state_.output_lengths);
    cudaFree(state_.output_scores);
}

