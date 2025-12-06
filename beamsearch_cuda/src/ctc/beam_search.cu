#include "beam_search.cuh"
#include "utils.cuh"
#include "kernels/initialize.cuh"
#include "kernels/expand.cuh"
#include "kernels/top_k.cuh"
#include "kernels/reconstruct.cuh"
#include "../scheduler/scheduler.h"
#include "../scheduler/mlp/mlp_decay_scheduler.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

CTCBeamSearch::CTCBeamSearch(const CTCBeamSearchConfig& config) : config_(config) {
    cudaError_t err = allocate_state();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CTC beam search state: " + std::string(cudaGetErrorString(err)));
    }

    if (config_.schedule.adaptive_beam_width) {
        switch (config_.schedule.scheduler_type) {
            case SchedulerType::LUT: {
                lut_scheduler_ = std::make_unique<DecayScheduleGenerator>(0, 0, 0.0f);
                if (config_.schedule.lut_path.empty() ||
                    !lut_scheduler_->loadFromBinary(config_.schedule.lut_path)) {
                    throw std::runtime_error("LUT scheduler requires valid lut_path");
                }
                break;
            }
            case SchedulerType::MLP: {
                mlp_scheduler_ = std::make_unique<MLPDecayScheduler>();
                if (config_.schedule.mlp_path.empty() || 
                    !mlp_scheduler_->loadFromFile(config_.schedule.mlp_path)) {
                        throw std::runtime_error("MLP scheduler requires valid mlp_path");
                    }
                break;
            }
            case SchedulerType::NAIVE:
            default:
                break;
        }
    }
}

CTCBeamSearch::~CTCBeamSearch() {
    free_state();
}

void CTCBeamSearch::decode(const float* log_probs, const int* input_lengths, cudaStream_t stream) {
    initialize(stream);
    launch(log_probs, input_lengths, stream);
    reconstruct(stream);
    cudaMemcpyAsync(state_.output.scores, state_.beam.score_total, 
                    config_.batch_size * config_.beam_width * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

int* CTCBeamSearch::get_sequences() const {
    return state_.output.sequences;
}

int* CTCBeamSearch::get_lengths() const {
    return state_.output.lengths;
}

float* CTCBeamSearch::get_scores() const {
    return state_.output.scores;
}

void CTCBeamSearch::initialize(cudaStream_t stream) {
    int num_beams = config_.batch_size * config_.beam_width;
    int threads = 256;
    int blocks = (num_beams + threads - 1) / threads;

    ::initialize<<<blocks, threads, 0, stream>>>(state_, config_);
}

int CTCBeamSearch::compute_beam_width(int t, float current_entropy) {
    int beam_width;

    if (!config_.schedule.adaptive_beam_width) {
        return config_.beam_width;
    }

    if (t < config_.schedule.init_steps) {
        return config_.schedule.init;
    }

    switch (config_.schedule.scheduler_type) {
        case SchedulerType::LUT: {
                Params params = lut_scheduler_->query(
                    t,
                    entropy_history_,
                    config_.schedule.min,
                    config_.beam_width
                );
                float dt = (float)(t - config_.schedule.init_steps);
                float w = params.A * expf(params.B * dt) + params.C;
                beam_width = (int)w;
                break;
            }
        case SchedulerType::MLP: {
                int input_size = mlp_scheduler_->getInputSize();
                std::vector<float> mlp_input(input_size);

                Params params = mlp_scheduler_->query(
                    mlp_input,
                    (float)config_.schedule.min,
                    (float)config_.beam_width
                );
                float dt = (float)(t - config_.schedule.init_steps);
                float w = params.A * expf(params.B * dt) + params.C;
                beam_width = (int)w;
                break;
            }
        case SchedulerType::NAIVE:
        default: {
            float dt = (float)(t - config_.schedule.init_steps);
            float w = config_.schedule.a * expf(-config_.schedule.b * dt) + config_.schedule.c;
            beam_width = (int)w;
            break;
        }
    }

    if (beam_width < config_.schedule.min) beam_width = config_.schedule.min;
    if (beam_width > config_.beam_width)  beam_width = config_.beam_width;

    return beam_width;
}

void CTCBeamSearch::launch(const float* log_probs, const int* input_lengths, cudaStream_t stream) {
    int current_beam_width;
    entropy_history_.clear();
    entropy_history_.reserve(config_.max_time);

    if (config_.schedule.adaptive_beam_width) {
        current_beam_width = config_.schedule.init;
    } else {
        current_beam_width = config_.beam_width;
    }
    
    for (int t = 0; t < config_.max_time; ++t) {
        float current_entropy = 0.0f;

        // compute average entropy across all batches
        if (config_.schedule.adaptive_beam_width) {
            const float* timestep_log_probs = log_probs + (long long)t * config_.batch_size * config_.num_classes;
            int num_elements = config_.batch_size * config_.num_classes;
            
            auto entropy_iter = thrust::make_transform_iterator(
                thrust::device_ptr<const float>(timestep_log_probs),
                CalcEntropyContribution()
            );
            
            float total_entropy = thrust::reduce(
                thrust::cuda::par.on(stream),
                entropy_iter,
                entropy_iter + num_elements,
                0.0f,
                thrust::plus<float>()
            );
            
            current_entropy = total_entropy / (float)config_.batch_size;
        }
        
        entropy_history_.push_back(current_entropy);
        current_beam_width = compute_beam_width(t, current_entropy); 

        int num_active_beams = config_.batch_size * current_beam_width;
        int num_active_candidates = num_active_beams * config_.num_classes;

        int threads = 256;
        int expBlocks = (num_active_candidates + threads - 1) / threads;

        ::expand<<<expBlocks, threads, 0, stream>>>(
            state_, config_,
            log_probs + (long long)t * config_.batch_size * config_.num_classes,
            input_lengths,
            t,
            current_beam_width
        );

        thrust::counting_iterator<int> iter(0);
        thrust::copy(thrust::cuda::par.on(stream), iter, iter + num_active_candidates, state_.cand.indices_sorted);
        
        // sorting by key places duplicate hypotheses together (to be deduplicated) 
        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned int>(state_.cand.keys),
            thrust::device_ptr<unsigned int>(state_.cand.keys + num_active_candidates),
            thrust::device_ptr<int>(state_.cand.indices_sorted)
        );
        
        // here-207: pre-processing for reduce_by_key
        auto perm_pb = thrust::make_permutation_iterator(
            thrust::device_ptr<float>(state_.cand.score_blank),
            thrust::device_ptr<int>(state_.cand.indices_sorted)
        );
        
        auto perm_pnb = thrust::make_permutation_iterator(
            thrust::device_ptr<float>(state_.cand.score_non_blank),
            thrust::device_ptr<int>(state_.cand.indices_sorted)
        );

        auto values_in = thrust::make_zip_iterator(thrust::make_tuple(
            perm_pb,
            perm_pnb,
            thrust::device_ptr<int>(state_.cand.indices_sorted)
        ));

        auto values_out = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float>(state_.unique.score_blank),
            thrust::device_ptr<float>(state_.unique.score_non_blank),
            thrust::device_ptr<int>(state_.unique.indices)
        ));

        // merge duplicate hypotheses and summing their scores
        auto new_end = thrust::reduce_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<unsigned int>(state_.cand.keys),
            thrust::device_ptr<unsigned int>(state_.cand.keys + num_active_candidates),
            values_in,
            thrust::device_ptr<unsigned int>(state_.unique.keys),
            values_out,
            thrust::equal_to<unsigned int>(),
            ScoreAndIndexReduce()
        );
        
        int num_unique = new_end.first - thrust::device_ptr<unsigned int>(state_.unique.keys);


        // here-252: pre-processing for summing blank, non-blank scores 
        thrust::gather(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique.indices),
            thrust::device_ptr<int>(state_.unique.indices) + num_unique,
            thrust::device_ptr<int>(state_.cand.token),
            thrust::device_ptr<int>(state_.unique.token)
        );

        thrust::gather(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique.indices),
            thrust::device_ptr<int>(state_.unique.indices) + num_unique,
            thrust::device_ptr<int>(state_.cand.last_token),
            thrust::device_ptr<int>(state_.unique.last_token)
        );
        
        thrust::gather(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique.indices),
            thrust::device_ptr<int>(state_.unique.indices) + num_unique,
            thrust::device_ptr<int>(state_.cand.parent_idx),
            thrust::device_ptr<int>(state_.unique.parent_idx) 
        );

        auto unique_score_zip = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<float>(state_.unique.score_blank),
            thrust::device_ptr<float>(state_.unique.score_non_blank)
        ));

        // summing blank, non-blank scores to get total scores
        thrust::transform(
            thrust::cuda::par.on(stream),
            unique_score_zip,
            unique_score_zip + num_unique,
            thrust::device_ptr<float>(state_.unique.score_total),
            CalcTotalScore()
        );

        thrust::sequence(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique.indices),
            thrust::device_ptr<int>(state_.unique.indices) + num_unique
        );
        
        // sorting unique hypotheses within batch by total scores
        thrust::sort(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<int>(state_.unique.indices),
            thrust::device_ptr<int>(state_.unique.indices) + num_unique,
            BatchScoreComp(state_.unique.keys, state_.unique.score_total, config_.hash_bits)
        );

        ::top_k<<<config_.batch_size, 256, 0, stream>>>(
            state_, config_, num_unique, t, current_beam_width
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
    
    cudaMalloc(&state_.beam.score_blank, num_beams * sizeof(float));
    cudaMalloc(&state_.beam.score_non_blank, num_beams * sizeof(float));
    cudaMalloc(&state_.beam.score_total, num_beams * sizeof(float));
    cudaMalloc(&state_.beam.prefix_hashes, num_beams * sizeof(unsigned int));
    cudaMalloc(&state_.beam.current_lengths, num_beams * sizeof(int));
    cudaMalloc(&state_.beam.last_tokens, num_beams * sizeof(int));
    
    cudaMalloc(&state_.beam.history_parents, config_.max_time * num_beams * sizeof(int));
    cudaMalloc(&state_.beam.history_tokens, config_.max_time * num_beams * sizeof(int));
    
    cudaMalloc(&state_.cand.keys, num_candidates * sizeof(unsigned int));
    cudaMalloc(&state_.cand.score_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.cand.score_non_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.cand.parent_idx, num_candidates * sizeof(int));
    cudaMalloc(&state_.cand.token, num_candidates * sizeof(int));
    cudaMalloc(&state_.cand.last_token, num_candidates * sizeof(int));
    
    cudaMalloc(&state_.cand.keys_sorted, num_candidates * sizeof(unsigned int));
    cudaMalloc(&state_.cand.indices_sorted, num_candidates * sizeof(int));
    
    cudaMalloc(&state_.unique.keys, num_candidates * sizeof(unsigned int));
    cudaMalloc(&state_.unique.score_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.unique.score_non_blank, num_candidates * sizeof(float));
    cudaMalloc(&state_.unique.score_total, num_candidates * sizeof(float));
    cudaMalloc(&state_.unique.parent_idx, num_candidates * sizeof(int));
    cudaMalloc(&state_.unique.token, num_candidates * sizeof(int));
    cudaMalloc(&state_.unique.last_token, num_candidates * sizeof(int));
    cudaMalloc(&state_.unique.indices, num_candidates * sizeof(int));
    
    cudaMalloc(&state_.output.sequences, num_beams * config_.max_output_length * sizeof(int));
    cudaMalloc(&state_.output.lengths, num_beams * sizeof(int));
    cudaMalloc(&state_.output.scores, num_beams * sizeof(float));
    
    return cudaSuccess;
}

void CTCBeamSearch::free_state() {
    cudaFree(state_.beam.score_blank);
    cudaFree(state_.beam.score_non_blank);
    cudaFree(state_.beam.score_total);
    cudaFree(state_.beam.prefix_hashes);
    cudaFree(state_.beam.current_lengths);
    cudaFree(state_.beam.last_tokens);
    cudaFree(state_.beam.history_parents);
    cudaFree(state_.beam.history_tokens);
    cudaFree(state_.cand.keys);
    cudaFree(state_.cand.score_blank);
    cudaFree(state_.cand.score_non_blank);
    cudaFree(state_.cand.parent_idx);
    cudaFree(state_.cand.token);
    cudaFree(state_.cand.last_token);
    cudaFree(state_.cand.keys_sorted);
    cudaFree(state_.cand.indices_sorted);
    cudaFree(state_.unique.keys);
    cudaFree(state_.unique.score_blank);
    cudaFree(state_.unique.score_non_blank);
    cudaFree(state_.unique.score_total);
    cudaFree(state_.unique.parent_idx);
    cudaFree(state_.unique.token);
    cudaFree(state_.unique.last_token);
    cudaFree(state_.unique.indices);
    cudaFree(state_.output.sequences);
    cudaFree(state_.output.lengths);
    cudaFree(state_.output.scores);
}

