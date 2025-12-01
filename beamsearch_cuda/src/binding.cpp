#include <torch/extension.h>
#include "beam_search_cuda_kernel.cuh"
#include <cuda_runtime.h>

template<typename T>
T* get_device_ptr(torch::Tensor tensor) {
    return tensor.data_ptr<T>();
}

BeamSearchState allocate_beam_search_state(
    int batch_size,
    int beam_width,
    int vocab_size,
    int max_seq_length,
    int eos_token_id
) {
    BeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.vocabSize = vocab_size;
    config.maxSeqLength = max_seq_length;
    config.eosTokenId = eos_token_id;

    BeamSearchState state;
    cudaError_t err = allocateBeamSearchState(state, config);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to allocate beam search state: ") +
                                cudaGetErrorString(err));
    }

    return state;
}

void initialize_beam_search(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width,
    int vocab_size,
    int max_seq_length,
    int eos_token_id,
    int start_token
) {
    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);

    BeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.vocabSize = vocab_size;
    config.maxSeqLength = max_seq_length;
    config.eosTokenId = eos_token_id;

    launchInitializeBeamSearch(*state, config, start_token, 0);
    cudaDeviceSynchronize();
}

void beam_search_step(
    uintptr_t state_ptr,
    torch::Tensor decoder_probs,
    int batch_size,
    int beam_width,
    int vocab_size,
    int max_seq_length,
    int eos_token_id,
    int current_step
) {
    TORCH_CHECK(decoder_probs.is_cuda(), "decoder_probs must be a CUDA tensor");
    TORCH_CHECK(decoder_probs.dtype() == torch::kFloat32, "decoder_probs must be float32");

    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);

    BeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.vocabSize = vocab_size;
    config.maxSeqLength = max_seq_length;
    config.eosTokenId = eos_token_id;

    float* decoder_probs_ptr = get_device_ptr<float>(decoder_probs);

    launchBeamSearchStep(*state, config, decoder_probs_ptr, current_step, 0);
    cudaDeviceSynchronize();
}

bool check_all_beams_finished(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width,
    int vocab_size,
    int max_seq_length,
    int eos_token_id
) {
    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);

    BeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.vocabSize = vocab_size;
    config.maxSeqLength = max_seq_length;
    config.eosTokenId = eos_token_id;

    return checkAllBeamsFinished(*state, config);
}

torch::Tensor get_beam_sequences(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width,
    int max_seq_length
) {
    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA);

    torch::Tensor sequences = torch::empty({batch_size, beam_width, max_seq_length}, options);

    int* sequences_ptr = get_device_ptr<int>(sequences);
    int num_beams = batch_size * beam_width;

    cudaMemcpy(sequences_ptr, state->beamSequences,
               num_beams * max_seq_length * sizeof(int),
               cudaMemcpyDeviceToDevice);

    return sequences;
}

torch::Tensor get_beam_scores(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width
) {
    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({batch_size, beam_width}, options);

    float* scores_ptr = get_device_ptr<float>(scores);
    int num_beams = batch_size * beam_width;

    cudaMemcpy(scores_ptr, state->beamScores,
               num_beams * sizeof(float),
               cudaMemcpyDeviceToDevice);

    return scores;
}

torch::Tensor get_beam_lengths(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width
) {
    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA);

    torch::Tensor lengths = torch::empty({batch_size, beam_width}, options);

    int* lengths_ptr = get_device_ptr<int>(lengths);
    int num_beams = batch_size * beam_width;

    cudaMemcpy(lengths_ptr, state->beamLengths,
               num_beams * sizeof(int),
               cudaMemcpyDeviceToDevice);

    return lengths;
}

void free_beam_search_state(uintptr_t state_ptr) {
    BeamSearchState* state = reinterpret_cast<BeamSearchState*>(state_ptr);
    freeBeamSearchState(*state);
    delete state;
}

uintptr_t create_beam_search_state(
    int batch_size,
    int beam_width,
    int vocab_size,
    int max_seq_length,
    int eos_token_id
) {
    BeamSearchState* state = new BeamSearchState();

    BeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.vocabSize = vocab_size;
    config.maxSeqLength = max_seq_length;
    config.eosTokenId = eos_token_id;

    cudaError_t err = allocateBeamSearchState(*state, config);

    if (err != cudaSuccess) {
        delete state;
        throw std::runtime_error(std::string("Failed to allocate beam search state: ") +
                                cudaGetErrorString(err));
    }

    return reinterpret_cast<uintptr_t>(state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_beam_search_state", &create_beam_search_state,
          "Create and allocate beam search state");
    m.def("initialize_beam_search", &initialize_beam_search,
          "Initialize beam search with start token");
    m.def("beam_search_step", &beam_search_step,
          "Execute one beam search step");
    m.def("check_all_beams_finished", &check_all_beams_finished,
          "Check if all beams have finished");
    m.def("get_beam_sequences", &get_beam_sequences,
          "Get beam sequences as tensor");
    m.def("get_beam_scores", &get_beam_scores,
          "Get beam scores as tensor");
    m.def("get_beam_lengths", &get_beam_lengths,
          "Get beam lengths as tensor");
    m.def("free_beam_search_state", &free_beam_search_state,
          "Free beam search state");
}
