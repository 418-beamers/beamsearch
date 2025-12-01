#include <torch/extension.h>
#include "beam_search_cuda_kernel.cuh"
#include <cuda_runtime.h>

template<typename T>
T* get_device_ptr(torch::Tensor tensor) {
    return tensor.data_ptr<T>();
}

// Create and allocate CTC beam search state
uintptr_t create_ctc_beam_search_state(
    int batch_size,
    int beam_width,
    int num_classes,
    int max_time,
    int max_output_length,
    int blank_id
) {
    CTCBeamSearchState* state = new CTCBeamSearchState();

    CTCBeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.numClasses = num_classes;
    config.maxTime = max_time;
    config.maxOutputLength = max_output_length;
    config.blankId = blank_id;

    cudaError_t err = allocateCTCBeamSearchState(*state, config);

    if (err != cudaSuccess) {
        delete state;
        throw std::runtime_error(std::string("Failed to allocate CTC beam search state: ") +
                                cudaGetErrorString(err));
    }

    return reinterpret_cast<uintptr_t>(state);
}

// Initialize CTC beam search
void initialize_ctc_beam_search(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width,
    int num_classes,
    int max_time,
    int max_output_length,
    int blank_id
) {
    CTCBeamSearchState* state = reinterpret_cast<CTCBeamSearchState*>(state_ptr);

    CTCBeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.numClasses = num_classes;
    config.maxTime = max_time;
    config.maxOutputLength = max_output_length;
    config.blankId = blank_id;

    launchInitializeCTCBeamSearch(*state, config, 0);
    cudaDeviceSynchronize();
}

// Run CTC beam search
void run_ctc_beam_search(
    uintptr_t state_ptr,
    torch::Tensor log_probs,
    int batch_size,
    int beam_width,
    int num_classes,
    int max_time,
    int max_output_length,
    int blank_id,
    torch::Tensor input_lengths
) {
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be a CUDA tensor");
    TORCH_CHECK(log_probs.dtype() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(log_probs.dim() == 3, "log_probs must be 3D [batch_size, max_time, num_classes]");

    CTCBeamSearchState* state = reinterpret_cast<CTCBeamSearchState*>(state_ptr);

    CTCBeamSearchConfig config;
    config.batchSize = batch_size;
    config.beamWidth = beam_width;
    config.numClasses = num_classes;
    config.maxTime = max_time;
    config.maxOutputLength = max_output_length;
    config.blankId = blank_id;

    float* log_probs_ptr = get_device_ptr<float>(log_probs);
    int* input_lengths_ptr = nullptr;

    if (input_lengths.defined()) {
        TORCH_CHECK(input_lengths.is_cuda(), "input_lengths must be a CUDA tensor");
        TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths must be int32");
        input_lengths_ptr = get_device_ptr<int>(input_lengths);
    }

    launchCTCBeamSearch(*state, config, log_probs_ptr, input_lengths_ptr, 0);
    cudaDeviceSynchronize();
}

// Get output sequences
torch::Tensor get_sequences(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width,
    int max_output_length
) {
    CTCBeamSearchState* state = reinterpret_cast<CTCBeamSearchState*>(state_ptr);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA);

    torch::Tensor sequences = torch::empty({batch_size, beam_width, max_output_length}, options);

    int* sequences_ptr = get_device_ptr<int>(sequences);
    int num_beams = batch_size * beam_width;

    cudaMemcpy(sequences_ptr, state->prefixes,
               num_beams * max_output_length * sizeof(int),
               cudaMemcpyDeviceToDevice);

    return sequences;
}

// Get sequence lengths
torch::Tensor get_sequence_lengths(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width
) {
    CTCBeamSearchState* state = reinterpret_cast<CTCBeamSearchState*>(state_ptr);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA);

    torch::Tensor lengths = torch::empty({batch_size, beam_width}, options);

    int* lengths_ptr = get_device_ptr<int>(lengths);
    int num_beams = batch_size * beam_width;

    cudaMemcpy(lengths_ptr, state->prefixLengths,
               num_beams * sizeof(int),
               cudaMemcpyDeviceToDevice);

    return lengths;
}

// Get scores
torch::Tensor get_scores(
    uintptr_t state_ptr,
    int batch_size,
    int beam_width
) {
    CTCBeamSearchState* state = reinterpret_cast<CTCBeamSearchState*>(state_ptr);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor scores = torch::empty({batch_size, beam_width}, options);

    float* scores_ptr = get_device_ptr<float>(scores);
    int num_beams = batch_size * beam_width;

    cudaMemcpy(scores_ptr, state->probTotal,
               num_beams * sizeof(float),
               cudaMemcpyDeviceToDevice);

    return scores;
}

// Free state
void free_ctc_beam_search_state(uintptr_t state_ptr) {
    CTCBeamSearchState* state = reinterpret_cast<CTCBeamSearchState*>(state_ptr);
    freeCTCBeamSearchState(*state);
    delete state;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_ctc_beam_search_state", &create_ctc_beam_search_state,
          "Create and allocate CTC beam search state");
    m.def("initialize_ctc_beam_search", &initialize_ctc_beam_search,
          "Initialize CTC beam search with empty prefix");
    m.def("run_ctc_beam_search", &run_ctc_beam_search,
          "Run CTC beam search on log probabilities");
    m.def("get_sequences", &get_sequences,
          "Get output sequences");
    m.def("get_sequence_lengths", &get_sequence_lengths,
          "Get output sequence lengths");
    m.def("get_scores", &get_scores,
          "Get beam scores");
    m.def("free_ctc_beam_search_state", &free_ctc_beam_search_state,
          "Free CTC beam search state");
}
