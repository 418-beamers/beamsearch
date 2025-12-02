#include <torch/extension.h>
#include "beam_search.cuh"
#include <cuda_runtime.h>
#include <vector>

template<typename T>
T* get_device_ptr(torch::Tensor tensor) {
    return tensor.data_ptr<T>();
}

uintptr_t create(
    int batch_size,
    int beam_width,
    int num_classes,
    int max_time,
    int max_output_length,
    int blank_id
) {
    CTCBeamSearchConfig config;
    config.batch_size = batch_size;
    config.beam_width = beam_width;
    config.num_classes = num_classes;
    config.max_time = max_time;
    config.max_output_length = max_output_length;
    config.blank_id = blank_id;

    CTCBeamSearch* decoder = new CTCBeamSearch(config);
    return reinterpret_cast<uintptr_t>(decoder);
}

std::vector<torch::Tensor> decode(
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
    CTCBeamSearch* decoder = reinterpret_cast<CTCBeamSearch*>(state_ptr);

    float* log_probs_ptr = get_device_ptr<float>(log_probs);
    int* input_lengths_ptr = nullptr;

    if (input_lengths.defined()) {
        input_lengths_ptr = get_device_ptr<int>(input_lengths);
    }

    decoder->decode(log_probs_ptr, input_lengths_ptr, 0);
    cudaDeviceSynchronize();

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor sequences = torch::empty({batch_size, beam_width, max_output_length}, int_opts);
    torch::Tensor lengths = torch::empty({batch_size, beam_width}, int_opts);
    torch::Tensor scores = torch::empty({batch_size, beam_width}, float_opts);

    int num_beams = batch_size * beam_width;

    cudaMemcpy(get_device_ptr<int>(sequences), decoder->get_sequences(),
               num_beams * max_output_length * sizeof(int),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(get_device_ptr<int>(lengths), decoder->get_lengths(),
               num_beams * sizeof(int),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(get_device_ptr<float>(scores), decoder->get_scores(),
               num_beams * sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    return {sequences, lengths, scores};
}

void free_state(uintptr_t state_ptr) {
    CTCBeamSearch* decoder = reinterpret_cast<CTCBeamSearch*>(state_ptr);
    delete decoder;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create", &create, "Create state");
    m.def("decode", &decode, "Decode sequences");
    m.def("free_state", &free_state, "Free state");
}

