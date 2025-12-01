#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <sstream>
#include <string>

#include "ctc_beam_search_cuda_kernel.cuh"
#define DEFAULT_STREAM 0

namespace { // to avoid collisions

void validate_inputs(
    const torch::Tensor& log_probs,
    const torch::Tensor& input_lengths
) {

    TORCH_CHECK(log_probs.defined(), "log_probs must be a valid tensor");

    TORCH_CHECK(
        log_probs.is_cuda(),
        "log_probs must be a CUDA tensor (got device ",
        log_probs.device(),
        ")"
    );

    TORCH_CHECK(
        log_probs.scalar_type() == torch::kFloat32,
        "log_probs must be float32 (got ",
        log_probs.scalar_type(),
        ")"
    );

    TORCH_CHECK(
        log_probs.dim() == 3,
        "log_probs must have shape (batch, time, vocab); got ",
        log_probs.sizes()
    );

    TORCH_CHECK(input_lengths.defined(), "input_lengths must be a valid tensor");

    TORCH_CHECK(
        input_lengths.dim() == 1,
        "input_lengths must be 1-D (batch), got ",
        input_lengths.sizes()
    );

    TORCH_CHECK(
        input_lengths.size(0) == log_probs.size(0),
        "input_lengths must have the same batch dimension as log_probs (expected ",
        log_probs.size(0),
        ", got ",
        input_lengths.size(0),
        ")"
    );

    TORCH_CHECK(
        input_lengths.scalar_type() == torch::kInt32 ||
            input_lengths.scalar_type() == torch::kInt64,
        "input_lengths must be int32 or int64 (got ",
        input_lengths.scalar_type(),
        ")"
    );

    torch::Tensor lengths_cpu = input_lengths.to(torch::kCPU, torch::kInt64);
    auto min_len = lengths_cpu.min().item<int64_t>();
    auto max_len = lengths_cpu.max().item<int64_t>();

    TORCH_CHECK(min_len > 0, "input_lengths must be strictly positive (min=", min_len, ")");

    TORCH_CHECK(
        max_len <= log_probs.size(1),
        "input_lengths cannot exceed time dimension (max=",
        max_len,
        ", T=",
        log_probs.size(1),
        ")"
    );
}

}  // namespace to avoid collisions

void ctc_hello(torch::Tensor log_probs, torch::Tensor input_lengths) {
    validate_inputs(log_probs, input_lengths);

    const int batch = static_cast<int>(log_probs.size(0));
    const int time = static_cast<int>(log_probs.size(1));
    const int vocab = static_cast<int>(log_probs.size(2));

    launch_ctc_hello_kernel(batch, time, vocab, DEFAULT_STREAM);

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ctc_hello",
        &ctc_hello,
        "CTC hello-world kernel call (validates inputs first)"
    );
}
