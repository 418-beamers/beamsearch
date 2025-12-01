#include "ctc_beam_search_cuda_kernel.cuh"
#include <cstdio>
#include <stdexcept>

namespace {

__global__ void ctc_hello_kernel(int batch_size, int time_steps, int vocab_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(
            "[CTC CUDA] hello world! batch=%d time=%d vocab=%d\n",
            batch_size,
            time_steps,
            vocab_size
        );
    }
}

}  // namespace to avoid collisions (for now)

void launch_ctc_hello_kernel(
    int batch_size,
    int time_steps,
    int vocab_size,
    cudaStream_t stream
) {

    dim3 blocks(1);
    dim3 threads(1);

    ctc_hello_kernel<<<blocks, threads, 0, stream>>>(batch_size, time_steps, vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("failed launching CTC hello kernel: ") + cudaGetErrorString(err)
        );
    }
}
