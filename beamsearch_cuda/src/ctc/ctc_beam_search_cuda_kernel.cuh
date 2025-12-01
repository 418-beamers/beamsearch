#ifndef CTC_BEAM_SEARCH_CUDA_KERNEL_CUH 
#define CTC_BEAM_SEARCH_CUDA_KERNEL_CUH

#include <cuda_runtime.h>

void launch_ctc_hello_kernel(
    int batch_size,
    int time_steps,
    int vocab_size,
    cudaStream_t stream = 0
);

#endif //