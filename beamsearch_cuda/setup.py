from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='beamsearch_cuda_native',
    ext_modules=[
        CUDAExtension(
            name='beamsearch_cuda_native',
            sources=[
                'src/ctc/binding.cpp',
                'src/ctc/ctc_beam_search_cuda.cu',
                'src/ctc/kernels/initialize_ctc_beam_search.cu',
                'src/ctc/kernels/expand_beams.cu',
                'src/ctc/kernels/select_top_k.cu',
                'src/ctc/kernels/reconstruct_sequences.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--extended-lambda'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
