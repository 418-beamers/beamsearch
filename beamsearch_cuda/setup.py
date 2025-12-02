from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='beamsearch_cuda_native',
    ext_modules=[
        CUDAExtension(
            name='beamsearch_cuda_native',
            sources=[
                'src/ctc/interface.cpp',
                'src/ctc/beam_search.cu',
                'src/ctc/kernels/initialize.cu',
                'src/ctc/kernels/expand.cu',
                'src/ctc/kernels/top_k.cu',
                'src/ctc/kernels/reconstruct.cu',
            ],
            include_dirs=['src/ctc'],
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
