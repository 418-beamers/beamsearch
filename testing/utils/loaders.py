"""
module loading for extensions
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from torch.utils.cpp_extension import load as load_extension

# caching for better performance
_cuda_decoder_module: Any | None = None
_hello_extension: Any | None = None


def get_project_root() -> Path:
    # project root
    return Path(__file__).resolve().parent.parent.parent


def load_candidate_module():

    global _cuda_decoder_module

    if _cuda_decoder_module is None:
        try:
            from beamsearch_cuda import beam_search as module
        except ModuleNotFoundError:
            module = None

        _cuda_decoder_module = module

    return _cuda_decoder_module


def load_hello_extension() -> Any:
    
    global _hello_extension

    if _hello_extension is None:
        project_root = get_project_root()
        src_dir = project_root / "beamsearch_cuda" / "src" / "hello"
        sources = [src_dir / "binding.cpp", src_dir / "ctc_beam_search_cuda.cu"]
        _hello_extension = load_extension(
            name="beamsearch_cuda_hello",
            sources=[str(path) for path in sources],
            extra_include_paths=[str(src_dir)],
            verbose=False,
        )

    return _hello_extension

