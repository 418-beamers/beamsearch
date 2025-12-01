"""
CUDA-accelerated CTC Beam Search Decoder

This package provides a high-performance implementation of CTC beam search decoding
using CUDA for GPU acceleration.
"""

from .beam_search import CTCBeamSearchDecoder, ctc_beam_search_decode

__version__ = "1.0.0"
__all__ = ['CTCBeamSearchDecoder', 'ctc_beam_search_decode']
