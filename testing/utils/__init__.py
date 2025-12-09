"""
util for module access
"""

from .timing import run_timed, print_timing_table
from .similarity import (
    levenshtein_distance,
    normalized_similarity,
    longest_common_prefix_length,
    compute_avg_edit_distance,
    summarize_similarity,
    compute_wer_cer,
)
from .tokenization import (
    make_tokens,
    detokenize,
    format_reference_outputs,
    format_candidate_outputs,
    print_decoder_outputs,
)
from .loaders import load_candidate_module, load_hello_extension
from .inputs import generate_test_inputs
from .runners import (
    DecoderResult,
    tokens_to_text,
    run_torchaudio_flashlight,
    run_cuda_decoder,
    run_pyctcdecode,
    run_pyctcdecode_parallel,
)
from .benchmark import (
    load_librispeech,
    run_acoustic_model,
    run_all_decoders,
    compute_benchmark_metrics,
    print_benchmark_table,
    print_sweep_summary,
)

try:
    from .real_audio import (
        RealAudioInputs,
        generate_real_audio_inputs,
        format_reference_outputs_wav2vec2,
        format_candidate_outputs_wav2vec2,
        get_wav2vec2_tokens,
    )
    _REAL_AUDIO_AVAILABLE = True

except ImportError:
    _REAL_AUDIO_AVAILABLE = False
    RealAudioInputs = None
    generate_real_audio_inputs = None
    format_reference_outputs_wav2vec2 = None
    format_candidate_outputs_wav2vec2 = None
    get_wav2vec2_tokens = None


def is_real_audio_available() -> bool:
    return _REAL_AUDIO_AVAILABLE


__all__ = [
    "run_timed",
    "print_timing_table",
    "levenshtein_distance",
    "normalized_similarity",
    "longest_common_prefix_length",
    "compute_avg_edit_distance",
    "summarize_similarity",
    "compute_wer_cer",
    "make_tokens",
    "detokenize",
    "format_reference_outputs",
    "format_candidate_outputs",
    "print_decoder_outputs",
    "load_candidate_module",
    "load_hello_extension",
    "generate_test_inputs"
    "DecoderResult",
    "tokens_to_text",
    "run_torchaudio_flashlight",
    "run_cuda_decoder",
    "run_pyctcdecode",
    "run_pyctcdecode_parallel",
    "load_librispeech",
    "run_acoustic_model",
    "run_all_decoders",
    "compute_benchmark_metrics",
    "print_benchmark_table",
    "print_sweep_summary",
    "is_real_audio_available",
    "RealAudioInputs",
    "generate_real_audio_inputs",
    "format_reference_outputs_wav2vec2",
    "format_candidate_outputs_wav2vec2",
    "get_wav2vec2_tokens",
]
