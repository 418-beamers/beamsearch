from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load

_CTC_EXTENSION = None

def _load_ctc_extension():

    # standard loading logic for cpp modules
    global _CTC_EXTENSION

    if _CTC_EXTENSION is None:

        base_dir = Path(__file__).resolve().parent
        src_dir = base_dir / "src" / "ctc"

        sources = [
            src_dir / "binding.cpp",
            src_dir / "ctc_beam_search_cuda.cu",
        ]

        _CTC_EXTENSION = load(
            name="ctc_beam_search_cuda",
            sources=[str(path) for path in sources],
            extra_include_paths=[str(src_dir)],
            verbose=False,
        )

    return _CTC_EXTENSION

# input validation
def _validate_inputs(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if log_probs.dim() != 3:
        raise ValueError(
            f"log_probs must have shape (batch, time, vocab), got {tuple(log_probs.shape)}"
        )

    if log_probs.device.type != "cuda":
        raise ValueError("log_probs must be on a CUDA device for the CTC extension")

    if log_probs.dtype != torch.float32:
        log_probs = log_probs.float()

    if input_lengths.dim() != 1:
        raise ValueError(
            f"input_lengths must be 1D with shape (batch,), got {tuple(input_lengths.shape)}"
        )
    if input_lengths.shape[0] != log_probs.shape[0]:
        raise ValueError(
            "input_lengths must match the batch dimension of log_probs "
            f"(expected {log_probs.shape[0]}, got {input_lengths.shape[0]})"
        )

    input_lengths = input_lengths.to(dtype=torch.int32, device=log_probs.device, non_blocking=True)
    log_probs = log_probs.contiguous()

    return log_probs, input_lengths.contiguous()

def ctc_beam_search(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    beam_width: int = 1,
    blank_idx: int = 0,
    top_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    temporary implementation that just calls a hello world kernel

    do input validation, 
    """

    log_probs, input_lengths = _validate_inputs(log_probs, input_lengths)
    ext = _load_ctc_extension()
    ext.ctc_hello(log_probs, input_lengths)

    batch, time, _ = log_probs.shape
    max_decoded_length = time

    hypotheses = torch.full(
        (batch, top_k, max_decoded_length),
        fill_value=blank_idx,
        dtype=torch.int32,
        device=log_probs.device,
    )

    scores = torch.full(
        (batch, top_k),
        fill_value=float("-inf"),
        dtype=log_probs.dtype,
        device=log_probs.device,
    )

    return hypotheses, scores