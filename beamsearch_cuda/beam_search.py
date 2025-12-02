from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load

_MODULE = None

def _load_ctc_extension():
    global _MODULE

    if _MODULE is None:
        base_dir = Path(__file__).resolve().parent
        src_dir = base_dir / "src" / "ctc"

        sources = [
            src_dir / "interface.cpp",
            src_dir / "beam_search.cu",
            src_dir / "kernels" / "initialize.cu",
            src_dir / "kernels" / "expand.cu",
            src_dir / "kernels" / "top_k.cu",
            src_dir / "kernels" / "reconstruct.cu",
        ]

        _MODULE = load(
            name="beamsearch_cuda_native",
            sources=[str(path) for path in sources],
            extra_include_paths=[str(src_dir)],
            verbose=False,
        )

    return _MODULE


def _prepare_input_lengths(input_lengths: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:

    if input_lengths is None:
        return torch.empty(0, dtype=torch.int32, device=device)
    
    if input_lengths.ndim != 1 or input_lengths.numel() != batch_size:
        raise ValueError(f"input_lengths must have shape ({batch_size},)")

    if input_lengths.device != device:
        input_lengths = input_lengths.to(device)

    if input_lengths.dtype != torch.int32:
        input_lengths = input_lengths.to(torch.int32)

    return input_lengths.contiguous()

def _validate_log_probs(log_probs: torch.Tensor):

    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must have shape (batch, time, vocab), got {tuple(log_probs.shape)}")

    if not log_probs.is_cuda:
        raise ValueError("log_probs must be on a CUDA device")

    if log_probs.dtype != torch.float32:
        raise ValueError("log_probs must be float32")

class CTCBeamSearchDecoder:
    def __init__(
        self,
        beam_width: int,
        num_classes: int,
        max_output_length: int,
        blank_id: int = 0,
        batch_size: int = 1,
        max_time: int = 100,
    ):
        self.beam_width = beam_width
        self.num_classes = num_classes
        self.max_output_length = max_output_length
        self.blank_id = blank_id
        self.batch_size = batch_size
        self.max_time = max_time
        self._ext = _load_ctc_extension()

        self.state_ptr = self._ext.create(
            batch_size,
            beam_width,
            num_classes,
            max_time,
            max_output_length,
            blank_id,
        )

    def __del__(self):
        if hasattr(self, "state_ptr"):
            self._ext.free_state(self.state_ptr)

    def decode(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_log_probs(log_probs)

        batch_size, max_time, num_classes = log_probs.shape
        if batch_size != self.batch_size:
            raise ValueError(f"batch_size mismatch: expected {self.batch_size}, got {batch_size}")

        if max_time != self.max_time:
            raise ValueError(f"max_time mismatch: expected {self.max_time}, got {max_time}")

        if num_classes != self.num_classes:
            raise ValueError(f"num_classes mismatch: expected {self.num_classes}, got {num_classes}")

        log_probs = log_probs.permute(1, 0, 2).contiguous()
        prepared_lengths = _prepare_input_lengths(input_lengths, batch_size, log_probs.device)

        return self._ext.decode(
            self.state_ptr,
            log_probs,
            self.batch_size,
            self.beam_width,
            self.num_classes,
            self.max_time,
            self.max_output_length,
            self.blank_id,
            prepared_lengths,
        )

    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        sequences, lengths, _ = self.decode(log_probs, input_lengths)

        best_sequences = sequences[:, 0, :]
        best_lengths = lengths[:, 0]
        
        return best_sequences, best_lengths

def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 0,
    input_lengths: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_time, num_classes = log_probs.shape

    decoder = CTCBeamSearchDecoder(
        beam_width=beam_width,
        num_classes=num_classes,
        max_output_length=max_time,
        blank_id=blank_id,
        batch_size=batch_size,
        max_time=max_time,
    )

    return decoder.decode(log_probs, input_lengths)

def ctc_beam_search(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    beam_width: int,
    blank_idx: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _validate_log_probs(log_probs)

    if top_k > beam_width:
        raise ValueError("top_k cannot exceed beam_width")

    batch_size, max_time, num_classes = log_probs.shape
    prepared_lengths = _prepare_input_lengths(input_lengths, batch_size, log_probs.device)
    
    decoder = CTCBeamSearchDecoder(
        beam_width=beam_width,
        num_classes=num_classes,
        max_output_length=max_time,
        blank_id=blank_idx,
        batch_size=batch_size,
        max_time=max_time,
    )

    sequences, _, scores = decoder.decode(log_probs, prepared_lengths)

    sorted_indices = scores.argsort(dim=1, descending=True) 
    
    top_k_indices = sorted_indices[:, :top_k] 
    
    top_scores = scores.gather(1, top_k_indices)
    
    B, Beam, L = sequences.shape
    top_k_indices_expanded = top_k_indices.unsqueeze(2).expand(-1, -1, L)
    hypotheses = sequences.gather(1, top_k_indices_expanded)
    
    return hypotheses, top_scores

__all__ = ["CTCBeamSearchDecoder", "ctc_beam_search_decode", "ctc_beam_search"]
