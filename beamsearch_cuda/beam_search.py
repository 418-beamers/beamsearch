from pathlib import Path
from typing import Tuple, NamedTuple
from enum import IntEnum
import math
import torch
from torch.utils.cpp_extension import load


_MODULE = None

# to mirror cpp interface
class SchedulerType(IntEnum):
    NAIVE = 0
    LUT = 1
    MLP = 2


class BeamSchedule(NamedTuple):
    adaptive_beam_width: bool = False
    scheduler_type: SchedulerType = SchedulerType.NAIVE
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    min: int = 0
    init: int = 0
    init_steps: int = 0
    lut_path: str = ""
    mlp_path: str = ""


def _load_ctc_extension():
    global _MODULE

    if _MODULE is None:
        base_dir = Path(__file__).resolve().parent
        src_dir = base_dir / "src" / "ctc"
        scheduler_dir = base_dir / "src" / "scheduler"

        sources = [
            src_dir / "interface.cpp",
            src_dir / "beam_search.cu",
            src_dir / "kernels" / "initialize.cu",
            src_dir / "kernels" / "expand.cu",
            src_dir / "kernels" / "top_k.cu",
            src_dir / "kernels" / "reconstruct.cu",
            scheduler_dir / "scheduler.cpp",
            scheduler_dir / "mlp" / "mlp_decay_scheduler.cpp",
        ]

        _MODULE = load(
            name="beamsearch_cuda_native",
            sources=[str(path) for path in sources],
            extra_include_paths=[str(src_dir), str(scheduler_dir), str(scheduler_dir / "mlp")],
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
        schedule: BeamSchedule = None,
    ):
        if schedule is None:
            schedule = BeamSchedule()

        self.beam_width = beam_width
        self.num_classes = num_classes
        self.max_output_length = max_output_length
        self.blank_id = blank_id
        self.batch_size = batch_size
        self.max_time = max_time
        self.schedule = schedule
        self._ext = _load_ctc_extension()

        batch_bits = math.ceil(math.log2(batch_size)) if batch_size > 1 else 1
        hash_bits = 32 - batch_bits
        
        if hash_bits < 1:
             raise ValueError(f"Batch size {batch_size} is too large to fit in 32-bit key with any hash bits left.")

        schedule_config = self._ext.BeamSchedule()
        schedule_config.adaptive_beam_width = schedule.adaptive_beam_width
        schedule_config.scheduler_type = self._ext.SchedulerType(int(schedule.scheduler_type))
        schedule_config.a = schedule.a
        schedule_config.b = schedule.b
        schedule_config.c = schedule.c
        schedule_config.min = schedule.min
        schedule_config.init = schedule.init
        schedule_config.init_steps = schedule.init_steps
        schedule_config.lut_path = schedule.lut_path
        schedule_config.mlp_path = schedule.mlp_path

        self.state_ptr = self._ext.create(
            batch_size,
            beam_width,
            num_classes,
            max_time,
            max_output_length,
            blank_id,
            batch_bits,
            hash_bits,
            schedule_config,
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

    def get_beam_width_history(self) -> list:
        return self._ext.get_beam_width_history(self.state_ptr)

    def get_entropy_history(self) -> list:
        return self._ext.get_entropy_history(self.state_ptr)

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
    schedule: BeamSchedule = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_time, num_classes = log_probs.shape

    decoder = CTCBeamSearchDecoder(
        beam_width=beam_width,
        num_classes=num_classes,
        max_output_length=max_time,
        blank_id=blank_id,
        batch_size=batch_size,
        max_time=max_time,
        schedule=schedule,
    )

    return decoder.decode(log_probs, input_lengths)

def ctc_beam_search(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    beam_width: int,
    blank_idx: int,
    top_k: int,
    schedule: BeamSchedule = None,
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
        schedule=schedule,
    )

    sequences, _, scores = decoder.decode(log_probs, prepared_lengths)

    sorted_indices = scores.argsort(dim=1, descending=True) 
    
    top_k_indices = sorted_indices[:, :top_k] 
    
    top_scores = scores.gather(1, top_k_indices)
    
    B, Beam, L = sequences.shape
    top_k_indices_expanded = top_k_indices.unsqueeze(2).expand(-1, -1, L)
    hypotheses = sequences.gather(1, top_k_indices_expanded)
    
    return hypotheses, top_scores

def generate_lut(output_path: str, time_resolution: int = 100, entropy_bins: int = 50, max_entropy: float = 10.0) -> bool:
    module = _load_ctc_extension()
    return module.generate_lut(output_path, time_resolution, entropy_bins, max_entropy)

__all__ = ["CTCBeamSearchDecoder", "ctc_beam_search_decode", "ctc_beam_search", "BeamSchedule", "SchedulerType", "generate_lut"]
