"""
synthetic input generation
"""

from __future__ import annotations

import torch

def generate_test_inputs(
    batch_size: int,
    time_steps: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    torch.manual_seed(seed)
    logits = torch.randn(batch_size, time_steps, vocab_size, device=device)
    log_probs = torch.log_softmax(logits, dim=-1)

    # keep as 25 for now
    min_len = max(25, time_steps // 3)

    input_lengths = torch.randint(
        low=min_len,
        high=time_steps + 1,
        size=(batch_size,),
        dtype=torch.int64,
        device=device,
    )

    return log_probs, input_lengths

