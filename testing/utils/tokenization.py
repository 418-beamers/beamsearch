"""
tokenization
"""

from __future__ import annotations
from typing import Any

import torch

# synthetic vocab
def make_tokens(vocab_size: int) -> list[str]:
    
    tokens = ["<blank>"]
    for idx in range(1, vocab_size):
        tokens.append(f"t{idx:02d}")
    return tokens[:vocab_size]

def detokenize(sequence: list[int], tokens: list[str], blank_idx: int) -> str:
    
    subwords = []

    for idx in sequence:
        if idx == blank_idx or idx is None or idx < 0 or idx >= len(tokens):
            continue
        subwords.append(tokens[idx])

    return " ".join(subwords)


def format_reference_outputs(
    ref_output: list,
    tokens: list[str],
    blank_idx: int,
    top_k: int,
) -> list[list[str]]:

    formatted = []

    for hypotheses in ref_output:
        sample_texts = []

        for h in hypotheses[:top_k]:
            seq = getattr(h, "tokens", [])
            sample_texts.append(detokenize(seq, tokens, blank_idx))

        formatted.append(sample_texts)

    return formatted


def format_candidate_outputs(
    candidate_sequences: torch.Tensor,
    candidate_lengths: torch.Tensor,
    tokens: list[str],
    blank_idx: int,
    top_k: int,
) -> list[list[str]]:
    
    formatted = []

    for sample_hyps, sample_lengths in zip(candidate_sequences, candidate_lengths):
        sample_texts = []

        for seq, length in zip(sample_hyps[:top_k], sample_lengths[:top_k]):
            valid_length = int(length)

            if valid_length < 0:
                valid_length = 0
            valid_length = min(valid_length, seq.numel())

            trimmed = seq[:valid_length]
            sample_texts.append(detokenize(trimmed.tolist(), tokens, blank_idx))

        formatted.append(sample_texts)

    return formatted

def print_decoder_outputs(label: str, decoded_texts: list[list[str]]) -> None:
    
    print("=" * 80)
    print(f"{label} decoder outputs:")

    for batch_idx, texts in enumerate(decoded_texts):
        print(f"sample {batch_idx}: {texts}")
