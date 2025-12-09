"""
runner functions for CTC beam search benchmarking
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List

import torch

# shoutout Corey :) 
@dataclass
class DecoderResult:
    name: str
    hypotheses: List[str]
    time_seconds: float
    device: str

def tokens_to_text(indices: List[int], tokens: List[str], blank_idx: int) -> str:
    chars = []
    for idx in indices:
        if idx == blank_idx or idx < 0 or idx >= len(tokens):
            continue
        t = tokens[idx]
        if t == '|':
            chars.append(' ')
        elif t not in ('<s>', '</s>', '<pad>', '<unk>'):
            chars.append(t)

    return ''.join(chars).strip()

def run_torchaudio_flashlight(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    tokens: List[str],
    blank_idx: int,
    beam_size: int,
    beam_threshold: float = 20.0,
) -> DecoderResult | None:
    from torchaudio.models.decoder import ctc_decoder

    decoder = ctc_decoder(
        lexicon=None,
        tokens=tokens,
        blank_token=tokens[blank_idx],
        sil_token='|',
        nbest=1,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
    )

    start = time.perf_counter()
    results = decoder(log_probs.cpu().float(), lengths.cpu().int())
    elapsed = time.perf_counter() - start

    hypotheses = [
        tokens_to_text(r[0].tokens.tolist(), tokens, blank_idx) if r else ""
        for r in results
    ]

    return DecoderResult("torchaudio-flashlight", hypotheses, elapsed, "cpu")

def run_cuda_decoder(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    tokens: List[str],
    blank_idx: int,
    beam_size: int,
    schedule_config = None,
) -> DecoderResult | None:
    from beamsearch_cuda.beam_search import CTCBeamSearchDecoder, BeamSchedule, SchedulerType

    B, T, V = log_probs.shape

    if schedule_config and schedule_config.get("adaptive"):

        type_map = {"naive": SchedulerType.NAIVE, "lut": SchedulerType.LUT, "mlp": SchedulerType.MLP}
        schedule = BeamSchedule(
            adaptive_beam_width=True,
            scheduler_type=type_map[schedule_config["type"]],
            a=schedule_config["a"],
            b=schedule_config["b"],
            c=schedule_config["c"],
            min=schedule_config["min"],
            init=schedule_config["init"] if schedule_config["init"] > 0 else beam_size,
            init_steps=schedule_config["init_steps"],
            lut_path=schedule_config.get("lut_path", ""),
            mlp_path=schedule_config.get("mlp_path", ""),
        )
    else:
        schedule = BeamSchedule()

    decoder = CTCBeamSearchDecoder(
        beam_width=beam_size,
        num_classes=V,
        max_output_length=T,
        blank_id=blank_idx,
        batch_size=B,
        max_time=T,
        schedule=schedule,
    )

    lp = log_probs.cuda().float().contiguous()
    ln = lengths.cuda().int().contiguous()

    torch.cuda.synchronize()
    start = time.perf_counter()
    seqs, lens, scores = decoder.decode(lp, ln)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    best_idx = scores.argsort(dim=1, descending=True)[:, 0]
    seqs, lens = seqs.cpu(), lens.cpu()

    hypotheses = []
    for b in range(B):
        idx = best_idx[b].item()
        seq = seqs[b, idx, : int(lens[b, idx])].tolist()
        hypotheses.append(tokens_to_text(seq, tokens, blank_idx))

    return DecoderResult("cuda-beamsearch", hypotheses, elapsed, "cuda")

def run_pyctcdecode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    tokens: List[str],
    blank_idx: int,
    beam_size: int,
) -> DecoderResult | None:
    from pyctcdecode import build_ctcdecoder

    labels = ["" if i == blank_idx else (' ' if t == '|' else t) for i, t in enumerate(tokens)]
    decoder = build_ctcdecoder(labels=labels)

    lp = log_probs.cpu().numpy()
    ln = lengths.cpu().tolist()

    start = time.perf_counter()
    hypotheses = [
        decoder.decode(lp[b, : int(ln[b]), :], beam_width=beam_size).strip()
        for b in range(lp.shape[0])
    ]
    elapsed = time.perf_counter() - start

    return DecoderResult("pyctcdecode", hypotheses, elapsed, "cpu")

def run_pyctcdecode_parallel(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    tokens: List[str],
    blank_idx: int,
    beam_size: int,
) -> DecoderResult | None:

    from pyctcdecode import build_ctcdecoder
    import multiprocessing as mp

    labels = ["" if i == blank_idx else (' ' if t == '|' else t) for i, t in enumerate(tokens)]
    decoder = build_ctcdecoder(labels=labels)

    lp = log_probs.cpu().numpy()
    ln = lengths.cpu().tolist()
    logits_list = [lp[b, : int(ln[b]), :] for b in range(lp.shape[0])]

    start = time.perf_counter()

    # run on mp for some parallelism
    with mp.get_context("fork").Pool() as pool:
        hypotheses = decoder.decode_batch(pool=pool, logits_list=logits_list, beam_width=beam_size)
    elapsed = time.perf_counter() - start

    return DecoderResult("pyctcdecode-batch", [h.strip() for h in hypotheses], elapsed, "cpu-parallel")
