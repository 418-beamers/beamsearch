"""
minimal testing harness to run the flashlight (https://arxiv.org/pdf/2201.12465) based ctc decoder 
as a reference to compare against our CUDA CTC Decoder 

currently we make a batch of (B,T,V) shaped log_probs to feed them through both decoders 
=> extension: add the ability to go from .wav to decoder 
"""

from __future__ import annotations

import argparse
import torch
from torchaudio.models.decoder import ctc_decoder

try:
    from beamsearch_cuda import beam_search as cuda_decoder_module
except ModuleNotFoundError:
    cuda_decoder_module = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="run our CUDA-based decoder against torchaudio, flashlight-based reference"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("--time-steps", type=int, default=120, help="sequence length")
    parser.add_argument("--vocab-size", type=int, default=32, help="vocab size")
    parser.add_argument("--beam-width", type=int, default=50, help="beam width")
    parser.add_argument("--top-k", type=int, default=3, help="hypotheses to keep per input in batch")

    parser.add_argument(
        "--candidate-device",
        type=str,
        default=None,
        help="device for the candidate decoder (defaults to CPU).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="manual seed for reproducibility",
    )
    return parser.parse_args()

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

def make_tokens(vocab_size):
    tokens = ["<blank>"]
    for idx in range(1, vocab_size):
        tokens.append(f"t{idx:02d}")
    return tokens[:vocab_size]

def detokenize(sequence, tokens, blank_idx): 
    subwords = []

    for idx in sequence:
        if idx == blank_idx or idx is None or idx < 0 or idx >= len(tokens):
            continue
        subwords.append(tokens[idx])

    return " ".join(subwords)

def main():
    args = parse_args()

    cpu = torch.device("cpu")
    candidate_device = torch.device(args.candidate_device) if args.candidate_device else cpu

    tokens = make_tokens(args.vocab_size)
    blank_idx = 0
    blank_token = tokens[blank_idx]

    log_probs_btv, input_lengths = generate_test_inputs(
        args.batch_size,
        args.time_steps,
        args.vocab_size,
        device=cpu,
        seed=args.seed,
    )

    # https://docs.pytorch.org/audio/master/generated/torchaudio.models.decoder.ctc_decoder.html#torchaudio.models.decoder.ctc_decoder
    decoder = ctc_decoder(
        tokens=tokens,
        lexicon=None,
        lm=None,
        nbest=args.top_k,
        beam_size=args.beam_width,
        beam_threshold=40.0,
        blank_token=blank_token,
        sil_token=blank_token,
        unk_word=blank_token,
    )

    emissions_cpu = log_probs_btv.to(dtype=torch.float32, device="cpu").contiguous() # temp fix
    input_lengths_cpu = input_lengths.to(dtype=torch.int64, device="cpu").contiguous() # temp fix 

    B, T, V = log_probs_btv.shape

    assert input_lengths_cpu.dim() == 1 and input_lengths_cpu.numel() == B, (input_lengths_cpu.shape, B)
    assert int(input_lengths_cpu.min()) > 0, input_lengths_cpu
    assert int(input_lengths_cpu.max()) <= T, (int(input_lengths_cpu.max()), T)

    ref_output = decoder(emissions_cpu, input_lengths_cpu)

    print("="*80)
    print("reference decoder outputs:")
    for batch_idx, hypotheses in enumerate(ref_output):
        texts = [
            detokenize(h.tokens, tokens, blank_idx) if hasattr(h, "tokens") else ""
            for h in hypotheses[: args.top_k]
        ]
        print(f"sample {batch_idx}: {texts}")


    print("="*80)
    if cuda_decoder_module and hasattr(cuda_decoder_module, "ctc_beam_search"):
        candidate_fn = cuda_decoder_module.ctc_beam_search

        log_probs_candidate = log_probs_btv.to(candidate_device)
        input_lengths_candidate = input_lengths.to(candidate_device)

        try:
            candidate_hypotheses, _ = candidate_fn(
                log_probs=log_probs_candidate,
                input_lengths=input_lengths_candidate,
                beam_width=args.beam_width,
                blank_idx=blank_idx,
                top_k=args.top_k,
            )

        except NotImplementedError:
            print("candidate raised NotImplementedError, skip")
            return

        candidate_hypotheses = candidate_hypotheses.to("cpu")

        print("="*80)
        print("candidate decoder outputs:")

        for batch_idx, sample_hyps in enumerate(candidate_hypotheses):
            decoded = [
                detokenize(h.tolist(), tokens, blank_idx) for h in sample_hyps[: args.top_k]
            ]
            print(f"sample {batch_idx}: {decoded}")
    else:
        print(
            "\nbeamsearch_cuda not importable or missing `ctc_beam_search`; "
            "skipping candidate comparison."
        )
    print("="*80)

if __name__ == "__main__":
    main()
