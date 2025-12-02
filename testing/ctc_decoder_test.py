"""
minimal testing harness to run the flashlight (https://arxiv.org/pdf/2201.12465) based ctc decoder 
as a reference to compare against our CUDA CTC Decoder 

currently we make a batch of (B,T,V) shaped log_probs to feed them through both decoders 
=> extension: add the ability to go from .wav to decoder 

timing: multi run mean, median, stdev 
similarity: levenshtein dist over normalized outputs
"""

from __future__ import annotations
import argparse
from pathlib import Path
import statistics
import sys
import time

import torch
from torchaudio.models.decoder import ctc_decoder
from torch.utils.cpp_extension import load as load_extension
from rich.console import Console
from rich.table import Table as RichTable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cuda_decoder_module = None
hello_extension = None

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
        "--timing-runs",
        type=int,
        default=3,
        help="number of timing repetitions per decoder",
    )

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

    # effectively a toolchain check at this point
    parser.add_argument(
        "--hello",
        action="store_true",
        help="run the hello-world CUDA extension instead of the beam search decoder",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print decoded sequences",
    )
    return parser.parse_args()


def load_candidate_module():
    global cuda_decoder_module

    if cuda_decoder_module is None:
        try:
            from beamsearch_cuda import beam_search as module
        except ModuleNotFoundError:
            module = None

        cuda_decoder_module = module

    return cuda_decoder_module

def load_hello_extension():
    global hello_extension

    if hello_extension is None:
        src_dir = PROJECT_ROOT / "beamsearch_cuda" / "src" / "hello"
        sources = [src_dir / "binding.cpp", src_dir / "ctc_beam_search_cuda.cu"]
        hello_extension = load_extension(
            name="beamsearch_cuda_hello",
            sources=[str(path) for path in sources],
            extra_include_paths=[str(src_dir)],
            verbose=False,
        )

    return hello_extension

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

# standardized format for similarity comp
def format_reference_outputs(ref_output, tokens, blank_idx, top_k):
    formatted = []

    for hypotheses in ref_output:
        sample_texts = []

        for h in hypotheses[:top_k]:
            seq = getattr(h, "tokens", [])
            sample_texts.append(detokenize(seq, tokens, blank_idx))

        formatted.append(sample_texts)

    return formatted

# standardized format for similarity comp
def format_candidate_outputs(candidate_sequences, candidate_lengths, tokens, blank_idx, top_k):
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

def print_decoder_outputs(label, decoded_texts):
    print("=" * 80)
    print(f"{label} decoder outputs:")

    for batch_idx, texts in enumerate(decoded_texts):
        print(f"sample {batch_idx}: {texts}")

def run_timed(label, runs, fn, args=None, sync_fn=None):
    if args is None:
        args = ()

    durations = []
    result = None

    for _ in range(max(1, runs)):
        
        # needed for our CUDA impl so timing is fair given async
        if sync_fn:
            sync_fn()

        start_time = time.perf_counter()
        result = fn(*args)

        # needed for our CUDA impl so timing is fair given async
        if sync_fn:
            sync_fn()

        end_time = time.perf_counter()
        durations.append(end_time - start_time)

    mean = statistics.mean(durations)
    median = statistics.median(durations)
    stdev = statistics.stdev(durations) if len(durations) > 1 else 0.0

    timing_stats = {
        "label": label,
        "mean": mean,
        "median": median,
        "stdev": stdev,
    }

    return result, timing_stats


def print_timing_table(timing_stats_list):
    if not timing_stats_list:
        return

    has_similarity = any(
        stats.get("avg_edit_distance") is not None for stats in timing_stats_list
    )

    console = Console()
    table = RichTable(title="Results Summary")
    table.add_column("Decoder", justify="left")
    table.add_column("Mean (s)", justify="right")
    table.add_column("Median (s)", justify="right")
    table.add_column("Std Dev (s)", justify="right")
    if has_similarity:
        table.add_column("Avg Edit Dist", justify="right")

    for stats in timing_stats_list:
        avg_dist = stats.get("avg_edit_distance")
        row = [
            str(stats.get("label", "")),
            f"{stats.get('mean', 0.0):.4f}",
            f"{stats.get('median', 0.0):.4f}",
            f"{stats.get('stdev', 0.0):.4f}",
        ]
        if has_similarity:
            row.append("" if avg_dist is None else f"{avg_dist:.3f}")
        table.add_row(*row)

    console.print(table)

# standard implementation for similarity metrics: 
# https://www.geeksforgeeks.org/dsa/introduction-to-levenshtein-distance/ 
def levenshtein_distance(tokens_a, tokens_b):
    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    previous_row = list(range(len(tokens_b) + 1))

    for i, token_a in enumerate(tokens_a, start=1):
        current_row = [i]

        for j, token_b in enumerate(tokens_b, start=1):
            cost = 0 if token_a == token_b else 1
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[j] + 1,
                    previous_row[j - 1] + cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]

def normalized_similarity(tokens_a, tokens_b):
    max_len = max(len(tokens_a), len(tokens_b), 1)

    if max_len == 0:
        return 1.0

    distance = levenshtein_distance(tokens_a, tokens_b)

    return 1.0 - (distance / max_len)

def longest_common_prefix_length(tokens_a, tokens_b):
    length = 0

    for token_a, token_b in zip(tokens_a, tokens_b):
        if token_a != token_b:
            break
        length += 1

    return length

# compute average edit distance between best paths
def compute_avg_edit_distance(reference_decodings, candidate_decodings):
    if not candidate_decodings:
        return None

    total_distance = 0.0
    sample_count = 0

    for ref_texts, cand_texts in zip(reference_decodings, candidate_decodings):
        ref_texts = ref_texts or [""]
        cand_texts = cand_texts or [""]

        ref_best = ref_texts[0]
        cand_best = cand_texts[0]

        ref_best_tokens = ref_best.split()
        cand_best_tokens = cand_best.split()

        distance = levenshtein_distance(ref_best_tokens, cand_best_tokens)
        total_distance += distance
        sample_count += 1

    if sample_count == 0:
        return None

    return total_distance / sample_count


# nice summary printout fn 
def summarize_similarity(reference_decodings, candidate_decodings):
    avg_distance = compute_avg_edit_distance(reference_decodings, candidate_decodings)
    if avg_distance is None:
        return None

    print("=" * 80)
    print("decoder similarity summary:")
    print(f"avg_edit_distance={avg_distance:.3f}")
    return avg_distance

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

    ref_output, ref_timing = run_timed(
        "reference",
        args.timing_runs,
        decoder,
        args=(emissions_cpu, input_lengths_cpu),
    )

    ref_decoded_texts = format_reference_outputs(
        ref_output, tokens, blank_idx, args.top_k
    )
    timing_stats_list = [ref_timing]

    if args.verbose:
        print_decoder_outputs("reference", ref_decoded_texts)
        print("=" * 80)
    if args.hello:
        if not torch.cuda.is_available():

            print("CUDA is required for the hello-world extension.")
            print("=" * 80)
            print_timing_table(timing_stats_list)
            return

        hello_ext = load_hello_extension()

        log_probs_candidate = log_probs_btv.to("cuda")
        input_lengths_candidate = input_lengths.to(device="cuda", dtype=torch.int32)
        
        hello_ext.ctc_hello(log_probs_candidate, input_lengths_candidate)

        print("hello-world CUDA extension executed.")
        print("=" * 80)
        print_timing_table(timing_stats_list)
        return

    candidate_module = load_candidate_module()

    if (
        candidate_module
        and hasattr(candidate_module, "CTCBeamSearchDecoder")
        and candidate_device.type == "cuda"
    ):

        CTCBeamSearchDecoder = candidate_module.CTCBeamSearchDecoder

        log_probs_candidate = log_probs_btv.to(candidate_device)
        input_lengths_candidate = input_lengths.to(device=candidate_device, dtype=torch.int32)

        try:
            candidate_decoder = CTCBeamSearchDecoder(
                beam_width=args.beam_width,
                num_classes=args.vocab_size,
                max_output_length=args.time_steps,
                blank_id=blank_idx,
                batch_size=args.batch_size,
                max_time=args.time_steps,
            )

            (sequences, lengths, scores), candidate_timing = run_timed(
                "candidate",
                args.timing_runs,
                candidate_decoder.decode,
                args=(log_probs_candidate, input_lengths_candidate),
                sync_fn=torch.cuda.synchronize,
            )

            timing_stats_list.append(candidate_timing)

            sorted_indices = scores.argsort(dim=1, descending=True)
            top_k_indices = sorted_indices[:, :args.top_k]
            top_lengths = lengths.gather(1, top_k_indices)
            

            B, _, L = sequences.shape
            top_k_indices_expanded = top_k_indices.unsqueeze(2).expand(-1, -1, L)
            candidate_hypotheses = sequences.gather(1, top_k_indices_expanded)

        except NotImplementedError:
            print("candidate raised NotImplementedError, skip")
            return

        candidate_hypotheses = candidate_hypotheses.to("cpu")
        candidate_lengths = top_lengths.to("cpu")

        candidate_decoded_texts = format_candidate_outputs(
            candidate_hypotheses, candidate_lengths, tokens, blank_idx, args.top_k
        )

        avg_distance = compute_avg_edit_distance(ref_decoded_texts, candidate_decoded_texts)
        if avg_distance is not None:
            candidate_timing["avg_edit_distance"] = avg_distance

        if args.verbose:
            print_decoder_outputs("candidate", candidate_decoded_texts)
            summarize_similarity(ref_decoded_texts, candidate_decoded_texts)
    else:
        print(
            "\nbeamsearch_cuda not importable or missing `ctc_beam_search`, "
            "or candidate device is not CUDA skipping candidate comparison."
        )
    print_timing_table(timing_stats_list)

if __name__ == "__main__":
    main()
