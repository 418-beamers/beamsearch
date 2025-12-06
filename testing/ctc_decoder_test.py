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
TESTING_DIR = Path(__file__).resolve().parent
BIN_DIR = TESTING_DIR / "bin"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import all of utils
from utils import (
    run_timed,
    print_timing_table,
    compute_avg_edit_distance,
    summarize_similarity,
    make_tokens,
    format_reference_outputs,
    format_candidate_outputs,
    print_decoder_outputs,
    load_candidate_module,
    load_hello_extension,
    generate_test_inputs,
    is_real_audio_available,
    generate_real_audio_inputs,
    format_reference_outputs_wav2vec2,
    format_candidate_outputs_wav2vec2,
)

DEFAULT_LUT_FILENAME = "scheduler_lut.bin"
DEFAULT_MLP_FILENAME = "mlp_weights.bin"

SOURCE_MLP_WEIGHTS = PROJECT_ROOT / "beamsearch_cuda" / "src" / "scheduler" / "mlp" / "mlp_weights.bin"

def ensure_scheduler_binaries(scheduler_type: str, lut_path: str, mlp_path: str) -> tuple:
    import shutil
    
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    
    resolved_lut = lut_path
    resolved_mlp = mlp_path
    
    if scheduler_type == "lut":
        if not lut_path:
            resolved_lut = str(BIN_DIR / DEFAULT_LUT_FILENAME)
        
        if not Path(resolved_lut).exists():
            print(f"lut binary not found at {resolved_lut}, generating...")
            _generate_lut_binary(resolved_lut)
            
    elif scheduler_type == "mlp":
        if not mlp_path:
            resolved_mlp = str(BIN_DIR / DEFAULT_MLP_FILENAME)
        
        if not Path(resolved_mlp).exists():
            if SOURCE_MLP_WEIGHTS.exists():
                print(f"mlp weights not found at {resolved_mlp}, copying from {SOURCE_MLP_WEIGHTS}...")
                shutil.copy(SOURCE_MLP_WEIGHTS, resolved_mlp)
            else:
                raise FileNotFoundError(
                    f"mlp weights not found at {resolved_mlp} and 
                    source {SOURCE_MLP_WEIGHTS} doesn't exist, please generate"
                )
    
    return resolved_lut, resolved_mlp

def _generate_lut_binary(output_path: str):
    candidate_module = load_candidate_module()
    
    if candidate_module is None or not hasattr(candidate_module, 'generate_lut'):
        raise RuntimeError(
            "cannot generate lut: beamsearch_cuda extension not available or missing generate_lut"
        )

    candidate_module.generate_lut(
        output_path,
        time_resolution=100,
        entropy_bins=50,
        max_entropy=10.0
    )

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

    parser.add_argument(
        "--real",
        action="store_true",
        help="use real audio with pre-trained Wav2Vec2 ASR model instead of synthetic data",
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="path to audio file for real mode (if not provided, downloads a sample)",
    )

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
    
    parser.add_argument("--adaptive-beam-width", action="store_true", help="enable adaptive beam width")
    parser.add_argument("--scheduler-type", type=str, default="naive", choices=["naive", "lut", "mlp"], help="scheduler type")
    parser.add_argument("--schedule-a", type=float, default=0.0, help="schedule param a (NAIVE mode)")
    parser.add_argument("--schedule-b", type=float, default=0.0, help="schedule param b (NAIVE mode)")
    parser.add_argument("--schedule-c", type=float, default=0.0, help="schedule param c (NAIVE mode)")
    parser.add_argument("--schedule-min", type=int, default=0, help="schedule min beam width")
    parser.add_argument("--schedule-init", type=int, default=0, help="schedule initial beam width")
    parser.add_argument("--schedule-init-steps", type=int, default=0, help="steps before decay starts")
    parser.add_argument("--lut-path", type=str, default="", help="path to LUT binary (default: testing/bin/scheduler_lut.bin)")
    parser.add_argument("--mlp-path", type=str, default="", help="path to MLP weights (default: testing/bin/mlp_weights.bin)")

    args = parser.parse_args()
    
    # argument validation for adaptive beam width
    if args.adaptive_beam_width:
        if args.schedule_init <= 0:
            parser.error("--adaptive-beam-width requires --schedule-init > 0")
        if args.schedule_min <= 0:
            parser.error("--adaptive-beam-width requires --schedule-min > 0")
        if args.schedule_min > args.schedule_init:
            parser.error("--schedule-min must be <= --schedule-init")
        if args.schedule_init > args.beam_width:
            parser.error("--schedule-init must be <= --beam-width")
            
        if args.scheduler_type == "naive":
            if args.schedule_a == 0.0 and args.schedule_b == 0.0 and args.schedule_c == 0.0:
                parser.error(
                    "--scheduler-type naive requires at least one of "
                    "--schedule-a, --schedule-b, --schedule-c to be non-zero"
                )
        
        if args.scheduler_type in ("lut", "mlp"):
            try:
                args.lut_path, args.mlp_path = ensure_scheduler_binaries(
                    args.scheduler_type, args.lut_path, args.mlp_path
                )
            except FileNotFoundError as e:
                parser.error(str(e))
    else:
        scheduler_args_set = (
            args.scheduler_type != "naive" or
            args.schedule_a != 0.0 or args.schedule_b != 0.0 or args.schedule_c != 0.0 or
            args.schedule_min != 0 or args.schedule_init != 0 or args.schedule_init_steps != 0 or
            args.lut_path or args.mlp_path
        )
        if scheduler_args_set:
            print("WARNING: invalid arguments without --adaptive-beam-width enabled")
    
    return args


def run_synthetic_mode(args, cpu, candidate_device):

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

    # https://docs.pytorch.org/audio/master/generated/torchaudio.models.decoder.ctc_decoder.html
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

    emissions_cpu = log_probs_btv.to(dtype=torch.float32, device="cpu").contiguous()
    input_lengths_cpu = input_lengths.to(dtype=torch.int64, device="cpu").contiguous()

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

    _run_candidate_decoder(
        args=args,
        candidate_device=candidate_device,
        log_probs_btv=log_probs_btv,
        input_lengths=input_lengths,
        tokens=tokens,
        blank_idx=blank_idx,
        ref_decoded_texts=ref_decoded_texts,
        timing_stats_list=timing_stats_list,
        format_ref_fn=format_reference_outputs,
        format_cand_fn=format_candidate_outputs,
    )

    print_timing_table(timing_stats_list)


def run_real_audio_mode(args, cpu, candidate_device):

    if not is_real_audio_available():
        print("=" * 80)
        print("ERROR: need torchaudio to be installed (pip install torchaudio)")
        print("=" * 80)
        return
    
    print("=" * 80)
    print("Real Audio Mode")
    print("=" * 80)
    
    model_device = candidate_device if candidate_device.type == "cuda" else cpu
    
    print("Loading audio...")
    real_inputs = generate_real_audio_inputs(
        audio_path=args.audio_file,
        device=model_device,
        batch_size=args.batch_size,
        use_sample=(args.audio_file is None),
    )
    
    print(f"Audio file: {real_inputs.audio_path}")
    print(f"Sample rate: {real_inputs.sample_rate} Hz")
    print(f"Log probs shape: {real_inputs.log_probs.shape}")
    print(f"vocab size: {len(real_inputs.tokens)}")
    print("=" * 80)
    
    tokens = real_inputs.tokens
    blank_idx = real_inputs.blank_idx
    blank_token = tokens[blank_idx]
    
    log_probs_btv = real_inputs.log_probs
    input_lengths = real_inputs.input_lengths
    
    B, T, V = log_probs_btv.shape
    
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
    
    emissions_cpu = log_probs_btv.to(dtype=torch.float32, device="cpu").contiguous()
    input_lengths_cpu = input_lengths.to(dtype=torch.int64, device="cpu").contiguous()
    
    ref_output, ref_timing = run_timed(
        "reference",
        args.timing_runs,
        decoder,
        args=(emissions_cpu, input_lengths_cpu),
    )
    
    ref_decoded_texts = format_reference_outputs_wav2vec2(
        ref_output, tokens, blank_idx, args.top_k
    )
    timing_stats_list = [ref_timing]
    
    if args.verbose:
        print_decoder_outputs("reference", ref_decoded_texts)
        print("=" * 80)
    else:
        print("ref decoder best hypothesis:")
        for i, texts in enumerate(ref_decoded_texts):
            if texts:
                print(f"  Sample {i}: \"{texts[0]}\"")
        print("=" * 80)
    
    _run_candidate_decoder(
        args=args,
        candidate_device=candidate_device,
        log_probs_btv=log_probs_btv,
        input_lengths=input_lengths,
        tokens=tokens,
        blank_idx=blank_idx,
        ref_decoded_texts=ref_decoded_texts,
        timing_stats_list=timing_stats_list,
        format_ref_fn=format_reference_outputs_wav2vec2,
        format_cand_fn=format_candidate_outputs_wav2vec2,
        is_real_audio=True,
    )
    
    print_timing_table(timing_stats_list)


def _run_candidate_decoder(
    args,
    candidate_device,
    log_probs_btv,
    input_lengths,
    tokens,
    blank_idx,
    ref_decoded_texts,
    timing_stats_list,
    format_ref_fn,
    format_cand_fn,
    is_real_audio=False,
):

    candidate_module = load_candidate_module()
    
    if not (
        candidate_module
        and hasattr(candidate_module, "CTCBeamSearchDecoder")
        and candidate_device.type == "cuda"
    ):
        print(
            "\nbeamsearch_cuda not importable or missing `CTCBeamSearchDecoder`, "
            "or candidate device is not CUDA - skipping candidate comparison."
        )
        return

    CTCBeamSearchDecoder = candidate_module.CTCBeamSearchDecoder
    BeamSchedule = candidate_module.BeamSchedule

    log_probs_candidate = log_probs_btv.to(candidate_device)
    input_lengths_candidate = input_lengths.to(device=candidate_device, dtype=torch.int32)

    B, T, V = log_probs_btv.shape

    SchedulerType = candidate_module.SchedulerType
    scheduler_type_map = {"naive": SchedulerType.NAIVE, "lut": SchedulerType.LUT, "mlp": SchedulerType.MLP}
    
    schedule = BeamSchedule(
        adaptive_beam_width=args.adaptive_beam_width,
        scheduler_type=scheduler_type_map[args.scheduler_type],
        a=args.schedule_a,
        b=args.schedule_b,
        c=args.schedule_c,
        min=args.schedule_min,
        init=args.schedule_init,
        init_steps=args.schedule_init_steps,
        lut_path=args.lut_path,
        mlp_path=args.mlp_path,
    )

    try:
        candidate_decoder = CTCBeamSearchDecoder(
            beam_width=args.beam_width,
            num_classes=V,
            max_output_length=T,
            blank_id=blank_idx,
            batch_size=B,
            max_time=T,
            schedule=schedule,
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

    candidate_decoded_texts = format_cand_fn(
        candidate_hypotheses, candidate_lengths, tokens, blank_idx, args.top_k
    )

    avg_distance = compute_avg_edit_distance(ref_decoded_texts, candidate_decoded_texts)
    if avg_distance is not None:
        candidate_timing["avg_edit_distance"] = avg_distance
    
    if args.verbose:
        print_decoder_outputs("candidate", candidate_decoded_texts)
        summarize_similarity(ref_decoded_texts, candidate_decoded_texts)
    elif is_real_audio:
        print("Candidate decoder best hypothesis:")
        for i, texts in enumerate(candidate_decoded_texts):
            if texts:
                print(f"  Sample {i}: \"{texts[0]}\"")
        print("=" * 80)
        if avg_distance is not None:
            print(f"Average edit distance: {avg_distance:.3f}")
            print("=" * 80)


def run_hello_mode(args, log_probs_btv, input_lengths, timing_stats_list):

    if not torch.cuda.is_available():
        print("CUDA is required for the hello-world extension")
        print("=" * 80)
        print_timing_table(timing_stats_list)
        return
    
    hello_ext = load_hello_extension()
    
    log_probs_candidate = log_probs_btv.to("cuda")
    input_lengths_candidate = input_lengths.to(device="cuda", dtype=torch.int32)
    
    hello_ext.ctc_hello(log_probs_candidate, input_lengths_candidate)
    
    print("hello-world CUDA extension executed successfully.")
    print("=" * 80)
    print_timing_table(timing_stats_list)


def main():
    args = parse_args()

    cpu = torch.device("cpu")
    candidate_device = torch.device(args.candidate_device) if args.candidate_device else cpu

    # toolchain check
    if args.hello:
        tokens = make_tokens(args.vocab_size)
        log_probs_btv, input_lengths = generate_test_inputs(
            args.batch_size,
            args.time_steps,
            args.vocab_size,
            device=cpu,
            seed=args.seed,
        )
        
        blank_idx = 0
        blank_token = tokens[blank_idx]
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
        
        emissions_cpu = log_probs_btv.to(dtype=torch.float32, device="cpu").contiguous()
        input_lengths_cpu = input_lengths.to(dtype=torch.int64, device="cpu").contiguous()
        
        _, ref_timing = run_timed(
            "reference",
            args.timing_runs,
            decoder,
            args=(emissions_cpu, input_lengths_cpu),
        )
        
        run_hello_mode(args, log_probs_btv, input_lengths, [ref_timing])
        return

    if args.real:
        run_real_audio_mode(args, cpu, candidate_device)
    else:
        run_synthetic_mode(args, cpu, candidate_device)


if __name__ == "__main__":
    main()
