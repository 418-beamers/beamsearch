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
import json
from pathlib import Path
import sys
from datetime import datetime

import torch
from torchaudio.models.decoder import ctc_decoder
from rich.console import Console

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
    load_librispeech,
    run_acoustic_model,
    run_all_decoders,
    compute_benchmark_metrics,
    print_benchmark_table,
    print_sweep_summary,
)

console = Console()

DEFAULT_LUT_FILENAME = "scheduler_lut.bin"
DEFAULT_MLP_FILENAME = "mlp_weights.bin"
SOURCE_MLP_WEIGHTS = PROJECT_ROOT / "beamsearch_cuda" / "src" / "scheduler" / "mlp" / "mlp_weights.bin"

BENCHMARK_DEFAULTS = {
    "num_samples": 500,
    "batch_size": 32,
    "beam_size": 50,
    "beam_threshold": 20.0,
    "max_audio_duration": 15.0,
    "data_dir": "./data",
    "output_file": "benchmark_results.json",
    "lut_path": str(PROJECT_ROOT / "beamsearch_cuda" / "src" / "scheduler" / "decay.lut"),
    "mlp_path": str(PROJECT_ROOT / "beamsearch_cuda" / "src" / "scheduler" / "mlp" / "model.bin"),
}


def ensure_scheduler_binaries(scheduler_type: str, lut_path: str, mlp_path: str) -> tuple:
    import shutil
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    resolved_lut, resolved_mlp = lut_path, mlp_path

    if scheduler_type == "lut":
        if not lut_path:
            resolved_lut = str(BIN_DIR / DEFAULT_LUT_FILENAME)
            
        if not Path(resolved_lut).exists():
            print(f"LUT binary not found at {resolved_lut}, generating...")
            _generate_lut_binary(resolved_lut)
            
    elif scheduler_type == "mlp":
        if not mlp_path:
            resolved_mlp = str(BIN_DIR / DEFAULT_MLP_FILENAME)

        if not Path(resolved_mlp).exists():
            if SOURCE_MLP_WEIGHTS.exists():
                print(f"MLP weights not found at {resolved_mlp}, copying from {SOURCE_MLP_WEIGHTS}...")
                shutil.copy(SOURCE_MLP_WEIGHTS, resolved_mlp)

            else:
                raise FileNotFoundError(f"MLP weights not found at {resolved_mlp}")
    return resolved_lut, resolved_mlp

def _generate_lut_binary(output_path: str):
    candidate_module = load_candidate_module()

    if candidate_module is None or not hasattr(candidate_module, 'generate_lut'):
        raise RuntimeError("Cannot generate LUT: beamsearch_cuda extension not available")

    candidate_module.generate_lut(output_path, time_resolution=100, entropy_bins=50, max_entropy=10.0)

def parse_args():
    p = argparse.ArgumentParser(description="CTC Decoder Testing and Benchmarking")

    p.add_argument("--benchmark", action="store_true", help="Run LibriSpeech benchmark mode")
    p.add_argument("--real", action="store_true", help="Use real audio with Wav2Vec2")
    p.add_argument("--hello", action="store_true", help="Run hello-world CUDA extension test")

    p.add_argument("--batch-size", type=int, default=2, help="Batch size")
    p.add_argument("--beam-width", type=int, default=50, help="Beam width")
    p.add_argument("--beam-threshold", type=float, default=40.0, help="Beam threshold")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--verbose", action="store_true", help="Print decoded sequences")
    p.add_argument("-q", "--quiet", action="store_true", help="Minimal output")

    p.add_argument("--time-steps", type=int, default=120, help="Sequence length (synthetic)")
    p.add_argument("--vocab-size", type=int, default=32, help="Vocab size (synthetic)")
    p.add_argument("--top-k", type=int, default=3, help="Hypotheses per sample")
    p.add_argument("--timing-runs", type=int, default=3, help="Timing repetitions")
    p.add_argument("--candidate-device", type=str, default=None, help="Device for candidate decoder")

    p.add_argument("--audio-file", type=str, default=None, help="Audio file path")

    p.add_argument("-n", "--num-samples", type=int, default=BENCHMARK_DEFAULTS["num_samples"])
    p.add_argument("--max-duration", type=float, default=BENCHMARK_DEFAULTS["max_audio_duration"])
    p.add_argument("--data-dir", type=str, default=BENCHMARK_DEFAULTS["data_dir"])
    p.add_argument("-o", "--output", type=str, default=BENCHMARK_DEFAULTS["output_file"])
    p.add_argument("--decoders", nargs="+", choices=["cuda-beamsearch", "torchaudio-flashlight", "pyctcdecode", "pyctcdecode-batch"])

    p.add_argument("--sweep", choices=["beam_width", "batch_size"], help="Parameter to sweep")
    p.add_argument("--beam-min", type=int, default=10)
    p.add_argument("--beam-max", type=int, default=100)
    p.add_argument("--beam-step", type=int, default=10)
    p.add_argument("--batch-min", type=int, default=1)
    p.add_argument("--batch-max", type=int, default=32)
    p.add_argument("--batch-step", type=int, default=4)

    p.add_argument("--adaptive-beam-width", action="store_true", help="Enable adaptive beam width")
    p.add_argument("--scheduler-type", type=str, default="naive", choices=["naive", "lut", "mlp"])
    p.add_argument("--schedule-a", type=float, default=0.0)
    p.add_argument("--schedule-b", type=float, default=0.0)
    p.add_argument("--schedule-c", type=float, default=0.0)
    p.add_argument("--schedule-min", type=int, default=0)
    p.add_argument("--schedule-init", type=int, default=0)
    p.add_argument("--schedule-init-steps", type=int, default=0)
    p.add_argument("--lut-path", type=str, default=BENCHMARK_DEFAULTS["lut_path"])
    p.add_argument("--mlp-path", type=str, default=BENCHMARK_DEFAULTS["mlp_path"])
    p.add_argument("-d", "--debug", action="store_true", help="Print adaptive beam debug info")

    args = p.parse_args()

    if args.adaptive_beam_width:
        if args.schedule_init <= 0:
            p.error("--adaptive-beam-width requires --schedule-init > 0")
        if args.schedule_min <= 0:
            p.error("--adaptive-beam-width requires --schedule-min > 0")
        if args.schedule_min > args.schedule_init:
            p.error("--schedule-min must be <= --schedule-init")
        if args.schedule_init > args.beam_width:
            p.error("--schedule-init must be <= --beam-width")
        if args.scheduler_type == "naive" and args.schedule_a == 0 and args.schedule_b == 0 and args.schedule_c == 0:
            p.error("--scheduler-type naive requires at least one of --schedule-a/b/c to be non-zero")
        if args.scheduler_type in ("lut", "mlp"):
            try:
                args.lut_path, args.mlp_path = ensure_scheduler_binaries(
                    args.scheduler_type, args.lut_path, args.mlp_path
                )
            except FileNotFoundError as e:
                p.error(str(e))

    return args


def run_synthetic_mode(args, cpu, candidate_device):
    tokens = make_tokens(args.vocab_size)
    blank_idx = 0
    blank_token = tokens[blank_idx]

    log_probs_btv, input_lengths = generate_test_inputs(
        args.batch_size, args.time_steps, args.vocab_size, device=cpu, seed=args.seed
    )

    decoder = ctc_decoder(
        tokens=tokens, lexicon=None, lm=None, nbest=args.top_k,
        beam_size=args.beam_width, beam_threshold=args.beam_threshold,
        blank_token=blank_token, sil_token=blank_token, unk_word=blank_token,
    )

    emissions_cpu = log_probs_btv.to(dtype=torch.float32, device="cpu").contiguous()
    input_lengths_cpu = input_lengths.to(dtype=torch.int64, device="cpu").contiguous()

    ref_output, ref_timing = run_timed("reference", args.timing_runs, decoder, args=(emissions_cpu, input_lengths_cpu))
    ref_decoded_texts = format_reference_outputs(ref_output, tokens, blank_idx, args.top_k)
    timing_stats_list = [ref_timing]

    if args.verbose:
        print_decoder_outputs("reference", ref_decoded_texts)
        print("=" * 80)

    _run_candidate_decoder(
        args=args, candidate_device=candidate_device,
        log_probs_btv=log_probs_btv, input_lengths=input_lengths,
        tokens=tokens, blank_idx=blank_idx,
        ref_decoded_texts=ref_decoded_texts, timing_stats_list=timing_stats_list,
        format_ref_fn=format_reference_outputs, format_cand_fn=format_candidate_outputs,
    )

    print_timing_table(timing_stats_list)


def run_real_audio_mode(args, cpu, candidate_device):
    if not is_real_audio_available():
        print("ERROR: torchaudio required (pip install torchaudio)")
        return

    print("=" * 80)
    print("Real Audio Mode")
    print("=" * 80)

    model_device = candidate_device if candidate_device.type == "cuda" else cpu

    real_inputs = generate_real_audio_inputs(
        audio_path=args.audio_file, device=model_device,
        batch_size=args.batch_size, use_sample=(args.audio_file is None),
    )

    print(f"Audio: {real_inputs.audio_path}")
    print(f"Sample rate: {real_inputs.sample_rate} Hz")
    print(f"Shape: {real_inputs.log_probs.shape}")
    print("=" * 80)

    tokens = real_inputs.tokens
    blank_idx = real_inputs.blank_idx
    log_probs_btv = real_inputs.log_probs
    input_lengths = real_inputs.input_lengths

    decoder = ctc_decoder(
        tokens=tokens, lexicon=None, lm=None, nbest=args.top_k,
        beam_size=args.beam_width, beam_threshold=args.beam_threshold,
        blank_token=tokens[blank_idx], sil_token=tokens[blank_idx], unk_word=tokens[blank_idx],
    )

    emissions_cpu = log_probs_btv.to(dtype=torch.float32, device="cpu").contiguous()
    input_lengths_cpu = input_lengths.to(dtype=torch.int64, device="cpu").contiguous()

    ref_output, ref_timing = run_timed("reference", args.timing_runs, decoder, args=(emissions_cpu, input_lengths_cpu))
    ref_decoded_texts = format_reference_outputs_wav2vec2(ref_output, tokens, blank_idx, args.top_k)
    timing_stats_list = [ref_timing]

    if args.verbose:
        print_decoder_outputs("reference", ref_decoded_texts)
    else:
        print("Reference best hypothesis:")
        for i, texts in enumerate(ref_decoded_texts):
            if texts:
                print(f"  Sample {i}: \"{texts[0]}\"")
    print("=" * 80)

    _run_candidate_decoder(
        args=args, candidate_device=candidate_device,
        log_probs_btv=log_probs_btv, input_lengths=input_lengths,
        tokens=tokens, blank_idx=blank_idx,
        ref_decoded_texts=ref_decoded_texts, timing_stats_list=timing_stats_list,
        format_ref_fn=format_reference_outputs_wav2vec2, format_cand_fn=format_candidate_outputs_wav2vec2,
        is_real_audio=True,
    )

    print_timing_table(timing_stats_list)


def _run_candidate_decoder(
    args, candidate_device, log_probs_btv, input_lengths, tokens, blank_idx,
    ref_decoded_texts, timing_stats_list, format_ref_fn, format_cand_fn, is_real_audio=False,
):
    candidate_module = load_candidate_module()
    if not (candidate_module and hasattr(candidate_module, "CTCBeamSearchDecoder") and candidate_device.type == "cuda"):
        print("\nbeamsearch_cuda not available or device not CUDA - skipping candidate.")
        return

    CTCBeamSearchDecoder = candidate_module.CTCBeamSearchDecoder
    BeamSchedule = candidate_module.BeamSchedule
    SchedulerType = candidate_module.SchedulerType

    log_probs_candidate = log_probs_btv.to(candidate_device)
    input_lengths_candidate = input_lengths.to(device=candidate_device, dtype=torch.int32)
    B, T, V = log_probs_btv.shape

    scheduler_type_map = {"naive": SchedulerType.NAIVE, "lut": SchedulerType.LUT, "mlp": SchedulerType.MLP}

    schedule = BeamSchedule(
        adaptive_beam_width=args.adaptive_beam_width,
        scheduler_type=scheduler_type_map[args.scheduler_type],
        a=args.schedule_a, b=args.schedule_b, c=args.schedule_c,
        min=args.schedule_min, init=args.schedule_init, init_steps=args.schedule_init_steps,
        lut_path=args.lut_path, mlp_path=args.mlp_path,
    )

    try:
        candidate_decoder = CTCBeamSearchDecoder(
            beam_width=args.beam_width, num_classes=V, max_output_length=T,
            blank_id=blank_idx, batch_size=B, max_time=T, schedule=schedule,
        )

        (sequences, lengths, scores), candidate_timing = run_timed(
            "candidate", args.timing_runs, candidate_decoder.decode,
            args=(log_probs_candidate, input_lengths_candidate), sync_fn=torch.cuda.synchronize,
        )
        timing_stats_list.append(candidate_timing)

        if args.debug and args.adaptive_beam_width:
            beam_widths = candidate_decoder.get_beam_width_history()
            entropies = candidate_decoder.get_entropy_history()
            print("=" * 80)
            print("Adaptive Beam Width Debug:")
            for t, (bw, ent) in enumerate(zip(beam_widths, entropies)):
                print(f"  t={t:3d} beam={bw:4d} entropy={ent:.4f}")
            print("=" * 80)

        sorted_indices = scores.argsort(dim=1, descending=True)
        top_k_indices = sorted_indices[:, :args.top_k]
        top_lengths = lengths.gather(1, top_k_indices)
        top_k_indices_expanded = top_k_indices.unsqueeze(2).expand(-1, -1, sequences.shape[2])
        candidate_hypotheses = sequences.gather(1, top_k_indices_expanded).cpu()
        candidate_lengths = top_lengths.cpu()

    except NotImplementedError:
        print("Candidate raised NotImplementedError, skipping.")
        return

    candidate_decoded_texts = format_cand_fn(candidate_hypotheses, candidate_lengths, tokens, blank_idx, args.top_k)
    avg_distance = compute_avg_edit_distance(ref_decoded_texts, candidate_decoded_texts)
    if avg_distance is not None:
        candidate_timing["avg_edit_distance"] = avg_distance

    if args.verbose:
        print_decoder_outputs("candidate", candidate_decoded_texts)
        summarize_similarity(ref_decoded_texts, candidate_decoded_texts)
    elif is_real_audio:
        print("Candidate best hypothesis:")
        for i, texts in enumerate(candidate_decoded_texts):
            if texts:
                print(f"  Sample {i}: \"{texts[0]}\"")
        if avg_distance is not None:
            print(f"Avg edit distance: {avg_distance:.3f}")
        print("=" * 80)

def run_hello_mode(args):
    if not torch.cuda.is_available():
        print("CUDA required for hello-world extension")
        return
    tokens = make_tokens(args.vocab_size)
    log_probs_btv, input_lengths = generate_test_inputs(
        args.batch_size, args.time_steps, args.vocab_size, device=torch.device("cpu"), seed=args.seed
    )
    hello_ext = load_hello_extension()
    hello_ext.ctc_hello(log_probs_btv.cuda(), input_lengths.cuda().int())
    print("Hello-world CUDA extension executed successfully.")

def run_benchmark_mode(args):
    if not args.quiet:
        console.print("\n[bold cyan]CTC Decoder Benchmark[/bold cyan]\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        gpu_info = f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""
        console.print(f"Device: {device}{gpu_info}")
        console.print("\n[blue]Loading Wav2Vec2 model...[/blue]")

    from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
    
    bundle = WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    tokens = list(bundle.get_labels())
    blank_idx = tokens.index('-')

    if not args.quiet:
        console.print(f"[green]Model loaded (vocab: {len(tokens)})[/green]")
        console.print(f"\n[blue]Loading LibriSpeech ({args.num_samples} samples)...[/blue]")

    samples = load_librispeech(
        args.data_dir, args.num_samples, args.max_duration, bundle.sample_rate
    )

    if not args.quiet:
        console.print(f"[green]Loaded {len(samples)} samples[/green]")
        console.print("\n[blue]Running acoustic model...[/blue]")

    all_log_probs, all_lengths, all_refs, total_audio = run_acoustic_model(
        samples, model, device, args.batch_size
    )

    if not args.quiet:
        console.print(f"[green]Processed {len(all_refs)} samples ({total_audio:.1f}s audio)[/green]")

    schedule_config = {
        "adaptive": args.adaptive_beam_width,
        "type": args.scheduler_type,
        "a": args.schedule_a, "b": args.schedule_b, "c": args.schedule_c,
        "min": args.schedule_min,
        "init": args.schedule_init if args.schedule_init > 0 else args.beam_width,
        "init_steps": args.schedule_init_steps,
        "lut_path": args.lut_path, "mlp_path": args.mlp_path,
    }

    if args.sweep == "beam_width":
        sweep_values = list(range(args.beam_min, args.beam_max + 1, args.beam_step))
        sweep_param = "beam_width"

    elif args.sweep == "batch_size":
        sweep_values = list(range(args.batch_min, args.batch_max + 1, args.batch_step))
        sweep_param = "batch_size"

    else:
        sweep_values, sweep_param = [None], None

    all_sweep_results = []

    for sweep_val in sweep_values:
        current_beam = args.beam_width
        current_batch = args.batch_size

        if sweep_param == "beam_width":
            current_beam = sweep_val

            if not args.quiet:
                console.print(f"\n[bold magenta]Beam Width: {sweep_val}[/bold magenta]")

        elif sweep_param == "batch_size":

            current_batch = sweep_val
            all_log_probs, all_lengths, all_refs, total_audio = run_acoustic_model(
                samples, model, device, sweep_val
            )

            if not args.quiet:
                console.print(f"\n[bold magenta]Batch Size: {sweep_val}[/bold magenta]")

        if not args.quiet:
            console.print("[blue]Running decoders...[/blue]")

        results = run_all_decoders(
            all_log_probs, all_lengths, tokens, blank_idx,
            beam_size=current_beam, beam_threshold=args.beam_threshold,
            schedule_config=schedule_config, decoder_filter=args.decoders, quiet=args.quiet,
        )

        final_results = compute_benchmark_metrics(results, all_refs, total_audio)

        if not args.quiet:
            for name, r in final_results.items():
                speedup = r.get('speedup', 1.0)
                console.print(f"  [green]{name}: WER={r['wer']*100:.2f}% CER={r['cer']*100:.2f}% RTFx={r['rtfx']:.0f} Speedup={speedup:.2f}x[/green]")
            title = f"Results (beam={current_beam})" if sweep_param else "Results"
            print_benchmark_table(final_results, title=title)

        all_sweep_results.append({
            "beam_width": current_beam,
            "batch_size": current_batch,
            "results": final_results,
        })

    if sweep_param and not args.quiet:
        print_sweep_summary(all_sweep_results, sweep_param)

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "samples": len(samples),
            "beam_size": args.beam_width,
            "batch_size": args.batch_size,
            "adaptive": args.adaptive_beam_width,
            "schedule_type": args.scheduler_type if args.adaptive_beam_width else None,
        },
        "total_audio_seconds": total_audio,
        "sweep_param": sweep_param,
        "sweep_results": all_sweep_results if sweep_param else None,
        "results": all_sweep_results[-1]["results"] if all_sweep_results else {},
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    if not args.quiet:
        console.print(f"\n[dim]Saved: {args.output}[/dim]")
        console.print("[bold green]Done[/bold green]\n")

def main():
    args = parse_args()
    cpu = torch.device("cpu")
    candidate_device = torch.device(args.candidate_device) if args.candidate_device else cpu

    if args.hello:
        run_hello_mode(args)
    elif args.benchmark:
        run_benchmark_mode(args)
    elif args.real:
        run_real_audio_mode(args, cpu, candidate_device)
    else:
        run_synthetic_mode(args, cpu, candidate_device)

if __name__ == "__main__":
    main()
