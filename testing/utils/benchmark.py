"""
benchmark utils
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import torch
import torchaudio
from rich.console import Console
from rich.table import Table

from .similarity import compute_wer_cer
from .runners import (
    run_torchaudio_flashlight,
    run_cuda_decoder,
    run_pyctcdecode,
    run_pyctcdecode_parallel,
    clear_decoder_cache,
)

console = Console()

# dataset loading
def load_librispeech(
    data_dir: str,
    num_samples: int,
    max_duration: float,
    target_sample_rate: int,
    dataset_url: str = "test-other", # noisy data (test-clean is studio quality)
): 
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(data_dir, url=dataset_url, download=True)

    samples = []
    for i in range(len(dataset)):
        if len(samples) >= num_samples:
            break
        waveform, sr, transcript, *_ = dataset[i]
        duration = waveform.shape[1] / sr
        if duration <= max_duration:
            if sr != target_sample_rate:
                waveform = torchaudio.transforms.Resample(sr, target_sample_rate)(waveform)
            samples.append((waveform.squeeze(0), transcript, duration))

    return samples

def run_acoustic_model(
    samples,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    pad_to_max: bool = True,
): 
    acoustic_batch_size = min(batch_size, 32)
    
    all_log_probs, all_lengths, all_refs, total_audio = [], [], [], 0
    accumulated_emissions = []
    accumulated_lengths = []
    
    for i in range(0, len(samples), acoustic_batch_size):
        batch = samples[i : i + acoustic_batch_size]
        waves = [s[0] for s in batch]
        wave_lens = [w.shape[0] for w in waves]
        max_len = max(wave_lens)
        padded = torch.zeros(len(waves), max_len)
        for j, w in enumerate(waves):
            padded[j, : w.shape[0]] = w

        with torch.no_grad():
            padded = padded.to(device)
            emissions, lengths = model(padded)
            if lengths is None:
                lengths = torch.tensor(
                    [emissions.shape[1] * (wl / max_len) for wl in wave_lens],
                    device=device,
                ).long()
            log_probs = torch.log_softmax(emissions, dim=-1)

        accumulated_emissions.append(log_probs.cpu())
        accumulated_lengths.append(lengths.cpu())
        all_refs.extend([s[1] for s in batch])
        total_audio += sum(s[2] for s in batch)

    flat_emissions = [item for batch in accumulated_emissions for item in batch]
    flat_lengths = [item for batch in accumulated_lengths for item in batch]

    if flat_emissions:
        min_time = min(e.shape[0] for e in flat_emissions)
        max_time = max(e.shape[0] for e in flat_emissions)
        avg_time = sum(e.shape[0] for e in flat_emissions) / len(flat_emissions)
        vocab_size = flat_emissions[0].shape[1]
    else:
        min_time, max_time, avg_time, vocab_size = 0, 0, 0, 0

    num_batches = (len(flat_emissions) + batch_size - 1) // batch_size
    mode = "padded" if pad_to_max else "variable"
    console.print(f"[dim]Batches: {num_batches} (size={batch_size}), Seq lengths: min={min_time}, avg={avg_time:.0f}, max={max_time}, vocab={vocab_size} ({mode})[/dim]")

    if pad_to_max:
        for i in range(0, len(flat_emissions), batch_size):
            batch_emissions = flat_emissions[i : i + batch_size]
            batch_lens = flat_lengths[i : i + batch_size]
            
            current_max_time = max(e.shape[0] for e in batch_emissions)
            B = len(batch_emissions)
            V = batch_emissions[0].shape[1]
            
            padded_batch = torch.full((B, current_max_time, V), float('-inf'))
            padded_batch[:, :, 0] = 0 
            
            batch_len_tensor = torch.tensor([l.item() for l in batch_lens], dtype=torch.int32)
            
            for j, e in enumerate(batch_emissions):
                T = e.shape[0]
                padded_batch[j, :T, :] = e
            
            all_log_probs.append(padded_batch)
            all_lengths.append(batch_len_tensor)
    else:
        for i in range(0, len(flat_emissions), batch_size):
             batch_emissions = flat_emissions[i : i + batch_size]
             batch_lens = flat_lengths[i : i + batch_size]

    return all_log_probs, all_lengths, all_refs, total_audio


def run_all_decoders(
    all_log_probs: List[torch.Tensor],
    all_lengths: List[torch.Tensor],
    tokens: List[str],
    blank_idx: int,
    beam_size: int,
    beam_threshold: float = 20.0,
    schedule_config = None,
    decoder_filter = None,
    quiet: bool = False,
    cache_cuda_decoder: bool = True,
    prob_top_k: int = 40,
):
    all_decoders = {
        "cuda-beamsearch": lambda lp, ln: run_cuda_decoder(
            lp, ln, tokens, blank_idx, beam_size, schedule_config, use_cache=cache_cuda_decoder, prob_top_k=prob_top_k
        ),
        "torchaudio-flashlight": lambda lp, ln: run_torchaudio_flashlight(
            lp, ln, tokens, blank_idx, beam_size, beam_threshold
        ),
        "pyctcdecode": lambda lp, ln: run_pyctcdecode(lp, ln, tokens, blank_idx, beam_size),
        "pyctcdecode-batch": lambda lp, ln: run_pyctcdecode_parallel(
            lp, ln, tokens, blank_idx, beam_size
        ),
    }

    decoders = {k: v for k, v in all_decoders.items() if not decoder_filter or k in decoder_filter}
    results = {}

    for name, decoder_fn in decoders.items():
        results[name] = {"hyps": [], "time": 0, "device": None}
        for lp, ln in zip(all_log_probs, all_lengths):
            try:
                r = decoder_fn(lp, ln)
                if r:
                    results[name]["hyps"].extend(r.hypotheses)
                    results[name]["time"] += r.time_seconds
                    results[name]["device"] = r.device
                else:
                    results[name] = None
                    break

            except Exception as e:
                if not quiet:
                    console.print(f"  [yellow]{name}: {e}[/yellow]")
                results[name] = None
                break
    
    out = {k: v for k, v in results.items() if v is not None}
    return out


def compute_benchmark_metrics(
    results,
    all_refs,
    total_audio: float,
):
    final = {}
    for name, data in results.items():
        if len(data["hyps"]) != len(all_refs):
            continue
        wer, cer = compute_wer_cer(all_refs, data["hyps"])
        rtfx = total_audio / data["time"] if data["time"] > 0 else 0
        final[name] = {
            "wer": wer,
            "cer": cer,
            "rtfx": rtfx,
            "time": data["time"],
            "device": data["device"],
        }

    # compute speedup relative to slowest decoder
    if final:
        slowest_time = max(r["time"] for r in final.values())
        for name in final:
            final[name]["speedup"] = slowest_time / final[name]["time"] if final[name]["time"] > 0 else 0

    return final


def print_benchmark_table(final_results, title: str = "Results"):
    # petty printing :)
    table = Table(title=title)
    table.add_column("Decoder", style="cyan")
    table.add_column("Device")
    table.add_column("WER %", justify="right")
    table.add_column("CER %", justify="right")
    table.add_column("RTFx", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="magenta")
    table.add_column("Time (s)", justify="right")

    for name, r in sorted(final_results.items(), key=lambda x: x[1]["rtfx"]):
        speedup = r.get("speedup", 1.0)
        table.add_row(
            name,
            r["device"],
            f"{r['wer']*100:.2f}",
            f"{r['cer']*100:.2f}",
            f"{r['rtfx']:.0f}",
            f"{speedup:.2f}x",
            f"{r['time']:.2f}",
        )
    console.print(table)

def print_sweep_summary(
    all_sweep_results,
    sweep_param: str,
):
    console.print("\n[bold cyan]SWEEP SUMMARY[/bold cyan]")
    summary = Table(title=f"Sensitivity: {sweep_param}")
    summary.add_column(sweep_param.replace("_", " ").title(), style="cyan")

    decoder_names = list(all_sweep_results[0]["results"].keys())
    for name in decoder_names:
        summary.add_column(f"{name} RTFx", justify="right", style="green")
        summary.add_column(f"{name} Speedup", justify="right", style="magenta")
        summary.add_column(f"{name} WER%", justify="right")
        summary.add_column(f"{name} CER%", justify="right")

    for entry in all_sweep_results:
        row = [str(entry[sweep_param])]
        for name in decoder_names:
            if name in entry["results"]:    
                r = entry["results"][name]
                speedup = r.get("speedup", 1.0)
                row.extend([
                    f"{r['rtfx']:.0f}",
                    f"{speedup:.2f}x",
                    f"{r['wer']*100:.2f}",
                    f"{r['cer']*100:.2f}",
                ])
            else:
                row.extend(["N/A", "N/A", "N/A", "N/A"])
        summary.add_row(*row)

    console.print(summary)
