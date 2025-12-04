"""
timing utils for benchmarking
"""

from __future__ import annotations
import statistics
import time
from typing import Any, Callable

from rich.console import Console
from rich.table import Table as RichTable


def run_timed(
    label: str,
    runs: int,
    fn: Callable,
    args: tuple = None,
    sync_fn: Callable | None = None,
) -> tuple[Any, dict]:

    # multiple runs for aggregate stats
    if args is None:
        args = ()

    durations = []
    result = None

    for _ in range(max(1, runs)):
        # needed for CUDA impl so timing is fair given async
        if sync_fn:
            sync_fn()

        start_time = time.perf_counter()
        result = fn(*args)

        # needed for CUDA impl so timing is fair given async
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


def print_timing_table(timing_stats_list: list[dict]) -> None:
    # pretty printing :)

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

