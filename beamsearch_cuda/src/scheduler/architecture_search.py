import torch
import torch.nn as nn
import torch.optim as optim
import struct
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json

INPUT_SIZE = 3
OUTPUT_SIZE = 3
MAX_ENTROPY = 10.0

HIDDEN_SIZES = [16, 12, 8, 6, 4, 3, 2, 1]

EPOCHS = 500
BATCH_SIZE = 1000
LEARNING_RATE = 0.01

NUM_BENCHMARK_ITERATIONS = 100000

ACCURACY_THRESHOLD_PCT = 5.0

SPEED_RATIO_TARGET = 2.0


@dataclass
class Params:
    A: float
    B: float
    C: float


@dataclass
class SearchResult:
    hidden_size: int
    mse_total: float
    mae_total: float
    query_time_ns: float
    lut_query_time_ns: float
    speed_ratio: float  
    memory_bytes: int
    training_loss: float


class BeamScheduleNet(nn.Module):
    def __init__(self, hidden_size: int):
        super(BeamScheduleNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, OUTPUT_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


def generate_batch(batch_size: int = BATCH_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand(batch_size, INPUT_SIZE) * 5.0
    means = torch.mean(inputs, dim=1, keepdim=True) / 5.0

    alpha = 0.2 + (0.8 * means)
    beta = means
    gamma = 0.0 + (0.3 * means)

    targets = torch.cat((alpha, beta, gamma), dim=1)
    return inputs, targets


def train_model(hidden_size: int, verbose: bool = True) -> Tuple[BeamScheduleNet, float]:
    model = BeamScheduleNet(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    final_loss = 0.0
    for epoch in range(EPOCHS):
        inputs, targets = generate_batch()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        if verbose and epoch % 100 == 0:
            print(f"    Epoch {epoch}: Loss {loss.item():.6f}")

    return model, final_loss


def save_weights(model: BeamScheduleNet, filename: str):
    with open(filename, "wb") as f:
        f.write(struct.pack("iii", INPUT_SIZE, model.hidden_size, OUTPUT_SIZE))
        w1 = model.fc1.weight.detach().numpy().flatten()
        b1 = model.fc1.bias.detach().numpy().flatten()
        f.write(struct.pack(f"{len(w1)}f", *w1))
        f.write(struct.pack(f"{len(b1)}f", *b1))
        w2 = model.fc2.weight.detach().numpy().flatten()
        b2 = model.fc2.bias.detach().numpy().flatten()
        f.write(struct.pack(f"{len(w2)}f", *w2))
        f.write(struct.pack(f"{len(b2)}f", *b2))


def get_memory_size(hidden_size: int) -> int:
    w1_size = hidden_size * INPUT_SIZE
    b1_size = hidden_size
    w2_size = OUTPUT_SIZE * hidden_size
    b2_size = OUTPUT_SIZE
    return (w1_size + b1_size + w2_size + b2_size) * 4  


def ground_truth(entropy_history: np.ndarray) -> Params:
    mean_entropy = np.mean(entropy_history[-INPUT_SIZE:]) / 5.0
    mean_entropy = np.clip(mean_entropy, 0, 1)

    alpha = 0.2 + (0.8 * mean_entropy)
    beta = mean_entropy
    gamma = 0.3 * mean_entropy

    return Params(A=alpha, B=beta, C=gamma)


def evaluate_accuracy(model: BeamScheduleNet, num_tests: int = 1000) -> Tuple[float, float]:
    np.random.seed(42)
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0

    with torch.no_grad():
        for _ in range(num_tests):
            history = np.random.rand(INPUT_SIZE) * 5.0
            gt = ground_truth(history)

            x = torch.tensor(history, dtype=torch.float32).unsqueeze(0)
            output = model(x).squeeze().numpy()

            gt_arr = np.array([gt.A, gt.B, gt.C])
            err = output - gt_arr

            mse_sum += np.mean(err ** 2)
            mae_sum += np.mean(np.abs(err))

    return mse_sum / num_tests, mae_sum / num_tests


def compile_benchmark():
    """Compile the C++ benchmark if needed."""
    benchmark_src = "benchmark.cpp"
    benchmark_bin = "benchmark"

    if os.path.exists(benchmark_bin):
        src_mtime = os.path.getmtime(benchmark_src)
        bin_mtime = os.path.getmtime(benchmark_bin)
        if bin_mtime > src_mtime:
            return True

    print("Compiling benchmark...")
    result = subprocess.run(
        ["g++", "-O3", "-std=c++17", "-march=native", "-ffast-math",
         "-o", benchmark_bin, benchmark_src],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return False
    return True


def run_benchmark(weights_file: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run C++ benchmark and extract query times.
    Returns (lut_time_ns, mlp_time_ns) or (None, None) on failure.
    """
    target_weights = "mlp/mlp_weights.bin"
    os.makedirs("mlp", exist_ok=True)

    backup_weights = None
    if os.path.exists(target_weights):
        backup_weights = target_weights + ".backup"
        os.rename(target_weights, backup_weights)

    try:
        with open(weights_file, "rb") as src, open(target_weights, "wb") as dst:
            dst.write(src.read())

        result = subprocess.run(
            ["./benchmark"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr}")
            return None, None

        output = result.stdout

        lut_time = None
        mlp_time = None

        for line in output.split('\n'):
            if 'Avg Query Time' in line:
                match = re.findall(r'(\d+\.?\d*)\s*ns', line)
                if len(match) >= 2:
                    lut_time = float(match[0])
                    mlp_time = float(match[1])
                break

        return lut_time, mlp_time

    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return None, None
    finally:
        if backup_weights and os.path.exists(backup_weights):
            if os.path.exists(target_weights):
                os.remove(target_weights)
            os.rename(backup_weights, target_weights)


def run_architecture_search(
    hidden_sizes: List[int] = HIDDEN_SIZES,
    speed_ratio_target: float = SPEED_RATIO_TARGET,
    accuracy_threshold_pct: float = ACCURACY_THRESHOLD_PCT,
    verbose: bool = True
) -> List[SearchResult]:
    results = []
    best_mse = float('inf')

    if not compile_benchmark():
        print("ERROR: Could not compile benchmark. Exiting.")
        return results

    print("\n" + "=" * 70)
    print("         ARCHITECTURE SEARCH PIPELINE")
    print("=" * 70)
    print(f"Hidden sizes to test: {hidden_sizes}")
    print(f"Speed ratio target: {speed_ratio_target}x")
    print(f"Accuracy threshold: {accuracy_threshold_pct}%")
    print("=" * 70 + "\n")

    for hidden_size in hidden_sizes:
        print(f"\n{'='*50}")
        print(f"Testing hidden_size = {hidden_size}")
        print('='*50)

        print("  Training...")
        model, training_loss = train_model(hidden_size, verbose=False)

        weights_file = f"mlp_weights_h{hidden_size}.bin"
        save_weights(model, weights_file)

        print("  Evaluating accuracy...")
        mse, mae = evaluate_accuracy(model)
        if mse < best_mse:
            best_mse = mse

        print("  Running performance benchmark...")
        lut_time, mlp_time = run_benchmark(weights_file)

        if lut_time is None or mlp_time is None:
            print(f"  WARNING: Benchmark failed for hidden_size={hidden_size}")
            os.remove(weights_file)
            continue

        speed_ratio = mlp_time / lut_time
        memory_bytes = get_memory_size(hidden_size)

        result = SearchResult(
            hidden_size=hidden_size,
            mse_total=mse,
            mae_total=mae,
            query_time_ns=mlp_time,
            lut_query_time_ns=lut_time,
            speed_ratio=speed_ratio,
            memory_bytes=memory_bytes,
            training_loss=training_loss
        )
        results.append(result)

        print(f"\n  Results for hidden_size={hidden_size}:")
        print(f"    MSE: {mse:.6f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    Query time: {mlp_time:.1f} ns (LUT: {lut_time:.1f} ns)")
        print(f"    Speed ratio: {speed_ratio:.2f}x {'✓' if speed_ratio <= speed_ratio_target else '✗'}")
        print(f"    Memory: {memory_bytes} bytes ({memory_bytes/1024:.2f} KB)")

        os.remove(weights_file)

    return results


def find_optimal_architecture(
    results: List[SearchResult],
    speed_ratio_target: float = SPEED_RATIO_TARGET,
    accuracy_threshold_pct: float = ACCURACY_THRESHOLD_PCT
) -> Optional[SearchResult]:

    if not results:
        return None

    best_mse = min(r.mse_total for r in results)
    accuracy_threshold = best_mse * (1 + accuracy_threshold_pct / 100)

    fast_enough = [r for r in results if r.speed_ratio <= speed_ratio_target]

    if not fast_enough:
        print(f"\nNo architecture meets speed target of {speed_ratio_target}x")
        fastest = min(results, key=lambda r: r.speed_ratio)
        print(f"Fastest architecture: hidden_size={fastest.hidden_size} "
              f"(speed_ratio={fastest.speed_ratio:.2f}x)")
        return fastest

    accurate_enough = [r for r in fast_enough if r.mse_total <= accuracy_threshold]

    if not accurate_enough:
        print(f"\nNo fast architecture meets accuracy threshold of {accuracy_threshold_pct}%")
        most_accurate = min(fast_enough, key=lambda r: r.mse_total)
        return most_accurate

    smallest = min(accurate_enough, key=lambda r: r.hidden_size)
    return smallest


def print_results_table(results: List[SearchResult]):
    """Print a formatted table of all results."""
    print("\n" + "=" * 90)
    print("                           ARCHITECTURE SEARCH RESULTS")
    print("=" * 90)
    print(f"{'Hidden':>8} {'MSE':>12} {'MAE':>12} {'MLP (ns)':>12} {'LUT (ns)':>12} "
          f"{'Ratio':>8} {'Memory':>10}")
    print("-" * 90)

    for r in results:
        print(f"{r.hidden_size:>8} {r.mse_total:>12.6f} {r.mae_total:>12.6f} "
              f"{r.query_time_ns:>12.1f} {r.lut_query_time_ns:>12.1f} "
              f"{r.speed_ratio:>7.2f}x {r.memory_bytes:>8} B")

    print("=" * 90)


def create_search_plots(results: List[SearchResult], output_file: str = "architecture_search.png"):
    """Create visualization of architecture search results."""
    if not results:
        return

    hidden_sizes = [r.hidden_size for r in results]
    mse_values = [r.mse_total for r in results]
    speed_ratios = [r.speed_ratio for r in results]
    memory_sizes = [r.memory_bytes for r in results]
    query_times = [r.query_time_ns for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MLP Architecture Search Results', fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(hidden_sizes, mse_values, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Hidden Layer Size')
    ax1.set_ylabel('MSE (Total)')
    ax1.set_title('Accuracy vs Architecture Size')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    ax2 = axes[0, 1]
    ax2.plot(hidden_sizes, speed_ratios, 'r-o', linewidth=2, markersize=8)
    ax2.axhline(y=SPEED_RATIO_TARGET, color='g', linestyle='--', label=f'Target ({SPEED_RATIO_TARGET}x)')
    ax2.axhline(y=1.0, color='orange', linestyle=':', label='LUT baseline')
    ax2.set_xlabel('Hidden Layer Size')
    ax2.set_ylabel('Speed Ratio (MLP/LUT)')
    ax2.set_title('Query Speed vs Architecture Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    ax3 = axes[1, 0]
    lut_time = results[0].lut_query_time_ns
    ax3.plot(hidden_sizes, query_times, 'b-o', linewidth=2, markersize=8, label='MLP')
    ax3.axhline(y=lut_time, color='g', linestyle='--', label=f'LUT ({lut_time:.1f} ns)')
    ax3.set_xlabel('Hidden Layer Size')
    ax3.set_ylabel('Query Time (ns)')
    ax3.set_title('Query Latency vs Architecture Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    ax4 = axes[1, 1]
    ax4.plot(hidden_sizes, memory_sizes, 'purple', marker='o', linewidth=2, markersize=8)
    ax4.set_xlabel('Hidden Layer Size')
    ax4.set_ylabel('Memory (bytes)')
    ax4.set_title('Memory Usage vs Architecture Size')
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()

    lut_memory = 100 * 50 * 12
    ax4.axhline(y=lut_memory, color='g', linestyle='--', label=f'LUT ({lut_memory/1024:.1f} KB)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def save_results_json(results: List[SearchResult], optimal: Optional[SearchResult],
                      output_file: str = "architecture_search_results.json"):
    """Save results to JSON for further analysis."""
    data = {
        "config": {
            "input_size": INPUT_SIZE,
            "output_size": OUTPUT_SIZE,
            "hidden_sizes_tested": HIDDEN_SIZES,
            "epochs": EPOCHS,
            "speed_ratio_target": SPEED_RATIO_TARGET,
            "accuracy_threshold_pct": ACCURACY_THRESHOLD_PCT
        },
        "results": [
            {
                "hidden_size": r.hidden_size,
                "mse_total": r.mse_total,
                "mae_total": r.mae_total,
                "query_time_ns": r.query_time_ns,
                "lut_query_time_ns": r.lut_query_time_ns,
                "speed_ratio": r.speed_ratio,
                "memory_bytes": r.memory_bytes,
                "training_loss": r.training_loss
            }
            for r in results
        ],
        "optimal": {
            "hidden_size": optimal.hidden_size,
            "mse_total": optimal.mse_total,
            "speed_ratio": optimal.speed_ratio,
            "memory_bytes": optimal.memory_bytes
        } if optimal else None
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    print("\n" + "=" * 70)
    print("    MLP ARCHITECTURE SEARCH FOR OPTIMAL PERFORMANCE")
    print("=" * 70)
    print("\nGoal: Find smallest MLP that achieves comparable query time to LUT")
    print(f"Target: MLP should be within {SPEED_RATIO_TARGET}x of LUT speed")
    print(f"Constraint: Accuracy degradation < {ACCURACY_THRESHOLD_PCT}%")
    print("=" * 70)

    results = run_architecture_search()

    if not results:
        print("\nNo results obtained. Exiting.")
        return

    print_results_table(results)

    optimal = find_optimal_architecture(results)

    if optimal:
        print("\n" + "=" * 70)
        print("                    OPTIMAL ARCHITECTURE FOUND")
        print("=" * 70)
        print(f"  Hidden Layer Size: {optimal.hidden_size}")
        print(f"  Architecture: {INPUT_SIZE} -> {optimal.hidden_size} -> {OUTPUT_SIZE}")
        print(f"  MSE: {optimal.mse_total:.6f}")
        print(f"  Query Time: {optimal.query_time_ns:.1f} ns")
        print(f"  Speed Ratio: {optimal.speed_ratio:.2f}x (vs LUT: {optimal.lut_query_time_ns:.1f} ns)")
        print(f"  Memory: {optimal.memory_bytes} bytes ({optimal.memory_bytes/1024:.3f} KB)")
        print("=" * 70)

        print("\nTraining and saving optimal model...")
        model, _ = train_model(optimal.hidden_size, verbose=False)
        save_weights(model, f"mlp/mlp_weights_optimal_h{optimal.hidden_size}.bin")
        print(f"Saved to: mlp/mlp_weights_optimal_h{optimal.hidden_size}.bin")

    create_search_plots(results)

    save_results_json(results, optimal)

    print("\nArchitecture search complete!")


if __name__ == "__main__":
    main()
