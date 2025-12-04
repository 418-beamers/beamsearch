import struct
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

DEFAULT_WINDOW_SIZE = 5
INPUT_SIZE = 3
MAX_ENTROPY = 10.0
TIME_RESOLUTION = 100
ENTROPY_BINS = 50


@dataclass
class Params:
    A: float
    B: float
    C: float


def ground_truth(entropy_history: np.ndarray) -> Params:
    mean_entropy = np.mean(entropy_history[-INPUT_SIZE:]) / 5.0  
    mean_entropy = np.clip(mean_entropy, 0, 1)

    alpha = 0.2 + (0.8 * mean_entropy)
    beta = mean_entropy
    gamma = 0.3 * mean_entropy

    return Params(A=alpha, B=beta, C=gamma)


class LUTScheduler:
    def __init__(self, time_res: int = TIME_RESOLUTION,
                 entropy_bins: int = ENTROPY_BINS,
                 max_entropy: float = MAX_ENTROPY):
        self.time_resolution = time_res
        self.entropy_bins = entropy_bins
        self.max_entropy = max_entropy
        self.lut = []

    def generate_synthetic_data(self):
        self.lut = []
        for t in range(self.time_resolution):
            for e in range(self.entropy_bins):
                norm_time = t / self.time_resolution
                curr_entropy = (e / self.entropy_bins) * self.max_entropy
                norm_entropy = curr_entropy / self.max_entropy

                A = 1.0 - (0.8 * norm_time)
                B = -1.0 + (0.9 * norm_entropy) - (0.2 * norm_time)
                C = 0.25 * norm_entropy

                if B > -0.05:
                    B = -0.05

                self.lut.append(Params(A=A, B=B, C=C))

    def query(self, current_t: int, entropy_points: List[float],
              min_beam: int = 10, max_beam: int = 200) -> Params:
        """Query the LUT for decay parameters."""
        smoothed_entropy = 0.0
        if entropy_points:
            lookback = min(len(entropy_points), DEFAULT_WINDOW_SIZE)
            smoothed_entropy = np.mean(entropy_points[-lookback:])

        t_idx = min(current_t, self.time_resolution - 1)
        e_idx = int((smoothed_entropy / self.max_entropy) * self.entropy_bins)
        e_idx = max(0, min(e_idx, self.entropy_bins - 1))

        index = t_idx * self.entropy_bins + e_idx
        entry = self.lut[index]

        result = Params(
            A=(max_beam - min_beam) * entry.A,
            B=entry.B,
            C=min_beam + (max_beam * entry.C)
        )
        return result

    def query_raw(self, current_t: int, entropy_points: List[float]) -> Params:
        smoothed_entropy = 0.0
        if entropy_points:
            lookback = min(len(entropy_points), DEFAULT_WINDOW_SIZE)
            smoothed_entropy = np.mean(entropy_points[-lookback:])

        t_idx = min(current_t, self.time_resolution - 1)
        e_idx = int((smoothed_entropy / self.max_entropy) * self.entropy_bins)
        e_idx = max(0, min(e_idx, self.entropy_bins - 1))

        index = t_idx * self.entropy_bins + e_idx
        return self.lut[index]


class MLPScheduler:
    def __init__(self):
        self.input_size = 0
        self.hidden_size = 0
        self.output_size = 0
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def load_from_file(self, filename: str) -> bool:
        try:
            with open(filename, 'rb') as f:
                self.input_size, self.hidden_size, self.output_size = struct.unpack('iii', f.read(12))

                w1_size = self.hidden_size * self.input_size
                self.W1 = np.array(struct.unpack(f'{w1_size}f', f.read(w1_size * 4)))
                self.W1 = self.W1.reshape(self.hidden_size, self.input_size)

                self.b1 = np.array(struct.unpack(f'{self.hidden_size}f', f.read(self.hidden_size * 4)))

                w2_size = self.output_size * self.hidden_size
                self.W2 = np.array(struct.unpack(f'{w2_size}f', f.read(w2_size * 4)))
                self.W2 = self.W2.reshape(self.output_size, self.hidden_size)

                self.b2 = np.array(struct.unpack(f'{self.output_size}f', f.read(self.output_size * 4)))

            print(f"MLP Loaded ({self.input_size}->{self.hidden_size}->{self.output_size})")
            return True
        except Exception as e:
            print(f"Error loading MLP: {e}")
            return False

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def query(self, history: List[float], min_beam: float = 10,
              max_beam: float = 200) -> Params:
        if len(history) != self.input_size:
            return Params(A=0, B=0, C=0)

        x = np.array(history)

        hidden = np.maximum(0, self.W1 @ x + self.b1)

        output = self.sigmoid(self.W2 @ hidden + self.b2)

        beam_range = max_beam - min_beam
        return Params(
            A=beam_range * output[0],
            B=-1.0 + (0.95 * output[1]),
            C=min_beam + (beam_range * output[2])
        )

    def query_raw(self, history: List[float]) -> Params:
        if len(history) != self.input_size:
            return Params(A=0, B=0, C=0)

        x = np.array(history)
        hidden = np.maximum(0, self.W1 @ x + self.b1)
        output = self.sigmoid(self.W2 @ hidden + self.b2)

        return Params(A=output[0], B=output[1], C=output[2])


def compute_errors(predictions: List[Params], ground_truths: List[Params]) -> dict:
    pred_A = np.array([p.A for p in predictions])
    pred_B = np.array([p.B for p in predictions])
    pred_C = np.array([p.C for p in predictions])

    gt_A = np.array([g.A for g in ground_truths])
    gt_B = np.array([g.B for g in ground_truths])
    gt_C = np.array([g.C for g in ground_truths])

    err_A = pred_A - gt_A
    err_B = pred_B - gt_B
    err_C = pred_C - gt_C

    return {
        'MSE_A': np.mean(err_A ** 2),
        'MSE_B': np.mean(err_B ** 2),
        'MSE_C': np.mean(err_C ** 2),
        'MSE_total': np.mean(err_A ** 2 + err_B ** 2 + err_C ** 2),
        'MAE_A': np.mean(np.abs(err_A)),
        'MAE_B': np.mean(np.abs(err_B)),
        'MAE_C': np.mean(np.abs(err_C)),
        'MAE_total': np.mean(np.abs(err_A) + np.abs(err_B) + np.abs(err_C)),
        'SSE_A': np.sum(err_A ** 2),
        'SSE_B': np.sum(err_B ** 2),
        'SSE_C': np.sum(err_C ** 2),
        'SSE_total': np.sum(err_A ** 2 + err_B ** 2 + err_C ** 2),
        'RMSE_total': np.sqrt(np.mean(err_A ** 2 + err_B ** 2 + err_C ** 2)),
        'max_err_A': np.max(np.abs(err_A)),
        'max_err_B': np.max(np.abs(err_B)),
        'max_err_C': np.max(np.abs(err_C)),
    }


def print_table(lut_errors: dict, mlp_errors: dict):
    print("\n" + "=" * 70)
    print("                    ACCURACY COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<25} {'LUT':>15} {'MLP':>15} {'Winner':>12}")
    print("-" * 70)

    metrics = [
        ('MSE (total)', 'MSE_total'),
        ('MAE (total)', 'MAE_total'),
        ('RMSE (total)', 'RMSE_total'),
        ('SSE (total)', 'SSE_total'),
        ('MSE (A)', 'MSE_A'),
        ('MSE (B)', 'MSE_B'),
        ('MSE (C)', 'MSE_C'),
        ('MAE (A)', 'MAE_A'),
        ('MAE (B)', 'MAE_B'),
        ('MAE (C)', 'MAE_C'),
        ('Max Error (A)', 'max_err_A'),
        ('Max Error (B)', 'max_err_B'),
        ('Max Error (C)', 'max_err_C'),
    ]

    for name, key in metrics:
        lut_val = lut_errors[key]
        mlp_val = mlp_errors[key]
        winner = "LUT" if lut_val < mlp_val else "MLP" if mlp_val < lut_val else "TIE"
        print(f"{name:<25} {lut_val:>15.6f} {mlp_val:>15.6f} {winner:>12}")

    print("=" * 70)


def create_plots(test_inputs: List[np.ndarray],
                 ground_truths: List[Params],
                 lut_preds: List[Params],
                 mlp_preds: List[Params],
                 output_file: str = "accuracy_comparison.png"):
    gt_A = np.array([g.A for g in ground_truths])
    gt_B = np.array([g.B for g in ground_truths])
    gt_C = np.array([g.C for g in ground_truths])

    lut_A = np.array([p.A for p in lut_preds])
    lut_B = np.array([p.B for p in lut_preds])
    lut_C = np.array([p.C for p in lut_preds])

    mlp_A = np.array([p.A for p in mlp_preds])
    mlp_B = np.array([p.B for p in mlp_preds])
    mlp_C = np.array([p.C for p in mlp_preds])

    mean_entropies = np.array([np.mean(inp[-INPUT_SIZE:]) for inp in test_inputs])
    sort_idx = np.argsort(mean_entropies)

    gt_A, gt_B, gt_C = gt_A[sort_idx], gt_B[sort_idx], gt_C[sort_idx]
    lut_A, lut_B, lut_C = lut_A[sort_idx], lut_B[sort_idx], lut_C[sort_idx]
    mlp_A, mlp_B, mlp_C = mlp_A[sort_idx], mlp_B[sort_idx], mlp_C[sort_idx]
    mean_entropies = mean_entropies[sort_idx]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('LUT vs MLP Scheduler Accuracy Comparison', fontsize=14, fontweight='bold')

    axes[0, 0].scatter(mean_entropies, gt_A, alpha=0.5, s=10, label='Ground Truth', c='green')
    axes[0, 0].scatter(mean_entropies, lut_A, alpha=0.5, s=10, label='LUT', c='blue')
    axes[0, 0].scatter(mean_entropies, mlp_A, alpha=0.5, s=10, label='MLP', c='red')
    axes[0, 0].set_xlabel('Mean Entropy')
    axes[0, 0].set_ylabel('Parameter A')
    axes[0, 0].set_title('Parameter A: Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(mean_entropies, lut_A - gt_A, alpha=0.5, s=10, label='LUT Error', c='blue')
    axes[0, 1].scatter(mean_entropies, mlp_A - gt_A, alpha=0.5, s=10, label='MLP Error', c='red')
    axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Mean Entropy')
    axes[0, 1].set_ylabel('Error (Pred - GT)')
    axes[0, 1].set_title('Parameter A: Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(lut_A - gt_A, bins=50, alpha=0.5, label='LUT', color='blue', density=True)
    axes[0, 2].hist(mlp_A - gt_A, bins=50, alpha=0.5, label='MLP', color='red', density=True)
    axes[0, 2].axvline(x=0, color='green', linestyle='--', alpha=0.7)
    axes[0, 2].set_xlabel('Error')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Parameter A: Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].scatter(mean_entropies, gt_B, alpha=0.5, s=10, label='Ground Truth', c='green')
    axes[1, 0].scatter(mean_entropies, lut_B, alpha=0.5, s=10, label='LUT', c='blue')
    axes[1, 0].scatter(mean_entropies, mlp_B, alpha=0.5, s=10, label='MLP', c='red')
    axes[1, 0].set_xlabel('Mean Entropy')
    axes[1, 0].set_ylabel('Parameter B')
    axes[1, 0].set_title('Parameter B: Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(mean_entropies, lut_B - gt_B, alpha=0.5, s=10, label='LUT Error', c='blue')
    axes[1, 1].scatter(mean_entropies, mlp_B - gt_B, alpha=0.5, s=10, label='MLP Error', c='red')
    axes[1, 1].axhline(y=0, color='green', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Mean Entropy')
    axes[1, 1].set_ylabel('Error (Pred - GT)')
    axes[1, 1].set_title('Parameter B: Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(lut_B - gt_B, bins=50, alpha=0.5, label='LUT', color='blue', density=True)
    axes[1, 2].hist(mlp_B - gt_B, bins=50, alpha=0.5, label='MLP', color='red', density=True)
    axes[1, 2].axvline(x=0, color='green', linestyle='--', alpha=0.7)
    axes[1, 2].set_xlabel('Error')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Parameter B: Error Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 0].scatter(mean_entropies, gt_C, alpha=0.5, s=10, label='Ground Truth', c='green')
    axes[2, 0].scatter(mean_entropies, lut_C, alpha=0.5, s=10, label='LUT', c='blue')
    axes[2, 0].scatter(mean_entropies, mlp_C, alpha=0.5, s=10, label='MLP', c='red')
    axes[2, 0].set_xlabel('Mean Entropy')
    axes[2, 0].set_ylabel('Parameter C')
    axes[2, 0].set_title('Parameter C: Predictions')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].scatter(mean_entropies, lut_C - gt_C, alpha=0.5, s=10, label='LUT Error', c='blue')
    axes[2, 1].scatter(mean_entropies, mlp_C - gt_C, alpha=0.5, s=10, label='MLP Error', c='red')
    axes[2, 1].axhline(y=0, color='green', linestyle='--', alpha=0.7)
    axes[2, 1].set_xlabel('Mean Entropy')
    axes[2, 1].set_ylabel('Error (Pred - GT)')
    axes[2, 1].set_title('Parameter C: Errors')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].hist(lut_C - gt_C, bins=50, alpha=0.5, label='LUT', color='blue', density=True)
    axes[2, 2].hist(mlp_C - gt_C, bins=50, alpha=0.5, label='MLP', color='red', density=True)
    axes[2, 2].axvline(x=0, color='green', linestyle='--', alpha=0.7)
    axes[2, 2].set_xlabel('Error')
    axes[2, 2].set_ylabel('Density')
    axes[2, 2].set_title('Parameter C: Error Distribution')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def create_decay_curve_plot(test_inputs: List[np.ndarray],
                            ground_truths: List[Params],
                            lut_preds: List[Params],
                            mlp_preds: List[Params],
                            output_file: str = "decay_curves.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Exponential Decay Curves: f(x) = Ae^(Bx) + C', fontsize=14, fontweight='bold')

    x = np.linspace(0, 10, 100)

    indices = [0, len(test_inputs)//5, 2*len(test_inputs)//5,
               3*len(test_inputs)//5, 4*len(test_inputs)//5, -1]

    for idx, (ax, sample_idx) in enumerate(zip(axes.flat, indices)):
        gt = ground_truths[sample_idx]
        lut = lut_preds[sample_idx]
        mlp = mlp_preds[sample_idx]
        inp = test_inputs[sample_idx]

        min_beam, max_beam = 10, 200
        beam_range = max_beam - min_beam

        gt_A_scaled = beam_range * gt.A
        gt_B_scaled = -1.0 + 0.95 * gt.B  
        gt_C_scaled = min_beam + beam_range * gt.C
        y_gt = gt_A_scaled * np.exp(gt_B_scaled * x) + gt_C_scaled

        y_lut = lut.A * np.exp(lut.B * x) + lut.C

        y_mlp = mlp.A * np.exp(mlp.B * x) + mlp.C

        ax.plot(x, y_gt, 'g-', linewidth=2, label='Ground Truth')
        ax.plot(x, y_lut, 'b--', linewidth=2, label='LUT')
        ax.plot(x, y_mlp, 'r:', linewidth=2, label='MLP')

        mean_ent = np.mean(inp[-INPUT_SIZE:])
        ax.set_title(f'Sample {idx+1} (mean entropy: {mean_ent:.2f})')
        ax.set_xlabel('x (time step)')
        ax.set_ylabel('Beam Width')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max_beam + 20])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Decay curves plot saved to: {output_file}")
    plt.close()


def main():
    print("=" * 70)
    print("         SCHEDULER ACCURACY BENCHMARK")
    print("=" * 70)

    print("\nInitializing schedulers...")
    lut = LUTScheduler()
    lut.generate_synthetic_data()
    print(f"LUT initialized ({TIME_RESOLUTION}x{ENTROPY_BINS} = {TIME_RESOLUTION*ENTROPY_BINS} entries)")

    mlp = MLPScheduler()
    if not mlp.load_from_file("mlp/mlp_weights_optimal_h2.bin"):
        print("ERROR: Could not load MLP weights. Exiting.")
        return

    print("\nGenerating test data...")
    np.random.seed(42)
    NUM_TESTS = 1000

    test_inputs = []
    ground_truths = []
    lut_preds = []
    mlp_preds = []

    for i in range(NUM_TESTS):
        history = list(np.random.rand(max(DEFAULT_WINDOW_SIZE, INPUT_SIZE)) * 5.0)
        test_inputs.append(np.array(history))

        gt = ground_truth(np.array(history))
        ground_truths.append(gt)

        lut_result = lut.query(50, history, min_beam=10, max_beam=200)
        lut_preds.append(lut_result)

        mlp_input = history[-INPUT_SIZE:]
        mlp_result = mlp.query(mlp_input, min_beam=10, max_beam=200)
        mlp_preds.append(mlp_result)

    print(f"Generated {NUM_TESTS} test cases")

    print("\nComputing raw (unscaled) predictions for fair comparison...")
    lut_raw = []
    mlp_raw = []

    for i in range(NUM_TESTS):
        history = list(test_inputs[i])
        lut_raw.append(lut.query_raw(50, history))
        mlp_raw.append(mlp.query_raw(history[-INPUT_SIZE:]))

    print("\nComputing error metrics...")
    lut_errors = compute_errors(lut_raw, ground_truths)
    mlp_errors = compute_errors(mlp_raw, ground_truths)

    print_table(lut_errors, mlp_errors)

    print("\n" + "=" * 70)
    print("                         SUMMARY")
    print("=" * 70)

    lut_wins = sum(1 for key in lut_errors if lut_errors[key] < mlp_errors.get(key, float('inf')))
    mlp_wins = sum(1 for key in mlp_errors if mlp_errors[key] < lut_errors.get(key, float('inf')))

    print(f"LUT wins on {lut_wins} metrics")
    print(f"MLP wins on {mlp_wins} metrics")

    if mlp_errors['MSE_total'] < lut_errors['MSE_total']:
        improvement = (lut_errors['MSE_total'] - mlp_errors['MSE_total']) / lut_errors['MSE_total'] * 100
        print(f"\nMLP is {improvement:.1f}% more accurate (by MSE)")
    else:
        improvement = (mlp_errors['MSE_total'] - lut_errors['MSE_total']) / mlp_errors['MSE_total'] * 100
        print(f"\nLUT is {improvement:.1f}% more accurate (by MSE)")

    print("=" * 70)

    print("\nGenerating plots...")
    create_plots(test_inputs, ground_truths, lut_raw, mlp_raw, "accuracy_comparison.png")
    create_decay_curve_plot(test_inputs, ground_truths, lut_preds, mlp_preds, "decay_curves.png")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
