#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

const int DEFAULT_WINDOW_SIZE = 5;

// ========================== Params ==========================
struct Params {
  float A;
  float B;
  float C;
};

// ========================== LUT Scheduler ==========================
class DecayScheduleGenerator {
private:
  int time_resolution;
  int entropy_bins;
  float max_entropy;
  std::vector<Params> lut;

public:
  DecayScheduleGenerator(int t_res, int e_res, float max_e)
      : time_resolution(t_res), entropy_bins(e_res), max_entropy(max_e) {
    lut.resize(time_resolution * entropy_bins);
  }

  void generateSyntheticData() {
    for (int t = 0; t < time_resolution; ++t) {
      for (int e = 0; e < entropy_bins; ++e) {
        float norm_time = (float)t / (float)time_resolution;
        float curr_entropy = ((float)e / float(entropy_bins)) * max_entropy;
        float norm_entropy = curr_entropy / max_entropy;

        float A = 1.0f - (0.8f * norm_time);
        float B = -1.0f + (0.9f * norm_entropy) - (0.2f * norm_time);
        float C = (0.25f * norm_entropy);
        if (B > -0.05f)
          B = -0.05f;

        Params val = {A, B, C};
        lut[t * entropy_bins + e] = val;
      }
    }
  }

  bool loadFromBinary(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
      return false;
    }

    infile.read((char *)&time_resolution, sizeof(int));
    infile.read((char *)&entropy_bins, sizeof(int));
    infile.read((char *)&max_entropy, sizeof(float));
    lut.resize(time_resolution * entropy_bins);

    infile.read((char *)lut.data(), lut.size() * sizeof(Params));
    infile.close();
    return true;
  }

  Params query(int current_t, const std::vector<float> &entropy_points,
               int min_beam, int max_beam) const {
    float smoothed_entropy = 0.0f;
    if (!entropy_points.empty()) {
      int history_size = entropy_points.size();
      int lookback = std::min(history_size, DEFAULT_WINDOW_SIZE);
      float sum = 0.0f;
      for (int i = 0; i < lookback; ++i) {
        sum += entropy_points[history_size - 1 - i];
      }
      smoothed_entropy = sum / lookback;
    }

    int t_idx = std::min(current_t, time_resolution - 1);
    int e_idx = (int)((smoothed_entropy / max_entropy) * entropy_bins);
    e_idx = std::max(0, std::min(e_idx, entropy_bins - 1));

    int index = (t_idx * entropy_bins) + e_idx;
    const Params &entry = lut[index];

    Params result;
    result.A = (max_beam - min_beam) * entry.A;
    result.B = entry.B;
    result.C = min_beam + (max_beam * entry.C);

    return result;
  }

  int getTimeRes() const { return time_resolution; }
  int getEntropyBins() const { return entropy_bins; }
  size_t getMemorySize() const { return lut.size() * sizeof(Params); }
};

// ========================== MLP Scheduler ==========================
class MLPDecayScheduler {
private:
  int input_size;
  int hidden_size;
  int output_size;

  std::vector<float> W1;
  std::vector<float> b1;
  std::vector<float> W2;
  std::vector<float> b2;

  inline float sigmoid(float x) const { return 1.0f / (1.0f + std::exp(-x)); }

public:
  MLPDecayScheduler() : input_size(0), hidden_size(0), output_size(0) {}

  bool loadFromFile(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
      return false;
    }

    infile.read((char *)&input_size, sizeof(int));
    infile.read((char *)&hidden_size, sizeof(int));
    infile.read((char *)&output_size, sizeof(int));

    W1.resize(hidden_size * input_size);
    b1.resize(hidden_size);
    W2.resize(output_size * hidden_size);
    b2.resize(output_size);

    infile.read((char *)W1.data(), W1.size() * sizeof(float));
    infile.read((char *)b1.data(), b1.size() * sizeof(float));
    infile.read((char *)W2.data(), W2.size() * sizeof(float));
    infile.read((char *)b2.data(), b2.size() * sizeof(float));

    infile.close();
    return true;
  }

  Params query(const std::vector<float> &history, float min_beam,
               float max_beam) const {
    if ((int)history.size() != input_size) {
      Params res = {0.0f, 0.0f, 0.0f};
      return res;
    }

    std::vector<float> hidden(hidden_size);

    for (int h = 0; h < hidden_size; ++h) {
      float sum = b1[h];

      for (int i = 0; i < input_size; ++i) {
        sum += history[i] * W1[h * input_size + i];
      }

      hidden[h] = (sum > 0.0f) ? sum : 0.0f;
    }

    std::vector<float> output(output_size);

    for (int o = 0; o < output_size; ++o) {
      float sum = b2[o];

      for (int h = 0; h < hidden_size; ++h) {
        sum += hidden[h] * W2[o * hidden_size + h];
      }

      output[o] = sigmoid(sum);
    }

    float beam_range = max_beam - min_beam;
    Params res;
    res.A = beam_range * output[0];
    res.B = -1.0f + (0.95f * output[1]);
    res.C = min_beam + (beam_range * output[2]);

    return res;
  }

  int getInputSize() const { return input_size; }
  int getHiddenSize() const { return hidden_size; }
  int getOutputSize() const { return output_size; }
  size_t getMemorySize() const {
    return (W1.size() + b1.size() + W2.size() + b2.size()) * sizeof(float);
  }
};

// ========================== Benchmark ==========================
// Prevent compiler from optimizing away results
static Params g_sink;
void escape(const Params &p) {
  g_sink.A += p.A;
  g_sink.B += p.B;
  g_sink.C += p.C;
}

template <typename Func> double benchmarkQuery(Func &&func, int iterations) {
  // Warmup
  for (int i = 0; i < 1000; ++i) {
    func();
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    func();
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  return (double)duration.count() / iterations;
}

void printSeparator() {
  std::cout << "+----------------------+----------------+----------------+"
            << std::endl;
}

void printHeader() {
  printSeparator();
  std::cout << "| Metric               |      LUT       |      MLP       |"
            << std::endl;
  printSeparator();
}

void printRow(const std::string &label, const std::string &lut_val,
              const std::string &mlp_val) {
  std::cout << "| " << std::left << std::setw(20) << label << " | " << std::right
            << std::setw(14) << lut_val << " | " << std::setw(14) << mlp_val
            << " |" << std::endl;
}

int main() {
  std::cout << "\n===== SCHEDULER BENCHMARK =====\n" << std::endl;

  // Initialize schedulers
  DecayScheduleGenerator lut(100, 50, 10.0f);
  lut.generateSyntheticData();

  MLPDecayScheduler mlp;
  if (!mlp.loadFromFile("mlp/mlp_weights_optimal_h2.bin")) {
    std::cerr << "Warning: Could not load MLP weights from mlp/mlp_weights.bin"
              << std::endl;
    std::cerr << "MLP benchmark will show N/A" << std::endl;
  }

  // Setup random input generator
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> entropy_dist(0.0f, 10.0f);
  std::uniform_int_distribution<int> time_dist(0, 99);

  // Generate test inputs
  const int NUM_QUERIES = 100000;
  std::vector<std::vector<float>> entropy_histories;
  std::vector<int> time_steps;

  int mlp_input_size = mlp.getInputSize();
  if (mlp_input_size == 0)
    mlp_input_size = 3; // default

  for (int i = 0; i < NUM_QUERIES; ++i) {
    std::vector<float> history;
    for (int j = 0; j < std::max(DEFAULT_WINDOW_SIZE, mlp_input_size); ++j) {
      history.push_back(entropy_dist(rng));
    }
    entropy_histories.push_back(history);
    time_steps.push_back(time_dist(rng));
  }

  // Benchmark LUT
  int lut_idx = 0;
  double lut_time_ns = benchmarkQuery(
      [&]() {
        Params result = lut.query(time_steps[lut_idx % NUM_QUERIES],
                                  entropy_histories[lut_idx % NUM_QUERIES], 10,
                                  200);
        escape(result);
        lut_idx++;
      },
      NUM_QUERIES);

  // Benchmark MLP
  double mlp_time_ns = 0.0;
  bool mlp_valid = (mlp.getInputSize() > 0);

  if (mlp_valid) {
    int mlp_idx = 0;
    std::vector<std::vector<float>> mlp_histories;
    for (int i = 0; i < NUM_QUERIES; ++i) {
      std::vector<float> h(entropy_histories[i].begin(),
                           entropy_histories[i].begin() + mlp.getInputSize());
      mlp_histories.push_back(h);
    }

    mlp_time_ns = benchmarkQuery(
        [&]() {
          Params result =
              mlp.query(mlp_histories[mlp_idx % NUM_QUERIES], 10.0f, 200.0f);
          escape(result);
          mlp_idx++;
        },
        NUM_QUERIES);
  }

  // Print results
  printHeader();

  // Memory usage
  std::ostringstream lut_mem, mlp_mem;
  lut_mem << std::fixed << std::setprecision(2)
          << (lut.getMemorySize() / 1024.0) << " KB";
  if (mlp_valid) {
    mlp_mem << std::fixed << std::setprecision(2)
            << (mlp.getMemorySize() / 1024.0) << " KB";
  } else {
    mlp_mem << "N/A";
  }
  printRow("Memory Usage", lut_mem.str(), mlp_mem.str());

  // Configuration
  std::ostringstream lut_cfg, mlp_cfg;
  lut_cfg << lut.getTimeRes() << "x" << lut.getEntropyBins();
  if (mlp_valid) {
    mlp_cfg << mlp.getInputSize() << "->" << mlp.getHiddenSize() << "->"
            << mlp.getOutputSize();
  } else {
    mlp_cfg << "N/A";
  }
  printRow("Configuration", lut_cfg.str(), mlp_cfg.str());

  printSeparator();

  // Query time (ns)
  std::ostringstream lut_ns, mlp_ns;
  lut_ns << std::fixed << std::setprecision(1) << lut_time_ns << " ns";
  if (mlp_valid) {
    mlp_ns << std::fixed << std::setprecision(1) << mlp_time_ns << " ns";
  } else {
    mlp_ns << "N/A";
  }
  printRow("Avg Query Time", lut_ns.str(), mlp_ns.str());

  // Throughput
  std::ostringstream lut_tp, mlp_tp;
  double lut_queries_per_sec = 1e9 / lut_time_ns;
  lut_tp << std::fixed << std::setprecision(2) << (lut_queries_per_sec / 1e6)
         << " M/s";
  if (mlp_valid) {
    double mlp_queries_per_sec = 1e9 / mlp_time_ns;
    mlp_tp << std::fixed << std::setprecision(2) << (mlp_queries_per_sec / 1e6)
           << " M/s";
  } else {
    mlp_tp << "N/A";
  }
  printRow("Throughput", lut_tp.str(), mlp_tp.str());

  // Speedup
  std::ostringstream speedup;
  if (mlp_valid) {
    double ratio = mlp_time_ns / lut_time_ns;
    if (ratio > 1.0) {
      speedup << std::fixed << std::setprecision(2) << ratio << "x faster";
    } else {
      speedup << std::fixed << std::setprecision(2) << (1.0 / ratio)
              << "x slower";
    }
  } else {
    speedup << "N/A";
  }
  printRow("LUT vs MLP", speedup.str(), "-");

  printSeparator();

  // Additional metrics at different iteration counts
  std::cout << "\n===== LATENCY DISTRIBUTION =====\n" << std::endl;

  std::vector<int> test_sizes = {1000, 10000, 100000, 1000000};

  std::cout << "+------------+----------------+----------------+-------------+"
            << std::endl;
  std::cout << "| Iterations |   LUT (ns)     |   MLP (ns)     |   Speedup   |"
            << std::endl;
  std::cout << "+------------+----------------+----------------+-------------+"
            << std::endl;

  for (int n : test_sizes) {
    lut_idx = 0;
    double lut_t = benchmarkQuery(
        [&]() {
          Params result = lut.query(time_steps[lut_idx % NUM_QUERIES],
                                    entropy_histories[lut_idx % NUM_QUERIES], 10,
                                    200);
          escape(result);
          lut_idx++;
        },
        n);

    double mlp_t = 0.0;
    if (mlp_valid) {
      int mlp_idx = 0;
      std::vector<std::vector<float>> mlp_histories;
      for (int i = 0; i < std::min(n, NUM_QUERIES); ++i) {
        std::vector<float> h(entropy_histories[i].begin(),
                             entropy_histories[i].begin() + mlp.getInputSize());
        mlp_histories.push_back(h);
      }

      mlp_t = benchmarkQuery(
          [&]() {
            Params result = mlp.query(mlp_histories[mlp_idx % mlp_histories.size()],
                                      10.0f, 200.0f);
            escape(result);
            mlp_idx++;
          },
          n);
    }

    std::cout << "| " << std::setw(10) << n << " | " << std::setw(14)
              << std::fixed << std::setprecision(1) << lut_t << " | ";

    if (mlp_valid) {
      std::cout << std::setw(14) << std::fixed << std::setprecision(1) << mlp_t
                << " | ";
      double ratio = mlp_t / lut_t;
      if (ratio > 1.0) {
        std::cout << std::setw(9) << std::fixed << std::setprecision(2) << ratio
                  << "x  |";
      } else {
        std::cout << std::setw(9) << std::fixed << std::setprecision(2)
                  << (1.0 / ratio) << "x* |";
      }
    } else {
      std::cout << std::setw(14) << "N/A"
                << " | " << std::setw(11) << "N/A"
                << " |";
    }
    std::cout << std::endl;
  }

  std::cout << "+------------+----------------+----------------+-------------+"
            << std::endl;
  std::cout << "Note: Speedup shows how much faster LUT is vs MLP" << std::endl;
  std::cout << "      (* indicates MLP is faster)" << std::endl;

  std::cout << "\n===== SUMMARY =====\n" << std::endl;
  if (mlp_valid) {
    double ratio = mlp_time_ns / lut_time_ns;
    if (ratio > 1.0) {
      std::cout << "LUT is approximately " << std::fixed << std::setprecision(1)
                << ratio << "x faster than MLP" << std::endl;
    } else {
      std::cout << "MLP is approximately " << std::fixed << std::setprecision(1)
                << (1.0 / ratio) << "x faster than LUT" << std::endl;
    }
    std::cout << "LUT query: " << std::fixed << std::setprecision(1)
              << lut_time_ns << " ns (" << (1e9 / lut_time_ns / 1e6)
              << " M queries/sec)" << std::endl;
    std::cout << "MLP query: " << std::fixed << std::setprecision(1)
              << mlp_time_ns << " ns (" << (1e9 / mlp_time_ns / 1e6)
              << " M queries/sec)" << std::endl;
  }

  return 0;
}
