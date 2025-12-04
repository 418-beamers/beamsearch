#include "scheduler.h"
#include "mlp/mlp_decay_scheduler.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// ========================== Benchmark ==========================
// Prevent compiler from optimizing away results
static Params g_sink;
void escape(const Params &p) {
  g_sink.A += p.A;
  g_sink.B += p.B;
  g_sink.C += p.C;
}

template <typename Func> double benchmarkQuery(Func &&func, int iterations) {
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

  DecayScheduleGenerator lut(100, 50, 10.0f);
  lut.generateSyntheticData();

  MLPDecayScheduler mlp;
  if (!mlp.loadFromFile("mlp/mlp_weights.bin")) {
    std::cerr << "Warning: Could not load MLP weights from mlp/mlp_weights.bin"
              << std::endl;
    std::cerr << "MLP benchmark will show N/A" << std::endl;
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> entropy_dist(0.0f, 10.0f);
  std::uniform_int_distribution<int> time_dist(0, 99);

  const int NUM_QUERIES = 100000;
  std::vector<std::vector<float> > entropy_histories;
  std::vector<int> time_steps;

  int mlp_input_size = mlp.getInputSize();
  if (mlp_input_size == 0)
    mlp_input_size = 3; 

  for (int i = 0; i < NUM_QUERIES; ++i) {
    std::vector<float> history;
    for (int j = 0; j < std::max(DEFAULT_WINDOW_SIZE, mlp_input_size); ++j) {
      history.push_back(entropy_dist(rng));
    }
    entropy_histories.push_back(history);
    time_steps.push_back(time_dist(rng));
  }

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

  double mlp_time_ns = 0.0;
  bool mlp_valid = (mlp.getInputSize() > 0);

  if (mlp_valid) {
    int mlp_idx = 0;
    std::vector<std::vector<float> > mlp_histories;
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

  printHeader();

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

  std::ostringstream lut_ns, mlp_ns;
  lut_ns << std::fixed << std::setprecision(1) << lut_time_ns << " ns";
  if (mlp_valid) {
    mlp_ns << std::fixed << std::setprecision(1) << mlp_time_ns << " ns";
  } else {
    mlp_ns << "N/A";
  }
  printRow("Avg Query Time", lut_ns.str(), mlp_ns.str());

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
      std::vector<std::vector<float> > mlp_histories;
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
