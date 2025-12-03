#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>

const int DEFAULT_WINDOW_SIZE = 5;

// f(x) = Ae^{Bx} + C
struct Params {
  float A;
  float B;
  float C;
};

class DecayScheduleGenerator {
  private:
    int time_resolution;
    int entropy_bins;
    float max_entropy;
    std::vector<Params> lut;

  public:
    DecayScheduleGenerator(int t_res, int e_res,  float max_e):
      time_resolution(t_res), entropy_bins(e_res), max_entropy(max_e) {
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
          if (B > -0.05f) B = -0.05f;

          Params val = {A, B, C};
          lut[t * entropy_bins + e] = val;
        }
      }
    }

  void saveToBinary(const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) return;

    // file header
    outfile.write((char*)&time_resolution, sizeof(int));
    outfile.write((char*)&entropy_bins, sizeof(int));
    outfile.write((char*)&max_entropy, sizeof(float));

    // data
    outfile.write((char*)lut.data(), lut.size() * sizeof(Params));

    outfile.close();
    std::cout << "LUT saved to " << filename << std::endl;
  }

  bool loadFromBinary(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
      std::cerr << "Error: Could not find LUT file " << filename << std::endl;
      return false;
    }

    infile.read((char*)&time_resolution, sizeof(int));
    infile.read((char*)&entropy_bins, sizeof(int));
    infile.read((char*)&max_entropy, sizeof(float));
    lut.resize(time_resolution * entropy_bins);

    infile.read((char*)lut.data(), lut.size() * sizeof(Params));
    infile.close();

    std::cout << "Loaded LUT (" << time_resolution << "x" << entropy_bins << ")" << std::endl;
    return true;
  }

  Params query (int current_t, const std::vector<float>& entropy_points, int min_beam, int max_beam) const {
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
    int e_idx = (int) ((smoothed_entropy / max_entropy) * entropy_bins);
    e_idx = std::max(0, std::min(e_idx, entropy_bins - 1));

    int index = (t_idx * entropy_bins) + e_idx;
    const Params& entry = lut[index];

    Params result;
    result.A = (max_beam - min_beam) * entry.A;
    result.B = entry.B;
    result.C = min_beam + (max_beam * entry.C);

    return result;
  }
};
