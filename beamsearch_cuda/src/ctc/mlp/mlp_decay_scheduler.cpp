#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>

struct Params {
  float A;
  float B;
  float C;
};

class MLPDecayScheduler {
  private:
    int input_size;
    int hidden_size;
    int output_size;

    std::vector<float> W1;
    std::vector<float> b1;
    std::vector<float> W2;
    std::vector<float> b2;

    inline float sigmoid(float x) const {
      return 1.0f / (1.0f + std::exp(-x));
    }
  public:
    MLPDecayScheduler() : input_size(0), hidden_size(0), output_size(0) {}

    bool loadFromFile(const std::string& filename) {
      std::ifstream infile(filename, std::ios::binary);
      if (!infile.is_open()) {
        std::cerr << "Error: could not open " << filename << std::endl;
        return false;
      }

      infile.read((char*)&input_size, sizeof(int));
      infile.read((char*)&hidden_size, sizeof(int));
      infile.read((char*)&output_size, sizeof(int));

      W1.resize(hidden_size * input_size);
      b1.resize(hidden_size);
      W2.resize(output_size * hidden_size);
      b2.resize(output_size);

      infile.read((char*)W1.data(), W1.size() * sizeof(float));
      infile.read((char*)b1.data(), b1.size() * sizeof(float));
      infile.read((char*)W2.data(), W2.size() * sizeof(float));
      infile.read((char*)b2.data(), b2.size() * sizeof(float));

      infile.close();
      std::cout << "MLP Loaded (" << input_size << "->" << hidden_size << "->" << output_size << ")\n";
      return true;
    }

  Params query(const std::vector<float>& history, float min_beam, float max_beam) const {
    if (history.size() != input_size) {
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
       float sum = b2[0];

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
};

int main() {
    // example usage
    MLPDecayScheduler net;
    
    if (!net.loadFromFile("mlp_weights.bin")) return -1;

    std::vector<float> history = {0.2f, 0.1f, 0.3f}; 

    Params p1 = net.query(history, 10.0f, 200.0f);
    std::cout << "Query A (10-200): " 
              << p1.A << "e^(" << p1.B << "x) + " << p1.C << std::endl;

    Params p2 = net.query(history, 5.0f, 50.0f);
    std::cout << "Query B (5-50):   " 
              << p2.A << "e^(" << p2.B << "x) + " << p2.C << std::endl;

    return 0;
}
