#ifndef MLP_DECAY_SCHEDULER_H
#define MLP_DECAY_SCHEDULER_H

#include "../scheduler.h"
#include <string>
#include <vector>

class MLPDecayScheduler {
  private:
    int input_size;
    int hidden_size;
    int output_size;

    std::vector<float> W1;
    std::vector<float> b1;
    std::vector<float> W2;
    std::vector<float> b2;

    inline float sigmoid(float x) const;

  public:
    MLPDecayScheduler();

    bool loadFromFile(const std::string& filename);
    Params query(const std::vector<float>& history, float min_beam, float max_beam) const;

    int getInputSize() const { return input_size; }
    int getHiddenSize() const { return hidden_size; }
    int getOutputSize() const { return output_size; }
    size_t getMemorySize() const { return (W1.size() + b1.size() + W2.size() + b2.size()) * sizeof(float); }
};

#endif
