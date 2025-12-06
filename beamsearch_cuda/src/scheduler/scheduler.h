#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <string>
#include <vector>

const int DEFAULT_WINDOW_SIZE = 5;

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
    DecayScheduleGenerator(int t_res, int e_res, float max_e);

    void generateSyntheticData();
    void saveToBinary(const std::string& filename);
    bool loadFromBinary(const std::string& filename);
    Params query(int current_t, const std::vector<float>& entropy_points, int min_beam, int max_beam) const;

    int getTimeRes() const { return time_resolution; }
    int getEntropyBins() const { return entropy_bins; }
    size_t getMemorySize() const { return lut.size() * sizeof(Params); }
};

#endif
