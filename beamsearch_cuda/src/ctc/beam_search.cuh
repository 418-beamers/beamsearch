#ifndef BEAM_SEARCH_CUH
#define BEAM_SEARCH_CUH

#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <vector>

class DecayScheduleGenerator;
class MLPDecayScheduler;

enum class SchedulerType {
    NAIVE,
    LUT,
    MLP
};

struct BeamSchedule {
    bool adaptive_beam_width;
    SchedulerType scheduler_type;

    // used with NAIVE
    float a;
    float b;
    float c;
    int min;
    int init;
    int init_steps;

    // file path for LUT
    std::string lut_path;
    std::string mlp_path;
};

struct CTCBeamSearchConfig {
    int batch_size;
    int beam_width;
    int num_classes;      
    int max_time;         
    int max_output_length; 
    int blank_id;
    int batch_bits;
    int hash_bits;

    BeamSchedule schedule;
};

struct BeamState {
    float* score_blank;       
    float* score_non_blank;    
    float* score_total;       
    unsigned int* prefix_hashes; 
    int* current_lengths;    
    int* last_tokens;        
    int* history_parents;    
    int* history_tokens;     
};

struct CandidateState {
    unsigned int* keys;   
    float* score_blank;
    float* score_non_blank;
    int* parent_idx;     
    int* token;         
    int* last_token;     
    unsigned int* keys_sorted;
    int* indices_sorted; 
};

struct UniqueState {
    unsigned int* keys;
    float* score_blank;
    float* score_non_blank;
    float* score_total;
    int* parent_idx;
    int* token;
    int* last_token;
    int* indices; 
};

struct OutputState {
    int* sequences;   
    int* lengths;
    float* scores;
};

struct CTCBeamSearchState {
    BeamState beam;
    CandidateState cand;
    UniqueState unique;
    OutputState output;
};

class CTCBeamSearch {
public:
    CTCBeamSearch(const CTCBeamSearchConfig& config);
    ~CTCBeamSearch();

    void decode(const float* log_probs, const int* input_lengths, cudaStream_t stream = 0);

    int* get_sequences() const;
    int* get_lengths() const;
    float* get_scores() const;
    
    const std::vector<int>& get_beam_width_history() const { return beam_width_history_; }
    const std::vector<float>& get_entropy_history() const { return entropy_history_; }

private:
    CTCBeamSearchConfig config_;
    CTCBeamSearchState state_;

    std::unique_ptr<DecayScheduleGenerator> lut_scheduler_;
    std::unique_ptr<MLPDecayScheduler> mlp_scheduler_;
    std::vector<float> entropy_history_;
    std::vector<int> beam_width_history_;

    void initialize(cudaStream_t stream);
    void launch(const float* log_probs, const int* input_lengths, cudaStream_t stream);
    void reconstruct(cudaStream_t stream);
    cudaError_t allocate_state();
    void free_state();

    int compute_beam_width(int t, float current_entropy);
};

#endif 

