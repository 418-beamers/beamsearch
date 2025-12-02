#ifndef BEAM_SEARCH_CUH
#define BEAM_SEARCH_CUH

#include <cuda_runtime.h>

struct CTCBeamSearchConfig {
    int batch_size;
    int beam_width;
    int num_classes;      
    int max_time;         
    int max_output_length; 
    int blank_id;
    int batch_bits;
    int hash_bits;
};

struct CTCBeamSearchState {
    float* prob_blank;       
    float* prob_non_blank;    
    float* prob_total;       
    unsigned int* prefix_hashes; 
    int* current_lengths;    
    int* last_tokens;        

    int* history_parents;    
    int* history_tokens;     

    unsigned int* cand_keys;   
    float* cand_prob_blank;
    float* cand_prob_non_blank;
    int* cand_parent_idx;     
    int* cand_token;         
    int* cand_last_token;     
    
    unsigned int* cand_keys_sorted;
    int* cand_indices_sorted; 

    unsigned int* unique_keys;
    float* unique_prob_blank;
    float* unique_prob_non_blank;
    float* unique_prob_total;
    int* unique_parent_idx;
    int* unique_token;
    int* unique_last_token;
    int* unique_indices; 
    
    int* output_sequences;   
    int* output_lengths;
    float* output_scores;
};

class CTCBeamSearch {
public:
    CTCBeamSearch(const CTCBeamSearchConfig& config);
    ~CTCBeamSearch();

    void decode(const float* log_probs, const int* input_lengths, cudaStream_t stream = 0);

    int* get_sequences() const;
    int* get_lengths() const;
    float* get_scores() const;

private:
    CTCBeamSearchConfig config_;
    CTCBeamSearchState state_;

    void initialize(cudaStream_t stream);
    void launch(const float* log_probs, const int* input_lengths, cudaStream_t stream);
    void reconstruct(cudaStream_t stream);
    cudaError_t allocate_state();
    void free_state();
};

#endif 

