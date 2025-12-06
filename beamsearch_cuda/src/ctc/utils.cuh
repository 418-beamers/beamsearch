#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <thrust/tuple.h>
#include <cmath>

#define NEG_INF -1e20f

struct BatchScoreComp {
    const unsigned int* keys;
    const float* scores;
    int shift;

    BatchScoreComp(const unsigned int* k, const float* s, int shift_amount) 
        : keys(k), scores(s), shift(shift_amount) {}

    __host__ __device__
    bool operator()(int a, int b) const {
        int batchA = keys[a] >> shift; 
        int batchB = keys[b] >> shift;
        if (batchA != batchB) {
            return batchA < batchB;
        }
        return scores[a] > scores[b]; 
    }
};

__host__ __device__ __forceinline__ float log_add_helper(float a, float b) {
    if (a <= NEG_INF) return b;
    if (b <= NEG_INF) return a;
    float maxVal = fmaxf(a, b);
    return maxVal + log1pf(expf(-fabsf(a - b)));
}

struct ScoreAndIndexReduce {
    __host__ __device__
    thrust::tuple<float, float, int> operator()(const thrust::tuple<float, float, int>& a,
                                           const thrust::tuple<float, float, int>& b) const {
        float a_sb  = thrust::get<0>(a);
        float a_snb = thrust::get<1>(a);
        
        float b_sb  = thrust::get<0>(b);
        float b_snb = thrust::get<1>(b);
        return thrust::make_tuple(log_add_helper(a_sb, b_sb), log_add_helper(a_snb, b_snb), thrust::get<2>(a));
    }
};

struct CalcTotalScore {
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        float sb  = thrust::get<0>(t);
        float snb = thrust::get<1>(t);
        return log_add_helper(sb, snb);
    }
};

// entropy contribution: -p * log(p) = -exp(log_p) * log_p
struct CalcEntropyContribution {
    __host__ __device__ 
    float operator()(float log_p) const {
        if (log_p < -30.0f) return 0.0f; // for numerical stability
        float p = expf(log_p);
        return -p * log_p;
    }
};

#endif // UTILS_CUH

