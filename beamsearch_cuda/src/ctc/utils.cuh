#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <thrust/tuple.h>
#include <cmath>

#define NEG_INF -1e20f

struct BatchScoreComp {
    const unsigned long long* keys;
    const float* scores;

    BatchScoreComp(const unsigned long long* k, const float* s) : keys(k), scores(s) {}

    __host__ __device__
    bool operator()(int a, int b) const {
        int batchA = (int)(keys[a] >> 32);
        int batchB = (int)(keys[b] >> 32);
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

struct ProbAndIndexReduce {
    __host__ __device__
    thrust::tuple<float, float, int> operator()(const thrust::tuple<float, float, int>& a,
                                           const thrust::tuple<float, float, int>& b) const {
        float a_pb  = thrust::get<0>(a);
        float a_pnb = thrust::get<1>(a);
        
        float b_pb  = thrust::get<0>(b);
        float b_pnb = thrust::get<1>(b);
        return thrust::make_tuple(log_add_helper(a_pb, b_pb), log_add_helper(a_pnb, b_pnb), thrust::get<2>(a));
    }
};

struct CalcTotalProb {
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        float pb  = thrust::get<0>(t);
        float pnb = thrust::get<1>(t);
        return log_add_helper(pb, pnb);
    }
};

#endif // UTILS_CUH

