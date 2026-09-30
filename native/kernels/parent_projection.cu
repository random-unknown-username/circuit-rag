#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/circuit/numerics.cuh"

namespace circuit {

// warp parallel parent proj kernel
// - grid: blockidx.x = suriving root idx
// - block: dim3(32, rank) where each warp computes one rank coordinate proj
// - zero wrap divergence, fully coalesced DRAM transactions
__global__ void parent_projection_kernel(
    const float* __restrict__ query,     // [D]
    const float query_norm_sq,           // ||q||_2^2
    const float* __restrict__ bases,     // [num_roots, D, rank]
    const int* __restrict__ survivor_indices, // [num_surviving_roots]
    const int num_surviving_roots,
    const int dim,
    const int rank,
    float* __restrict__ projected_queries, // [num_roots, rank]
    float* __restrict__ perp_norms // [num_roots]
) {
    int s_idx = blockIdx.x;
    if (s_idx >= num_surviving_roots) return;

    int root_idx = survivor_indices[s_idx];
    const float* __restrict__ basis_ptr = bases + (size_t)root_idx * dim * rank;
    float* __restrict__ q_P_out = projected_queries + (size_t)root_idx * rank;

    extern __shared__ float s_qP[]; // rank floats

    int lane = threadIdx.x; // 0..31
    int r = threadIdx.y; // 0..rank-1

    if (r < rank) {
        float comp = 0.0f;
        // basis layout is [dim, rank] -> basis_ptr[d * rank + r]
        for (int d = lane; d < dim; d += 32) {
            comp += query[d] * basis_ptr[d * rank + r];
        }
        
        // warp reduction across 32 threads
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            comp += __shfl_down_sync(0xffffffff, comp, offset);
        }

        if (lane == 0) {
            s_qP[r] = comp;
            q_P_out[r] = comp;
        }
    }

    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float q_P_norm_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < rank; ++j) {
            float val = s_qP[j];
            q_P_norm_sq += val*val;
        }
        float perp_sq = query_norm_sq - q_P_norm_sq;
        perp_norms[root_idx] = (perp_sq > 0.0f) ? sqrtf(perp_sq) : 0.0f;
    }
}

// Zero-sync Parent Projection Kernel:
// - Directly uses survivor_mask so it can be launched on stream without CPU sync
__global__ void parent_projection_direct_kernel(
    const float* __restrict__ query,              // [D]
    const float query_norm_sq,                    // ||q||_2^2
    const float* __restrict__ bases,              // [num_roots, D, rank]
    const int* __restrict__ survivor_mask,        // [num_roots]
    const int num_roots,
    const int dim,
    const int rank,
    float* __restrict__ projected_queries,        // [num_roots, rank]
    float* __restrict__ perp_norms                // [num_roots]
) {
    int root_idx = blockIdx.x;
    if (root_idx >= num_roots) return;
    if (survivor_mask[root_idx] == 0) return;

    const float* __restrict__ basis_ptr = bases + (size_t)root_idx * dim * rank;
    float* __restrict__ q_P_out = projected_queries + (size_t)root_idx * rank;

    extern __shared__ float s_qP[]; // size: rank floats

    int lane = threadIdx.x; // 0..31
    int r = threadIdx.y;    // 0..rank-1

    if (r < rank) {
        float comp = 0.0f;
        for (int d = lane; d < dim; d += 32) {
            comp += query[d] * basis_ptr[d * rank + r];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            comp += __shfl_down_sync(0xffffffff, comp, offset);
        }

        if (lane == 0) {
            s_qP[r] = comp;
            q_P_out[r] = comp;
        }
    }

    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float q_P_norm_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < rank; ++j) {
            float val = s_qP[j];
            q_P_norm_sq += val * val;
        }
        float perp_sq = query_norm_sq - q_P_norm_sq;
        perp_norms[root_idx] = (perp_sq > 0.0f) ? sqrtf(perp_sq) : 0.0f;
    }
}

// device cnt parent proj kernel
// reads num_survivors directly from gpu vram without cpc sync
__global__ void parent_projection_device_count_kernel(
    const float* __restrict__ query,     // [D]
    const float query_norm_sq,           // ||q||_2^2
    const float* __restrict__ bases,     // [num_roots, D, rank]
    const int* __restrict__ survivor_indices, // [num_roots]
    const int* __restrict__ num_survivors_ptr, // [1] in device memory
    const int dim,
    const int rank,
    float* __restrict__ projected_queries, // [num_roots, rank]
    float* __restrict__ perp_norms // [num_roots]
) {
    int num_survivors = *num_survivors_ptr;
    int s_idx = blockIdx.x;
    if (s_idx >= num_survivors) return;

    int root_idx = survivor_indices[s_idx];
    const float* __restrict__ basis_ptr = bases + (size_t)root_idx * dim * rank;
    float* __restrict__ q_P_out = projected_queries + (size_t)root_idx * rank;

    extern __shared__ float s_qP[];

    int lane = threadIdx.x;
    int r = threadIdx.y;

    if (r < rank) {
        float comp = 0.0f;
        for (int d = lane; d < dim; d += 32) {
            comp += query[d] * basis_ptr[d * rank + r];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            comp += __shfl_down_sync(0xffffffff, comp, offset);
        }

        if (lane == 0) {
            s_qP[r] = comp;
            q_P_out[r] = comp;
        }
    }

    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float q_P_norm_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j < rank; ++j) {
            float val = s_qP[j];
            q_P_norm_sq += val*val;
        }
        float perp_sq = query_norm_sq - q_P_norm_sq;
        perp_norms[root_idx] = (perp_sq > 0.0f) ? sqrtf(perp_sq) : 0.0f;
    }
}

} // namespace circuit