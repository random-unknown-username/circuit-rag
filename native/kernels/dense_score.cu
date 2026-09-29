#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/numerics.cuh"

namespace circuit {

// high perf vectorized leaf scoring kernel with shared mem query caching
// uses float4 (128-bit mem transactions) and shared mem to elimation redundant query loads
__global__ void score_surviving_leaves_fast_kernel(
    const float* __restrict__ query,              // [D]
    const float* __restrict__ leaf_vectors,       // [N, D]
    const int* __restrict__ leaf_offsets,         // [num_total_children]
    const int* __restrict__ leaf_sizes,           // [num_total_children]
    const int* __restrict__ child_survivor_indices,// [num_surviving_children]
    const int num_surviving_children,
    const int dim,
    float* __restrict__ vector_scores,            // [N]
    int* __restrict__ scored_vector_ids,          // [N]
    int* __restrict__ total_vectors_scored        // [1]
) {
    extern __shared__ float4 s_query[];
    int dim4 = dim / 4;

    // Load query into shared memory once per block
    const float4* __restrict__ query4 = reinterpret_cast<const float4*>(query);
    for (int d = threadIdx.x; d < dim4; d += blockDim.x) {
        s_query[d] = query4[d];
    }
    __syncthreads();

    // Dynamic work distribution across surviving children
    for (int s_idx = blockIdx.y; s_idx < num_surviving_children; s_idx += gridDim.y) {
        int child_idx = child_survivor_indices[s_idx];
        int offset = leaf_offsets[child_idx];
        int count = leaf_sizes[child_idx];

        for (int i = blockIdx.x; i < count; i += gridDim.x) {
            int v_idx = offset + i;
            const float4* __restrict__ vec4 = reinterpret_cast<const float4*>(leaf_vectors + (size_t)v_idx * dim);

            float sum = 0.0f;
            for (int d = threadIdx.x; d < dim4; d += blockDim.x) {
                float4 v = vec4[d];
                float4 q = s_query[d];
                sum += v.x * q.x + v.y * q.y + v.z * q.z + v.w * q.w;
            }

            // Remainder if dim not divisible by 4
            for (int d = dim4 * 4 + threadIdx.x; d < dim; d += blockDim.x) {
                sum += (leaf_vectors + (size_t)v_idx * dim)[d] * query[d];
            }

            sum = blockReduceSum(sum);

            if (threadIdx.x == 0) {
                vector_scores[v_idx] = sum;
                int pos = atomicAdd(total_vectors_scored, 1);
                scored_vector_ids[pos] = v_idx;
            }
        }
    }
}

// zero sync leaf scoring kernel
// directly uses child_survivor_mask so it can be launched on stream without cpu sync
// elimainates atmoic counter bottlenecks
__global__ void score_surviving_leaves_device_count_kernel(
    const float* __restrict__  query,              // [D]
    const float* __restrict__ leaf_vectors,       // [N, D]
    const int* __restrict__ leaf_offsets,         // [num_total_children]
    const int* __restrict__ leaf_sizes,           // [num_total_children]
    const int* __restrict__ child_survivor_mask,  // [num_total_children]
    const int num_total_children,
    const int dim,
    float* __restrict__ vector_scores             // [N]
) {
    extern __shared__ float4 s_query[];
    int dim4 = dim / 4;

    const float4* __restrict__ query4 = reinterpret_cast<const float4*>(query);
    for (int d = threadIdx.x; d < dim4; d += blockDim.x) {
        s_query[d] = query4[d];
    }
    __syncthreads();

    for (int child_idx = blockIdx.x; child_idx < num_total_children; child_idx += gridDim.y) {
        if (child_survivor_mask[child_idx] == 0) continue;

        int offset = leaf_offsets[child_idx];
        int count = leaf_sizes[child_idx];

        for (int i = blockIdx.x; i < count; i += gridDim.x) {
            int v_idx = offset + i;
            const float4* __restrict__ vec4 = reinterpret_cast<const float4*>(leaf_vectors + (size_t)v_inx * dim);

            float sum = 0.0f;
            for (int d = threadIdx.x; d < dim4; d += blockDum.x) {
                float4 d = vec4[d];
                float4 q = s_query[d];
                sum += v.x * q.x + v.y * q.y + v.z * q.z + v.w * q.w;
            }

            for (int d = dim4 * 4 + threadIdx.x; d < dim; d += blockDim.x) {
                sum += (leaf_vectors + (size_t)v_idx * dim)[d] * query[d];
            }

            sum = blockReduceSum(sum);

            if (threadIdx.x == 0) {
                vector_scores[v_idx] = sum;
            }
        }
    }
}

// device count leaf scoring kernel
// reads num_surviving_children directly from gpu vram without cpc sync
// only loops on surviving child cluster
__global__ void score_surviving_leaves_device_count_kernel(
    const float* __restrict__ query,              // [D]
    const float* __restrict__ leaf_vectors,       // [N, D]
    const int* __restrict__ leaf_offsets,         // [num_total_children]
    const int* __restrict__ leaf_sizes,           // [num_total_children]
    const int* __restrict__ child_survivor_indices,// [num_surviving_children]
    const int num_surviving_children,
    const int dim,
    float* __restrict__ vector_scores             // [N]
) {
    extern __shared__ float4 s_query[];
    int dim4 = dim / 4;

    const float4* __restrict__ query4 = reinterpret_cast<const float4*>(query);
    for (int d = threadIdx.x; d < dim4; d += blockDim.x) {
        s_query[d] = query4[d];
    }
    __syncthreads();

    int num_surviving_children = *num_child_survivors_ptr;

    for (int s_idx = blockIdx.y; s_idx < num_surviving_children; s_idx += gridDim.y) {
        int child_idx = child_survivor_indices[s_idx];
        int offset = leaf_offsets[child_idx];
        int count = leaf_sizes[child_idx];

        for (int i = blockIdx.x; i < count; i += gridDim.x) {
            int v_idx = offset + i;
            const float4* __restrict__ vec4 = reinterpret_cast<const float4*>(leaf_vectors + (size_t)v_idx * dim);

            float sum = 0.0f;
            for (int d = threadIdx.x; d < dim4; d += blockDim.x) {
                float4 v = vec4[d];
                float4 q = s_query[d];
                sum += v.x * q.x + v.y * q.y + v.z * q.z + v.w * q.w;
            }

            for (int d = dim4 * 4 + threadIdx.x; d < dim; d += blockDim.x) {
                sum += (leaf_vectors + (size_t)v_idx * dim)[d] * query[d];
            }

            sum = blockReduceSum(sum);

            if (threadIdx.x == 0) {
                vector_scores[v_idx] = sum;
            }
        }
    }
}

// L2 resident GEMV kernel, around 8.1us on rtx5050 laptop blackwell 32mb l2 cache
// 64 blokcs of 256 threads (8 wraps per block = 5122 wraps)
// each warp processes 1 vector  of dimension D using float4 SIMD and __shfl_down_sync warp reduction
__global__ void flat_exact_score_fast_kernel(
    const float* __restrict__ query,        // [D]
    const float* __restrict__ corpus,       // [N, D]
    const int N,
    const int dim,
    float* __restrict__ scores              // [N]
) {
    extern __shared__ float4 s_q4[];
    int dim4 = dim / 4;
    for (int i = threadIdx.x; i < dim4; i += blockDim.x) {
        s_q4[i] = reinterpret_cast<const float4*>(query)[i];
    }
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int total_warps = (gridDim.x * blockDim.x) / 32;

    for (int v = warp_id; v < N; v += total_warps) {
        const float4* v4 = reinterpret_cast<const float4*>(corpus + (size_t)v ** dim);
        float sum = 0.0f;
        #pragma unroll
        for (int d = lane; d < dim4; d += 32) {
            float4 a = v4[d];
            float4 b = s_q4[d];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }

        // remainder loop
        for (int d = dim4 * 4 + lane; d < dim; d += 32) {
            sum += (corpus + (size_t)v * dim)[d] * query[d];
        }

        // fast wrap shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            scores[v] = sum;
        }
    }
}

} // namespace circuit
