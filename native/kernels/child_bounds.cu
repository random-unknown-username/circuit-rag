#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/numerics.cuh"

namespace circuit {

// optimized child bounds kernel
// vectorized 128bit float4 loads for rank-8 coordinate boxes
// fully unrolled SIMD interval evaluation
// fast rejection of pruned parents
__global__ void evaluate_child_bounds_kernel(
    const float* __restrict__ projected_queries, // [num_roots, rank]
    const float* __restrict__ perp_norms,        // [num_roots]
    const float* __restrict__ root_dot_products, // [num_roots] (q^T c_P)
    const float* __restrict__ child_intervals_l, // [num_total_children, rank]
    const float* __restrict__ child_intervals_u, // [num_total_children, rank]
    const float* __restrict__ child_radii,       // [num_total_children]
    const int* __restrict__ child_parent_ids,    // [num_total_children]
    const int* __restrict__ root_survivor_mask,  // [num_roots]
    const float threshold,
    const int num_total_children,
    const int rank,
    int* __restrict__ child_survivor_mask,       // [num_total_children]
    int* __restrict__ child_survivor_indices,    // [num_total_children]
    int* __restrict__ num_child_survivors        // [1]
) { 
    int child_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (child_idx >= num_total_children) return;

    int parent_id = child_parent_ids[child_idx];
    if (root_survivor_mask[parent_id] == 0) {
        child_survivor_mask[child_idx] = 0;
        return;
    }

    const float* __restrict__ q_P = projected_queries + (size_t)parent_id * rank;
    float rho_B = child_radii[child_idx];
    float q_perp_norm = perp_norms[parent_id];
    float q_dot_cP = root_dot_products[parent_id];

    float interval_sum = 0.0f;

    // fast path rank = 8 (2 128bit float3 loads/interval)
    if (rank == 8) { 
        const float4 __restrict__ l4 = reinterpret_cast<const float4*>(child_intervals_l + (size_t)child_idx * 8);
        const float4 __restrict__ u4 = reinterpret_cast<const float4*>(child_intervals_u + (size_t)child_idx * 8);
        const float4 __restrict__ q4 = reinterpret_cast<const float4*>(q_P);

        float4 l0 = l4[0];
        float4 l1 = l4[1];
        float4 u0 = u4[0];
        float4 u1 = u4[1];
        float4 q0 = q4[0];
        float4 q1 = q4[1];

        interval_sum = fmaxf(q0.x * l0.x, q0.x * u0.x) +
                    fmaxf(q0.y * l0.y, q0.y * u0.y) +
                    fmaxf(q0.z * l0.z, q0.z * u0.z) +
                    fmaxf(q0.w * l0.w, q0.w * u0.w) +
                    fmaxf(q1.x * l1.x, q1.x * u1.x) +
                    fmaxf(q1.y * l1.y, q1.y * u1.y) +
                    fmaxf(q1.z * l1.z, q1.z * u1.z) +
                    fmaxf(q1.w * l1.w, q1.w * u1.w);
    } else {
        const float* __restrict__ l_B = child_intervals_l + (size_t)child_idx * rank;
        const float* __restrict__ u_B = child_intervals_u + (size_t)child_idx * rank;
        #pragma unroll
        for (int j = 0; j < rank; ++j) {
            float q_j = q_P[j];
            interval_sum += fmaxf(q_j * l_B[j], q_j * u_B[j]);
        }
    }

    float upper_bound = q_dot_cP + interval_sum + q_perp_norm * rho_B;

    if (upper_bound >= threshold) {
        child_survivor_mask[child_idx] = 1;
        int pos = atomicAdd(num_child_survivors, 1);
        child_survivor_indices[pos] = child_idx;
    } else {
        child_survivor_mask[child_idx] = 0;
    }
}


} // namespace circuit