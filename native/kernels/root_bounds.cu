#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/circuit/numerics.cuh"

namespace circuit {

// root bounds kernel
// - vectorized 128bit float4 loads for query and root centers
// - directly writes root_dot_products (eliminating external touch.matmul)
// - fast warp shuffle and shared memory reduction
__global__ void evaluate_root_bounds_kernel(
    const float* __restrict__ query,         // [D]
    const float* __restrict__ root_centers,  // [num_roots, D]
    const float* __restrict__ root_radii,    // [num_roots] (delta + rho)
    const float query_norm,
    const float threshold,
    const int num_roots,
    const int dim,
    int* __restrict__ survivors_mask,
    int* __restrict__ survivors_indices,
    int* __restrict__ num_survivors,
    float* __restrict__ root_dot_products
) {
    int root_idx = blockIdx.x;
    if (root_idx >= num_roots) return;

    const float4* __restrict__ q4 = reinterpret_cast<const float4*>(query);
    const float4* __restrict__ c4 = reinterpret_cast<const float4*>(root_centers + (size_t)root_idx * dim);
    int dim4 = dim / 4;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < dim4; d += blockDim.x) {
        float4 v_q = q4[d];
        float4 v_c = c4[d];
        sum += v_q.x * v_c.x + v_q.y * v_c.y + v_q.z * v_c.z + v_q.w * v_c.w;
    }

    // remainder loop
    for (int d = dim4 * 4 + threadIdx.x; d < dim; d += blockDim.x) {
        sum += query[d] * (root_centers + (size_t)root_idx * dim)[d];
    }

    sum = blockReduceSum(sum);

    if (threadIdx.x == 0) {
        root_dot_products[root_idx] = sum;
        float bound = sum + query_norm * root_radii[root_idx];
        if (bound >= threshold) {
            survivors_mask[root_idx] = 1;
            int pos = atomicAdd(num_survivors, 1);
            survivors_indices[pos] = root_idx;
        } else {
            survivors_mask[root_idx] = 0;
        }
    }
}

} // namespace circuit