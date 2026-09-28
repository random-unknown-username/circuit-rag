#include <cuda_runtime.h>
#include "../include/circuit/dispatch.h"
#include <algorithm>

namespace circuit {

// kernel declaration
__global__ void evaluate_root_bounds_kernel(
    const float* __restrict__ query,
    const float* __restrict__ root_centers,
    const float* __restrict__ root_radii,
    const float query_norm,
    const float threshold,
    const int num_roots,
    const int dim,
    int* __restrict__ survivor_mask,
    int* __restrict__ survivor_indices,
    int* __restrict__ num_survivors,
    float* __restrict__ root_dot_products
);

__global__ void parent_projection_kernel(
    const float* __restrict__ query,
    const float query_norm_sq,
    const float* __restrict__ bases,
    const int* __restrict__ survivor_indices,
    const int num_survivors,
    const int dim,
    const int rank,
    float* __restrict__ projected_queries,
    float* __restrict__ perp_norms
);

__global__ void evalute_child_bounds_kernel(
    const float* __restrict__ projected_queries,
    const float* __restrict__ perp_norms,
    const float* __restrict__ root_dot_products,
    const float* __restrict__ child_intervals_l,
    const float* __restrict__ child_intervals_u,
    const float* __restrict__ child_radii,
    const int* __restrict__ child_parent_ids,
    const int* __restrict__ root_survivor_mask,
    const float threshold,
    const int num_total_children,
    const int rank,
    int* __restrict__ child_survivor_mask,
    int* __restrict__ child_survivor_indices,
    int* __restrict__ num_child_survivors
);

__global__ void evaluate_child_bounds_kernel(
    const float* __restrict__ projected_queries,
    const float* __restrict__ perp_norms,
    const float* __restrict__ root_dot_products,
    const float* __restrict__ child_intervals_l,
    const float* __restrict__ child_intervals_u,
    const float* __restrict__ child_radii,
    const int* __restrict__ child_parent_ids,
    const int* __restrict__ root_survivor_mask,
    const float threshold,
    const int num_total_children,
    const int rank,
    int* __restrict__ child_survivor_mask,
    int* __restrict__ child_survivor_indices,
    int* __restrict__ num_child_survivors
);

__global__ void parent_projection_direct_kernel(
    const float* __restrict__ query,
    const float query_norm_sq,
    const float* __restrict__ bases,
    const int* __restrict__ survivor_mask,
    const int num_roots,
    const int dim,
    const int rank,
    float* __restrict__ projected_queries,
    float* __restrict__ perp_norms
);

__global__ void parent_projection_device_count_kernel(
    const float* __restrict__ query,
    const float query_norm_sq,
    const float* __restrict__ bases,
    const int* __restrict__ survivor_indices,
    const int* __restrict__ num_survivors_ptr,
    const int dim,
    const int rank,
    float* __restrict__ projected_queries,
    float* __restrict__ perp_norms
);

__global__ void score_surviving_leaves_fast_kernel(
    const float* __restrict__ query,
    const float* __restrict__ leaf_vectors,
    const int* __restrict__ leaf_offsets,
    const int* __restrict__ leaf_sizes,
    const int* __restrict__ child_survivor_indices,
    const int num_surviving_children,
    const int dim,
    float* __restrict__ vector_scores,
    int* __restrict__ scored_vector_ids,
    int* __restrict__ total_vectors_scored
);

__global__ void score_surviving_leaves_direct_kernel(
    const float* __restrict__ query,
    const float* __restrict__ leaf_vectors,
    const int* __restrict__ leaf_offsets,
    const int* __restrict__ leaf_sizes,
    const int* __restrict__ child_survivor_mask,
    const int num_total_children,
    const int dim,
    float* __restrict__ vector_scores
);

__global__ void score_surviving_leaves_device_count_kernel(
    const float* __restrict__ query,
    const float* __restrict__ leaf_vectors,
    const int* __restrict__ leaf_offsets,
    const int* __restrict__ leaf_sizes,
    const int* __restrict__ child_survivor_indices,
    const int* __restrict__ num_child_survivors_ptr,
    const int dim,
    float* __restrict__ vector_scores
);

void launch_evalute_root_bounds(
    const float* query,
    const float* root_centers,
    const float* root_radii,
    float query_norm,
    float threshold,
    int num_roots,
    int dim,
    int* survivor_mask,
    int* survivor_indices,
    int* num_survivors,
    float* root_dot_products,
    cudaStream_t stream
) {
    int threads = 128;
    int blocks = num_roots;
    evaluate_root_bounds_kernel<<<blocks, threads, 0, stream>>>(
        query, root_centers, root_radii, query_norm, threshold, num_roots, dim, survivor_mask, survivor_indices, num_survivors, root_dot_products
    );
}

void launch_parent_projection(
    const float* query,
    float query_norm_sq,
    const float* bases,
    const int* survivor_indices,
    int num_surviving_roots,
    int dim,
    int rank,
    float* projected_queries,
    float* perp_norms,
    cudaStream_t stream
) {
    if (num_surviving_roots <= 0) return;
    dim3 block(32, rank);
    size_t smem = rank * sizeof(float);
    parent_projection_kernel<<<num_surviving_roots, block, smem, stream>>>(
        query, query_norm_sq, bases, survivor_indices, 
        num_surviving_roots, dim, rank, projected_queries, perp_norms
    );
}

void launch_evaluate_child_bounds(
    const float* projected_queries,
    const float* perp_norms,
    const float* root_dot_products,
    const float* child_intervals_l,
    const float* child_intervals_u,
    const float* child_radii,
    const int* child_parent_ids,
    const int* root_survivor_mask,
    float threshold,
    int num_total_children,
    int rank,
    int* child_survivor_mask,
    int* child_survivor_indices,
    int* num_child_survivors,
    cudaStream_t stream
) {
    int threads = 128;
    int blocks = (num_total_children + threads - 1) / threads;
    evaluate_child_bounds_kernel<<<blocks, threads, 0, stream>>>(
        projected_queries, perp_norms, root_dot_products,
        child_intervals_l, child_intervals_u, child_radii,
        child_parent_ids, root_survivor_mask, threshold,
        num_total_children, rank,
        child_survivor_mask, child_survivor_indices, num_child_survivors
    );
}

void launch_score_surviving_leaves(
    const float* query,
    const float* leaf_vectors,
    const int* leaf_offsets,
    const int* leaf_sizes,
    const int* child_survivor_indices,
    int num_surviving_children,
    int dim,
    float* vector_scores,
    int* scored_vector_ids,
    int* total_vectors_scored,
    cudaStream_t stream
) {
    if (num_surviving_children <= 0) return;
    int threads = 128;
    dim3 grid(16, std::min(num_surviving_children, 2048));
    size_t smem = (dim / 4) * sizeof(float4);
    score_surviving_leaves_fast_kernel<<<grid, threads, smem, stream>>>(
        query, leaf_vectors, leaf_offsets, leaf_sizes,
        child_survivor_indices, num_surviving_children, dim,
        vector_scores, scored_vector_ids, total_vectors_scored
    );
}

void launch_parent_projection_direct(
    const float* query,
    float query_norm_sq,
    const float* bases,
    const int* survivor_mask,
    int num_roots,
    int dim,
    int rank,
    float* projected_queries,
    float* perp_norms,
    cudaStream_t stream
) {
    if (num_roots <= 0) return;
    dim3 block(32, rank);
    size_t smem = rank * sizeof(float);
    parent_projection_direct_kernel<<<num_roots, block, smem, stream>>>(
        query, query_norm_sq, bases, survivor_mask, 
        num_roots, dim, rank, projected_queries, perp_norms
    );
}


void launch_score_surviving_leaves_direct(
    const float* query,
    const float* leaf_vectors,
    const int* leaf_offsets,
    const int* leaf_sizes,
    const int* child_survivor_mask,
    int num_total_children,
    int dim,
    float* vector_scores,
    cudaStream_t stream
) {
    if (num_total_children <= 0) return;
    int threads = 128;
    dim3 grid(16, std::min(num_total_children, 1024));
    size_t smem = (dim / 4) * sizeof(float4);
    score_surviving_leaves_direct_kernel<<<grid, threads, smem, stream>>>(
        query, leaf_vectors, leaf_offsets, leaf_sizes,
        child_survivor_mask, num_total_children, dim, vector_scores
    );
}

void launch_hierarchcial_search(
    const float* query,
    const float* root_centers,
    const float* root_radii,
    const float* bases,
    const float* child_intervals_l,
    const float* child_intervals_u,
    const float* child_radii,
    const int* child_parent_ids,
    const float* leaf_vectors,
    const int* leaf_offsets,
    const int* leaf_sizes,
    int num_roots,
    int num_total_children,
    int dim,
    int rank,
    float query_norm,
    float query_norm_sq,
    float threshold,
    int* survivor_mask,
    int* survivor_indices,
    int* num_survivors,
    float* root_dot_products,
    float* projected_queries,
    float* perp_norms,
    int* child_survivor_mask,
    int* child_survivor_indices,
    int* num_child_survivors,
    float* vector_scores,
    cudaStream_t stream
) {
    // eval root bounds
    evaluate_root_bounds_kernel<<<num_roots, 128, 0, stream>>>(
        query, root_centers, root_radii, query_norm, threshold, 
        num_roots, dim, survivor_mask, survivor_indices, num_survivors, root_dot_products
    );

    // parent proj on surviving roots
    dim3 block_proj(32, rank);
    size_t smem_proj = rank * sizeof(float);
    parent_projection_device_count_kernel<<<std::min(num_roots, 512), block_proj, smem_proj, stream>>>(
        query, query_norm_sq, bases, survivor_indices, num_survivors, dim, rank, projected_queries, perp_norms
    );

    // eval child bounds 
    int child_threads = 128;
    int child_blocks = (num_total_children + child_threads - 1) / child_threads;
    evaluate_child_bounds_kernel<<<child_blocks, child_threads, 0, stream>>>(
        projected_queries, perp_norms, root_dot_products,
        child_intervals_l, child_intervals_u, child_radii,
        child_parent_ids, survivor_mask, threshold,
        num_total_children, rank,
        child_survivor_mask, child_survivor_indices, num_child_survivors
    );

    // direct leaf scoring on surviving children (vetorized float4 SIMD)
    int score_threads = 128;
    dim3 grid_score(16, std::min(num_total_children, 256));
    size_t smem_score = (dim/4)*sizeof(float4);
    score_surviving_leaves_device_count_kernel<<<grid_score, score_threads, smem_score, stream>>>(
        query, leaf_vectors, leaf_offsets, leaf_sizes,
        child_survivor_indices, num_child_survivors, dim, vector_scores
    );
}


__global__ void flat_exact_score_fast_kernel(
    const float* __restrict__ query,
    const float* __restrict__ corpus,
    int N,
    int dim,
    float* __restrict__ scores,
);

void launch_flat_exact_score(
    const float* query,
    const float* corpus,
    int N,
    int dim,
    float* scores,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = 64;
    size_t smem = (dim/4)*sizeof(float4);
    flat_exact_score_fast_kernel<<blocks, threads, smem, stream>>>(
        query, corpus, N, dim, scores
    );
}

} // namespace circuit
