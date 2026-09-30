#pragma once

#include <cuda_runtime.h>

namespace circuit {

void launch_evaluate_root_bounds(   
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
    cudaStream_t stream = 0
);

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
    cudaStream_t stream = 0 
);

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
    cudaStream_t stream = 0
);

void launch_score_surviving_leaves(
    const float* query,
    const float* leaf_vectors,
    const int* leaf_offsets,
    const int* leaf_sizes,
    const int* child_survivor_indices,
    int num_surviving_children,
    int dim,
    float* vector_scores,
    cudaStream_t stream = 0
);

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
    cudaStream_t stream = 0
);

void launch_score_surviving_leaves_direct(
    const float* query,
    const float* leaf_vectors,
    const int* leaf_offsets,
    const int* leaf_sizes,
    const int* child_survivor_mask,
    int num_total_children,
    int dim,
    float* vector_scores,
    cudaStream_t stream = 0
);

void launch_hierarchical_search(
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
    cudaStream_t stream = 0
);

void launch_select_topk(
    const float* scores,
    const int* original_ids,
    int N,
    int K,
    float* out_scores,
    int* out_ids,
    cudaStream_t stream = 0
);

void launch_flat_exact_score(
    const float* query,
    const float* corpus,
    int N,
    int dim,
    float* scores,
    cudaStream_t stream = 0
);

} // namespace circuit