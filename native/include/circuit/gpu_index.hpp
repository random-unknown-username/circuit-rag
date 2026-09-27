#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <map>
#include "circuit.h"

struct TensorMeta {
    std::string dtype;
    std::vector<int> shape;
    size_t offset;
    size_t nbytes;
};

struct circuit_index_opaque {
    // Metadata 
    int dim = 0;
    int rank = 0;
    int num_roots = 0;
    int num_total_children = 0;
    int N = 0;

    // mmap file handle
    int fd = -1;
    void* mmap_ptr = nullptr;
    size_t file_size = 0;

    // GPU device tensors
    float* dev_root_centers = nullptr;
    float* dev_root_radii = nullptr;
    float* dev_bases = nullptr;
    float* dev_child_intervals_l = nullptr;
    float* dev_child_intervals_u = nullptr;
    float* dev_child_radii = nullptr;
    int* dev_child_parent_ids = nullptr;
    int* dev_leaf_offsets = nullptr;
    int* dev_leaf_sizes = nullptr;
    float* dev_leaf_vectors = nullptr;
    int* dev_original_ids = nullptr;

    // GPU pre alloc search workspace
    int* ws_survivor_mask = nullptr;
    int* ws_survivor_indices = nullptr;
    int* ws_num_survivors = nullptr;
    float* ws_root_dot_products = nullptr;
    float* ws_projected_queries = nullptr;
    float* ws_perp_norms = nullptr;
    int* ws_child_survivor_mask = nullptr;
    int* ws_child_survivor_indices = nullptr;
    int* ws_num_child_survivors = nullptr;
    float* ws_vector_scores = nullptr;

    // pinned query and top-k buffers
    float* dev_query = nullptr;
    float* dev_top_scores = nullptr;
    int* dev_top_ids = nullptr;

    // exec stream cuBLAS graph and handle
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_warmed_up = false;

    // lifecycle methods
    bool load(const char* filepath, int device_id);
    void warmup_cuda_graph(int k = 10);
    int search(const float* query, int k, float* out_scores, int* out_ids, circuit_trace_t* trace);
    int search_device(const float* in_dev_query, int k, float* out_dev_scores, int* out_dev_ids, circuit_trace_t* trace);
    void destroy();
};

