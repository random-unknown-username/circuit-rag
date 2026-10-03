#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <optional>
#include <algorithm>
#include "../include/circuit/dispatch.h"

// cpp wrapper for pytorch
std::vector<torch::Tensor> evaluate_root_bounds_cuda(
    torch::Tensor query, // [D]
    torch::Tensor root_centers, // [num_roots, D]
    torch::Tensor root_radii, // [num_roots]
    float threshold,
    float query_norm = 1.0f
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(root_centers.is_cuda(), "root_centers must be CUDA tensor");
    TORCH_CHECK(root_radii.is_cuda(), "root_radii must be a CUDA tensor");

    int dim = query.size(0);
    int num_roots = root_centers.size(0);

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(query.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());

    auto survivor_mask = torch::zeros({num_roots}, options_int);
    auto survivor_indices = torch::zeros({num_roots}, options_int);
    auto num_survivors = torch::zeros({1}, options_int);
    auto root_dot_products = torch::empty({num_roots}, options_float);

    circuit::launch_evaluate_root_bounds(
        query.data_ptr<float>(),
        root_centers.data_ptr<float>(),
        root_radii.data_ptr<float>(),
        query_norm,
        threshold,
        num_roots,
        dim,
        survivor_mask.data_ptr<int>(),
        survivor_indices.data_ptr<int>(),
        num_survivors.data_ptr<int>(),
        root_dot_products.data_ptr<float>()
    );

    return {survivor_mask, survivor_indices, num_survivors, root_dot_products};
}

std::vector<torch::Tensor> project_parents_cuda(
    torch::Tensor query, // [D]
    torch::Tensor bases, // [num_bases, D, rank]
    torch::Tensor survivor_indices, // [num_survivors]
    int num_survivors,
    int rank,
    float query_norm_sq = 1.0f
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA tensor");
    TORCH_CHECK(bases.is_cuda(), "bases must be CUDA tensor");

    int dim = query.size(0);
    int num_bases = bases.size(0);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    auto projected_queries = torch::zeros({num_bases, rank}, options_float);
    auto perp_norms = torch::zeros({num_bases}, options_float);

    if (num_survivors > 0) {
        circuit::launch_parent_projection(
            query.data_ptr<float>(),
            query_norm_sq,
            bases.data_ptr<float>(),
            survivor_indices.data_ptr<int>(),
            num_survivors,
            dim,
            rank,
            projected_queries.data_ptr<float>(),
            perp_norms.data_ptr<float>()
        );
    }
    return {projected_queries, perp_norms};
}

std::vector<torch::Tensor> evaluate_child_bounds_cuda(
    torch::Tensor projected_queries,  // [num_roots, rank]
    torch::Tensor perp_norms,         // [num_roots]
    torch::Tensor root_dot_products,  // [num_roots]
    torch::Tensor child_intervals_l,  // [num_total_children, rank]
    torch::Tensor child_intervals_u,  // [num_total_children, rank]
    torch::Tensor child_radii,        // [num_total_children]
    torch::Tensor child_parent_ids,   // [num_total_children]
    torch::Tensor root_survivor_mask, // [num_roots]
    float threshold,
    int rank
) {
    int num_total_children = child_radii.size(0);

    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(projected_queries.device());
    auto child_survivor_mask = torch::zeros({num_total_children}, options_int);
    auto child_survivor_indices = torch::zeros({num_total_children}, options_int);
    auto num_child_survivors = torch::zeros({1}, options_int);

    circuit::launch_evaluate_child_bounds(
        projected_queries.data_ptr<float>(),
        perp_norms.data_ptr<float>(),
        root_dot_products.data_ptr<float>(),
        child_intervals_l.data_ptr<float>(),
        child_intervals_u.data_ptr<float>(),
        child_radii.data_ptr<float>(),
        child_parent_ids.data_ptr<int>(),
        root_survivor_mask.data_ptr<int>(),
        threshold,
        num_total_children,
        rank,
        child_survivor_mask.data_ptr<int>(),
        child_survivor_indices.data_ptr<int>(),
        num_child_survivors.data_ptr<int>()
    );
    return {child_survivor_mask, child_survivor_indices, num_child_survivors};
}

std::vector<torch::Tensor> score_surviving_leaves_cuda(
    torch::Tensor query, // [D]
    torch::Tensor leaf_vectors, // [N,D]
    torch::Tensor leaf_offsets, // [num_tota_children]
    torch::Tensor leaf_sizes, // [num_total_children]
    torch::Tensor child_survivor_indices, // [num_survivors]
    int num_survivors
) {
    int N = leaf_vectors.size(0);
    int dim = query.size(0);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());

    auto vector_scores = torch::full({N}, -1e9f, options_float);

    if (num_survivors > 0) {
        circuit::launch_score_surviving_leaves(
            query.data_ptr<float>(),
            leaf_vectors.data_ptr<float>(),
            leaf_offsets.data_ptr<int>(),
            leaf_sizes.data_ptr<int>(),
            child_survivor_indices.data_ptr<int>(),
            num_survivors,
            dim,
            vector_scores.data_ptr<float>()
        );
    }
    return {vector_scores};
}

torch::Tensor hierarchical_search_cuda(
    torch::Tensor query,              // [D]
    torch::Tensor root_centers,       // [num_roots, D]
    torch::Tensor root_radii,         // [num_roots]
    torch::Tensor bases,              // [num_roots, D, rank]
    torch::Tensor child_intervals_l,  // [num_total_children, rank]
    torch::Tensor child_intervals_u,  // [num_total_children, rank]
    torch::Tensor child_radii,        // [num_total_children]
    torch::Tensor child_parent_ids,   // [num_total_children]
    torch::Tensor leaf_vectors,       // [N, D]
    torch::Tensor leaf_offsets,       // [num_total_children]
    torch::Tensor leaf_sizes,         // [num_total_children]
    torch::Tensor survivor_mask,      // [num_roots]
    torch::Tensor survivor_indices,   // [num_roots]
    torch::Tensor num_survivors,      // [1]
    torch::Tensor root_dot_products,  // [num_roots]
    torch::Tensor projected_queries,  // [num_roots, rank]
    torch::Tensor perp_norms,         // [num_roots]
    torch::Tensor child_survivor_mask,// [num_total_children]
    torch::Tensor child_survivor_indices,// [num_total_children]
    torch::Tensor num_child_survivors,// [1]
    torch::Tensor vector_scores,      // [N]
    float threshold,
    int rank,
    float query_norm = 1.0f
) {
    int dim = query.size(0);
    int num_roots = root_centers.size(0);
    int num_total_children = child_radii.size(0);
    float query_norm_sq = query_norm * query_norm;

    // Reset scores to -1e9f
    vector_scores.fill_(-1e9f);
    num_survivors.zero_();
    num_child_survivors.zero_();

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    circuit::launch_hierarchical_search(
        query.data_ptr<float>(),
        root_centers.data_ptr<float>(),
        root_radii.data_ptr<float>(),
        bases.data_ptr<float>(),
        child_intervals_l.data_ptr<float>(),
        child_intervals_u.data_ptr<float>(),
        child_radii.data_ptr<float>(),
        child_parent_ids.data_ptr<int>(),
        leaf_vectors.data_ptr<float>(),
        leaf_offsets.data_ptr<int>(),
        leaf_sizes.data_ptr<int>(),
        num_roots,
        num_total_children,
        dim,
        rank,
        query_norm,
        query_norm_sq,
        threshold,
        survivor_mask.data_ptr<int>(),
        survivor_indices.data_ptr<int>(),
        num_survivors.data_ptr<int>(),
        root_dot_products.data_ptr<float>(),
        projected_queries.data_ptr<float>(),
        perp_norms.data_ptr<float>(),
        child_survivor_mask.data_ptr<int>(),
        child_survivor_indices.data_ptr<int>(),
        num_child_survivors.data_ptr<int>(),
        vector_scores.data_ptr<float>(),
        stream
    );

    return vector_scores;
}

torch::Tensor flat_exact_score_cuda(torch::Tensor query, torch::Tensor corpus) {
    TORCH_CHECK(query.is_cuda() && corpus.is_cuda(), "query and corpus must be CUDA tensors");
    int N = corpus.size(0);
    int dim = corpus.size(1);
    auto scores = torch::empty({N}, query.options().dtype(torch::kFloat32));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    circuit::launch_flat_exact_score(
        query.data_ptr<float>(), corpus.data_ptr<float>(), N, dim, scores.data_ptr<float>(), stream
    );
    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evaluate_root_bounds", &evaluate_root_bounds_cuda, "Evaluate root sphere certificates (CUDA)",
          py::arg("query"), py::arg("root_centers"), py::arg("root_radii"), py::arg("threshold"), py::arg("query_norm") = 1.0f);
    m.def("project_parents", &project_parents_cuda, "Project query onto surviving parents (CUDA)",
          py::arg("query"), py::arg("bases"), py::arg("survivor_indices"), py::arg("num_survivors"), py::arg("rank"), py::arg("query_norm_sq") = 1.0f);
    m.def("evaluate_child_bounds", &evaluate_child_bounds_cuda, "Evaluate child low-rank bounds (CUDA)",
          py::arg("projected_queries"), py::arg("perp_norms"), py::arg("root_dot_products"),
          py::arg("child_intervals_l"), py::arg("child_intervals_u"), py::arg("child_radii"),
          py::arg("child_parent_ids"), py::arg("root_survivor_mask"), py::arg("threshold"), py::arg("rank"));
    m.def("score_surviving_leaves", &score_surviving_leaves_cuda, "Score surviving leaf vectors (CUDA)",
          py::arg("query"), py::arg("leaf_vectors"), py::arg("leaf_offsets"), py::arg("leaf_sizes"),
          py::arg("child_survivor_indices"), py::arg("num_survivors"));
    m.def("flat_exact_score", &flat_exact_score_cuda, "Flat exact scoring (CUDA/cuBLAS)",
          py::arg("query"), py::arg("corpus"));
    m.def("hierarchical_search", &hierarchical_search_cuda, "Zero-sync unified hierarchical search (CUDA)",
          py::arg("query"), py::arg("root_centers"), py::arg("root_radii"), py::arg("bases"),
          py::arg("child_intervals_l"), py::arg("child_intervals_u"), py::arg("child_radii"),
          py::arg("child_parent_ids"), py::arg("leaf_vectors"), py::arg("leaf_offsets"), py::arg("leaf_sizes"),
          py::arg("survivor_mask"), py::arg("survivor_indices"), py::arg("num_survivors"),
          py::arg("root_dot_products"), py::arg("projected_queries"), py::arg("perp_norms"),
          py::arg("child_survivor_mask"), py::arg("child_survivor_indices"), py::arg("num_child_survivors"),
          py::arg("vector_scores"), py::arg("threshold"), py::arg("rank"), py::arg("query_norm") = 1.0f);
}
