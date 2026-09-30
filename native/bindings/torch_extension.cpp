#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <optional>
#include <algorithm>
#include "../include/circuit/dispatch.h"
#include "../include/circuit/batched.h"
#include "../include/circuit/bitmask.h"
#include "../include/circuit/maxsim.h"

// cpp wrapper for pytorch
std::vector<torch::Tensor> circuit_search_torch(
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
    float query_norm = 1.0f
) {
    TORCH_CHECK(query.is.cuda(), "query must be CUDA tensor");
    TORCH_CHECK(bases.is_cuda(), "bases must be CUDA tensor");

    int dim = query.size(0);
    int num_bases = bases.size(0);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    auto projected_queries = torch::zeros({num_roots, rank}, options_float);
    auto perp_norms = torch::zeros({num_roots}, options_float);

    if (num_survivors > 0) {
        circuit::launch_parent_projection(
            query,data_ptr<float>(),
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
    auto num_child_survivors = torch::zero({1}, options_int);

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
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(query.device());

    auto vector_scores = torch::full({N}, -1e9f, options_float);
    auto scored_vector_ids = torch::zero({N}, options_int);
    auto total_vectors_scored = torch::zero({1}, options_int);

    if (num_survivors > 0) {
        circuit::launch_score_surviving_leaves(
            query.data_ptr<float>(),
            leaf_vectors.data_ptr<float>(),
            leaf_offsets.data_ptr<int>(),
            leaf_sizes.data_ptr<int>(),
            child_survivor_indices.data_ptr<int>(),
            num_survivors,
            dim,
            vector_scores.data_ptr<float>(),
            scored_vector_ids.data_ptr<int>(),
            total_vectors_scored.data_ptr<int>()
        );
    }
    return {vector_scores, scored_vector_ids, total_vectors_scored};
}

torch::Tensor hierarchial_search_cuda(
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

std::pair<torch::Tensor, torch::Tensor> batched_search_cuda(
    torch::Tensor queries,       // [B, D]
    torch::Tensor corpus,        // [N, D]
    std::optional<torch::Tensor> original_ids,  // [N]
    int K
) {
    TORCH_CHECK(queries.is_cuda() && corpus.is_cuda(), "queries and corpus must be CUDA tensors");
    TORCH_CHECK(queries.device() == corpus.device(), "queries and corpus must reside on the same CUDA device");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [B, D]");
    TORCH_CHECK(corpus.dim() == 2, "corpus must be 2D [N, D]");
    TORCH_CHECK(queries.size(1) == corpus.size(1), "queries and corpus dimension must match");
    TORCH_CHECK(K > 0, "K must be positive");

    int B = queries.size(0);
    int dim = queries.size(1);
    int N = corpus.size(0);
    int actual_k = std::min(K, N);

    auto queries_c = queries.contiguous();
    auto corpus_c = corpus.contiguous();

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(queries.device());

    torch::Tensor ids_i32;
    const int* orig_ids_ptr = nullptr;
    if (original_ids.has_value() && original_ids->defined()) {
        TORCH_CHECK(original_ids->is_cuda(), "original_ids must be a CUDA tensor");
        TORCH_CHECK(original_ids->device() == queries.device(), "original_ids must reside on the same CUDA device as queries");
        if (original_ids->dtype() != torch::kInt32) {
            ids_i32 = original_ids->to(torch::kInt32).contiguous();
        } else {
            ids_i32 = original_ids->contiguous();
        }
        orig_ids_ptr = ids_i32.data_ptr<int>();
    }

    auto stream = c10::cuda::getCurrentCUDAStream(queries.device().index()).stream();

    // fast: Custom 2D grid CUDA Top-K kernel for actual_k <= 256
    if (actual_k <= 256) {
        auto scores_ws = torch::empty({B, N}, options_float);
        auto out_scores = torch::empty({B, actual_k}, options_float);
        auto out_ids = torch::empty({B, actual_k}, options_int);

        circuit::launch_batched_search(
            queries_c.data_ptr<float>(),
            corpus_c.data_ptr<float>(),
            orig_ids_ptr,
            B, N, dim, actual_k,
            scores_ws.data_ptr<float>(),
            out_scores.data_ptr<float>(),
            out_ids.data_ptr<int>(),
            stream,
            /*stride_k=*/actual_k
        );

        return {out_scores, out_ids};
    } else {
        // fallback: cuBLAS dense scoring + torch::topk for actual_k > 256
        auto scores_ws = torch::empty({B, N}, options_float);
        circuit::launch_batched_dense_score(
            queries_c.data_ptr<float>(),
            corpus_c.data_ptr<float>(),
            B, N, dim,
            scores_ws.data_ptr<float>(),
            stream
        );

        auto topk_res = torch::topk(scores_ws, actual_k, /*dim=*/1);
        auto out_scores = std::get<0>(topk_res);
        auto topk_indices = std::get<1>(topk_res);

        torch::Tensor out_ids;
        if (ids_i32.defined()) {
            out_ids = ids_i32.index_select(0, topk_indices.view(-1)).view({B, actual_k});
        } else {
            out_ids = topk_indices.to(torch::kInt32);
        }

        return {out_scores, out_ids};
    }
}

torch::Tensor batched_dense_score_cuda(
    torch::Tensor queries,       // [B, D]
    torch::Tensor corpus         // [N, D]
) {
    TORCH_CHECK(queries.is_cuda() && corpus.is_cuda(), "queries and corpus must be CUDA tensors");
    TORCH_CHECK(queries.device() == corpus.device(), "queries and corpus must reside on the same CUDA device");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [B, D]");
    TORCH_CHECK(corpus.dim() == 2, "corpus must be 2D [N, D]");
    TORCH_CHECK(queries.size(1) == corpus.size(1), "queries and corpus dimension must match");

    int B = queries.size(0);
    int dim = queries.size(1);
    int N = corpus.size(0);

    auto queries_c = queries.contiguous();
    auto corpus_c = corpus.contiguous();
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
    auto scores = torch::empty({B, N}, options_float);

    auto stream = c10::cuda::getCurrentCUDAStream(queries.device().index()).stream();
    circuit::launch_batched_dense_score(
        queries_c.data_ptr<float>(),
        corpus_c.data_ptr<float>(),
        B, N, dim,
        scores.data_ptr<float>(),
        stream
    );
    return scores;
}

std::pair<torch::Tensor, torch::Tensor> batched_topk_cuda(
    torch::Tensor scores,                       // [B, N]
    std::optional<torch::Tensor> original_ids,  // [N]
    int K
) {
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(scores.dim() == 2, "scores must be 2D [B, N]");
    TORCH_CHECK(K > 0, "K must be positive");

    int B = scores.size(0);
    int N = scores.size(1);
    int actual_k = std::min(K, N);

    auto scores_c = scores.contiguous();
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(scores.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(scores.device());

    torch::Tensor ids_i32;
    const int* orig_ids_ptr = nullptr;
    if (original_ids.has_value() && original_ids->defined()) {
        TORCH_CHECK(original_ids->is_cuda(), "original_ids must be a CUDA tensor");
        TORCH_CHECK(original_ids->device() == scores.device(), "original_ids must reside on the same CUDA device as scores");
        if (original_ids->dtype() != torch::kInt32) {
            ids_i32 = original_ids->to(torch::kInt32).contiguous();
        } else {
            ids_i32 = original_ids->contiguous();
        }
        orig_ids_ptr = ids_i32.data_ptr<int>();
    }

    auto stream = c10::cuda::getCurrentCUDAStream(scores.device().index()).stream();

    // fast: Custom 2D grid CUDA Top-K kernel for actual_k <= 256
    if (actual_k <= 256) {
        auto out_scores = torch::empty({B, actual_k}, options_float);
        auto out_ids = torch::empty({B, actual_k}, options_int);

        circuit::launch_batched_select_topk(
            scores_c.data_ptr<float>(),
            orig_ids_ptr,
            B, N, actual_k,
            out_scores.data_ptr<float>(),
            out_ids.data_ptr<int>(),
            stream,
            /*stride_k=*/actual_k
        );

        return {out_scores, out_ids};
    } else {
        // fallback: torch::topk for actual_k > 256
        auto topk_res = torch::topk(scores_c, actual_k, /*dim=*/1);
        auto out_scores = std::get<0>(topk_res);
        auto topk_indices = std::get<1>(topk_res);

        torch::Tensor out_ids;
        if (ids_i32.defined()) {
            out_ids = ids_i32.index_select(0, topk_indices.view(-1)).view({B, actual_k});
        } else {
            out_ids = topk_indices.to(torch::kInt32);
        }

        return {out_scores, out_ids};
    }
}

torch::Tensor flat_score_filtered_cuda(
    torch::Tensor query,
    torch::Tensor corpus,
    torch::Tensor bitmask
) {
    TORCH_CHECK(query.is_cuda() && corpus.is_cuda() && bitmask.is_cuda(),
                "query, corpus, and bitmask must be CUDA tensors");
    TORCH_CHECK(query.device() == corpus.device() && query.device() == bitmask.device(),
                "query, corpus, and bitmask must reside on the same CUDA device");
    TORCH_CHECK(corpus.dim() == 2, "corpus must be 2D [N, D]");
    int N = corpus.size(0);
    int dim = corpus.size(1);
    TORCH_CHECK(query.numel() == dim, "query size must match corpus dimension");

    auto query_c = query.contiguous();
    auto corpus_c = corpus.contiguous();
    auto bitmask_c = bitmask.contiguous();

    torch::Tensor mask_u64;
    int expected_words = (N + 63) / 64;
    if (bitmask_c.dtype() == torch::kBool) {
        auto padded = bitmask_c.to(torch::kInt64);
        int pad_len = expected_words * 64 - bitmask_c.numel();
        if (pad_len > 0) {
            padded = torch::nn::functional::pad(padded, torch::nn::functional::PadFuncOptions({0, pad_len}).value(0));
        }
        auto reshaped = padded.view({expected_words, 64});
        auto powers_list = std::vector<int64_t>(64);
        for (int i = 0; i < 64; ++i) {
            powers_list[i] = (i < 63) ? (1LL << i) : (int64_t)0x8000000000000000ULL;
        }
        auto powers = torch::tensor(powers_list, torch::TensorOptions().dtype(torch::kInt64).device(query.device()));
        mask_u64 = torch::sum(reshaped * powers, /*dim=*/1);
    } else {
        TORCH_CHECK(bitmask_c.dtype() == torch::kInt64, "bitmask must be int64 (packed uint64) or bool");
        TORCH_CHECK(bitmask_c.numel() >= expected_words, "bitmask words count must be at least (N + 63) / 64");
        mask_u64 = bitmask_c;
    }

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    auto scores = torch::empty({N}, options_float);

    auto stream = c10::cuda::getCurrentCUDAStream(query.device().index()).stream();
    circuit::launch_flat_score_filtered(
        query_c.data_ptr<float>(),
        corpus_c.data_ptr<float>(),
        reinterpret_cast<const uint64_t*>(mask_u64.data_ptr<int64_t>()),
        N,
        dim,
        scores.data_ptr<float>(),
        stream
    );
    return scores;
}

std::pair<torch::Tensor, torch::Tensor> batched_search_filtered_cuda(
    torch::Tensor queries,   // [B, D]
    torch::Tensor corpus,  // [N, D]
    torch::Tensor bitmask, // [(N+63)/64]
    std::optional<torch::Tensor> original_ids, // [N] (nullable)
    int K
) {
    TORCH_CHECK(queries.is_cuda() && corpus.is_cuda() && bitmask.is_cuda(),
                "queries, corpus, and bitmask must be CUDA tensors");
    TORCH_CHECK(queries.device() == corpus.device() && queries.device() == bitmask.device(),
                "inputs must reside on the same CUDA device");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D [B, D]");
    TORCH_CHECK(corpus.dim() == 2, "corpus must be 2D [N, D]");
    TORCH_CHECK(queries.size(1) == corpus.size(1), "queries and corpus dimension must match");
    TORCH_CHECK(K > 0, "K must be positive");

    int B = queries.size(0);
    int dim = queries.size(1);
    int N = corpus.size(0);
    int actual_k = std::min(K, N);

    auto queries_c = queries.contiguous();
    auto corpus_c = corpus.contiguous();
    auto bitmask_c = bitmask.contiguous();

    torch::Tensor mask_u64;
    int expected_words = (N + 63) / 64;
    if (bitmask_c.dtype() == torch::kBool) {
        auto padded = bitmask_c.to(torch::kInt64);
        int pad_len = expected_words * 64 - bitmask_c.numel();
        if (pad_len > 0) {
            padded = torch::nn::functional::pad(padded, torch::nn::functional::PadFuncOptions({0, pad_len}).value(0));
        }
        auto reshaped = padded.view({expected_words, 64});
        auto powers_list = std::vector<int64_t>(64);
        for (int i = 0; i < 64; ++i) {
            powers_list[i] = (i < 63) ? (1LL << i) : (int64_t)0x8000000000000000ULL;
        }
        auto powers = torch::tensor(powers_list, torch::TensorOptions().dtype(torch::kInt64).device(queries.device()));
        mask_u64 = torch::sum(reshaped * powers, /*dim=*/1);
    } else {
        TORCH_CHECK(bitmask_c.dtype() == torch::kInt64, "bitmask must be int64 (packed uint64) or bool");
        mask_u64 = bitmask_c;
    }

    torch::Tensor ids_i32;
    const int* orig_ids_ptr = nullptr;
    if (original_ids.has_value() && original_ids->defined()) {
        TORCH_CHECK(original_ids->is_cuda(), "original_ids must be a CUDA tensor");
        TORCH_CHECK(original_ids->device() == queries.device(), "original_ids must reside on the same CUDA device");
        if (original_ids->dtype() != torch::kInt32) {
            ids_i32 = original_ids->to(torch::kInt32).contiguous();
        } else {
            ids_i32 = original_ids->contiguous();
        }
        orig_ids_ptr = ids_i32.data_ptr<int>();
    }

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(queries.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(queries.device());
    auto stream = c10::cuda::getCurrentCUDAStream(queries.device().index()).stream();

    if (actual_k <= 256) {
        auto scores_ws = torch::empty({B, N}, options_float);
        auto out_scores = torch::empty({B, actual_k}, options_float);
        auto out_ids = torch::empty({B, actual_k}, options_int);

        circuit::launch_batched_search_filtered(
            queries_c.data_ptr<float>(),
            corpus_c.data_ptr<float>(),
            reinterpret_cast<const uint64_t*>(mask_u64.data_ptr<int64_t>()),
            orig_ids_ptr,
            B, N, dim, actual_k,
            scores_ws.data_ptr<float>(),
            out_scores.data_ptr<float>(),
            out_ids.data_ptr<int>(),
            stream,
            /*stride_k=*/actual_k,
            /*num_words=*/static_cast<int>(mask_u64.numel())
        );

        return {out_scores, out_ids};
    } else {
        auto scores_ws = torch::empty({B, N}, options_float);
        circuit::launch_batched_dense_score(
            queries_c.data_ptr<float>(),
            corpus_c.data_ptr<float>(),
            B, N, dim,
            scores_ws.data_ptr<float>(),
            stream
        );

        int num_words = mask_u64.numel();
        circuit::launch_apply_bitmask_to_scores(
            scores_ws.data_ptr<float>(),
            reinterpret_cast<const uint64_t*>(mask_u64.data_ptr<int64_t>()),
            orig_ids_ptr,
            B, N, num_words,
            stream
        );

        auto topk_res = torch::topk(scores_ws, actual_k, /*dim=*/1);
        auto out_scores = std::get<0>(topk_res);
        auto topk_indices = std::get<1>(topk_res);

        torch::Tensor out_ids;
        if (ids_i32.defined()) {
            out_ids = ids_i32.index_select(0, topk_indices.view(-1)).view({B, actual_k});
        } else {
            out_ids = topk_indices.to(torch::kInt32);
        }

        auto invalid = out_scores <= -1e8f;
        out_ids.masked_fill_(invalid, -1);

        return {out_scores, out_ids};
    }
}

torch::Tensor combine_bitmasks_cuda(
    torch::Tensor mask_a,
    torch::Tensor mask_b,
    std::string op = "and"
) {
    TORCH_CHECK(mask_a.is_cuda() && mask_b.is_cuda(), "mask_a and mask_b must be CUDA tensors");
    TORCH_CHECK(mask_a.device() == mask_b.device(), "masks must reside on the same CUDA device");
    TORCH_CHECK(mask_a.dtype() == torch::kInt64 && mask_b.dtype() == torch::kInt64, "masks must be int64 tensors");
    TORCH_CHECK(mask_a.numel() == mask_b.numel(), "mask_a and mask_b must have the same number of words");

    int num_words = mask_a.numel();
    auto mask_a_c = mask_a.contiguous();
    auto mask_b_c = mask_b.contiguous();
    auto mask_out = torch::empty_like(mask_a_c);

    int op_code = 0;
    std::string op_lower = op;
    std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
    if (op_lower == "and") {
        op_code = 0;
    } else if (op_lower == "or") {
        op_code = 1;
    } else if (op_lower == "and_not" || op_lower == "andnot" || op_lower == "not") {
        op_code = 2;
    } else if (op_lower == "xor") {
        op_code = 3;
    } else {
        TORCH_CHECK(false, "Unknown combine operation: " + op + ". Supported: 'and', 'or', 'and_not', 'xor'");
    }

    auto stream = c10::cuda::getCurrentCUDAStream(mask_a.device().index()).stream();
    circuit::launch_combine_bitmasks(
        reinterpret_cast<const uint64_t*>(mask_a_c.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(mask_b_c.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(mask_out.data_ptr<int64_t>()),
        num_words,
        op_code,
        stream
    );
    return mask_out;
}

torch::Tensor evaluate_compound_bitmask_cuda(
    std::vector<torch::Tensor> leaf_masks,
    py::object bytecode_obj,
    int num_words = -1,
    int N = 0
) {
    circuit::BytecodeStream bc;
    if (py::isinstance<torch::Tensor>(bytecode_obj)) {
        auto bytecode = bytecode_obj.cast<torch::Tensor>();
        TORCH_CHECK(bytecode.numel() <= circuit::MAX_BYTECODE_BYTES, "Bytecode length exceeds MAX_BYTECODE_BYTES (128)");
        bc.length = bytecode.numel();
        auto bc_cpu = bytecode.to(torch::kCPU).to(torch::kUInt8).contiguous();
        std::memcpy(bc.code, bc_cpu.data_ptr<uint8_t>(), bc.length);
    } else if (py::isinstance<py::bytes>(bytecode_obj) || py::isinstance<py::bytearray>(bytecode_obj)) {
        std::string s = bytecode_obj.cast<std::string>();
        TORCH_CHECK(s.size() <= circuit::MAX_BYTECODE_BYTES, "Bytecode length exceeds MAX_BYTECODE_BYTES (128)");
        bc.length = s.size();
        std::memcpy(bc.code, s.data(), bc.length);
    } else {
        TORCH_CHECK(false, "bytecode must be a torch.Tensor, bytes, or bytearray");
    }

    circuit::LeafPointers lp;
    lp.count = leaf_masks.size();
    TORCH_CHECK(lp.count <= circuit::MAX_COMPOUND_LEAVES, "number of leaf masks exceeds MAX_COMPOUND_LEAVES (32)");

    torch::Device device = torch::kCUDA;
    if (lp.count > 0) {
        device = leaf_masks[0].device();
        TORCH_CHECK(device.is_cuda(), "leaf masks must reside on a CUDA device");
        if (num_words <= 0) {
            num_words = leaf_masks[0].numel();
        }
    } else {
        TORCH_CHECK(num_words > 0, "num_words must be positive if no leaf masks are provided");
    }

    for (int i = 0; i < lp.count; ++i) {
        TORCH_CHECK(leaf_masks[i].is_cuda(), "All leaf masks must be CUDA tensors");
        TORCH_CHECK(leaf_masks[i].device() == device, "All leaf masks must be on the same CUDA device");
        TORCH_CHECK(leaf_masks[i].dtype() == torch::kInt64, "Leaf masks must be int64 (packed uint64)");
        TORCH_CHECK(leaf_masks[i].numel() >= num_words, "Leaf mask size smaller than num_words");
        auto leaf_c = leaf_masks[i].contiguous();
        lp.ptrs[i] = reinterpret_cast<const uint64_t*>(leaf_c.data_ptr<int64_t>());
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto mask_out = torch::empty({num_words}, options);

    auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
    circuit::launch_evaluate_compound_bitmask(
        lp,
        bc,
        reinterpret_cast<uint64_t*>(mask_out.data_ptr<int64_t>()),
        num_words,
        N,
        stream
    );

    return mask_out;
}

torch::Tensor scan_numeric_range_cuda(
    torch::Tensor column,
    float threshold,
    py::object op_obj,
    int num_words = -1
) {
    TORCH_CHECK(column.is_cuda(), "column must be a CUDA tensor");
    TORCH_CHECK(column.dim() == 1, "column must be a 1D tensor [N]");

    int op_code = 0;
    if (py::isinstance<py::str>(op_obj)) {
        std::string op_str = op_obj.cast<std::string>();
        std::transform(op_str.begin(), op_str.end(), op_str.begin(), ::tolower);
        if (op_str == "$eq" || op_str == "eq" || op_str == "==") {
            op_code = 0;
        } else if (op_str == "$ne" || op_str == "ne" || op_str == "!=") {
            op_code = 1;
        } else if (op_str == "$gt" || op_str == "gt" || op_str == ">") {
            op_code = 2;
        } else if (op_str == "$gte" || op_str == "gte" || op_str == ">=") {
            op_code = 3;
        } else if (op_str == "$lt" || op_str == "lt" || op_str == "<") {
            op_code = 4;
        } else if (op_str == "$lte" || op_str == "lte" || op_str == "<=") {
            op_code = 5;
        } else {
            TORCH_CHECK(false, "Unknown operator string for scan_numeric_range: ", op_str);
        }
    } else if (py::isinstance<py::int_>(op_obj)) {
        op_code = op_obj.cast<int>();
        TORCH_CHECK(op_code >= 0 && op_code <= 5, "Operator code must be between 0 and 5");
    } else {
        TORCH_CHECK(false, "Operator must be a string or integer");
    }

    auto col_f32 = (column.dtype() == torch::kFloat32) ? column.contiguous() : column.to(torch::kFloat32).contiguous();
    int N = col_f32.numel();
    int actual_words = (num_words > 0) ? num_words : (N + 63) / 64;

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(column.device());
    auto mask_out = torch::empty({actual_words}, options);

    auto stream = c10::cuda::getCurrentCUDAStream(column.device().index()).stream();
    circuit::launch_scan_numeric_range(
        col_f32.data_ptr<float>(),
        threshold,
        op_code,
        reinterpret_cast<uint64_t*>(mask_out.data_ptr<int64_t>()),
        N,
        actual_words,
        stream
    );

    return mask_out;
}

void set_bitmask_bit_cuda(
    torch::Tensor bitmask,
    int doc_id,
    bool valid = true
) {
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be a CUDA tensor");
    TORCH_CHECK(bitmask.dtype() == torch::kInt64, "bitmask must be int64 (packed uint64)");
    auto stream = c10::cuda::getCurrentCUDAStream(bitmask.device().index()).stream();
    circuit::launch_set_bitmask_bit(
        reinterpret_cast<uint64_t*>(bitmask.data_ptr<int64_t>()),
        doc_id,
        valid,
        bitmask.numel(),
        stream
    );
}

void set_bitmask_bits_cuda(
    torch::Tensor bitmask,
    torch::Tensor doc_ids,
    bool valid = true
) {
    TORCH_CHECK(bitmask.is_cuda() && doc_ids.is_cuda(), "bitmask and doc_ids must be CUDA tensors");
    TORCH_CHECK(bitmask.dtype() == torch::kInt64, "bitmask must be int64");
    auto ids_i32 = (doc_ids.dtype() == torch::kInt32) ? doc_ids.contiguous() : doc_ids.to(torch::kInt32).contiguous();
    auto stream = c10::cuda::getCurrentCUDAStream(bitmask.device().index()).stream();
    circuit::launch_set_bitmask_bits(
        reinterpret_cast<uint64_t*>(bitmask.data_ptr<int64_t>()),
        ids_i32.data_ptr<int>(),
        ids_i32.numel(),
        valid,
        bitmask.numel(),
        stream
    );
}


torch::Tensor maxsim_token_score_cuda(
    torch::Tensor query_tokens,
    torch::Tensor corpus_tokens,
    std::optional<torch::Tensor> doc_lens,
    std::optional<torch::Tensor> bitmask
) {
    TORCH_CHECK(query_tokens.is_cuda() && corpus_tokens.is_cuda(), "query_tokens and corpus_tokens must be CUDA tensors");
    TORCH_CHECK(query_tokens.device() == corpus_tokens.device(), "tensors must reside on the same CUDA device");
    TORCH_CHECK(query_tokens.dim() == 2, "query_tokens must be 2D [L_q, D]");
    TORCH_CHECK(corpus_tokens.dim() == 3, "corpus_tokens must be 3D [N, L_d, D]");
    TORCH_CHECK(query_tokens.size(1) == corpus_tokens.size(2), "token dimensions must match");

    int L_q = query_tokens.size(0);
    int N = corpus_tokens.size(0);
    int L_d = corpus_tokens.size(1);
    int dim = query_tokens.size(1);

    auto q_c = query_tokens.contiguous();
    auto corpus_c = corpus_tokens.contiguous();

    const int* doc_lens_ptr = nullptr;
    torch::Tensor doc_lens_i32;
    if (doc_lens.has_value() && doc_lens->defined()) {
        TORCH_CHECK(doc_lens->is_cuda(), "doc_lens must be on CUDA");
        doc_lens_i32 = doc_lens->to(torch::kInt32).contiguous();
        doc_lens_ptr = doc_lens_i32.data_ptr<int>();
    }

    const uint64_t* mask_ptr = nullptr;
    torch::Tensor mask_u64;
    if (bitmask.has_value() && bitmask->defined()) {
        TORCH_CHECK(bitmask->is_cuda(), "bitmask must be on CUDA");
        auto bitmask_c = bitmask->contiguous();
        int expected_words = (N + 63) / 64;
        if (bitmask_c.dtype() == torch::kBool) {
            auto padded = bitmask_c.to(torch::kInt64);
            int pad_len = expected_words * 64 - bitmask_c.numel();
            if (pad_len > 0) {
                padded = torch::nn::functional::pad(padded, torch::nn::functional::PadFuncOptions({0, pad_len}).value(0));
            }
            auto reshaped = padded.view({expected_words, 64});
            auto powers_list = std::vector<int64_t>(64);
            for (int i = 0; i < 64; ++i) {
                powers_list[i] = (i < 63) ? (1LL << i) : (int64_t)0x8000000000000000ULL;
            }
            auto powers = torch::tensor(powers_list, torch::TensorOptions().dtype(torch::kInt64).device(query_tokens.device()));
            mask_u64 = torch::sum(reshaped * powers, /*dim=*/1);
        } else {
            mask_u64 = bitmask_c.to(torch::kInt64);
        }
        mask_ptr = reinterpret_cast<const uint64_t*>(mask_u64.data_ptr<int64_t>());
    }

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(query_tokens.device());
    auto scores = torch::empty({N}, options_float);

    auto stream = c10::cuda::getCurrentCUDAStream(query_tokens.device().index()).stream();
    circuit::launch_maxsim_token_score(
        q_c.data_ptr<float>(),
        corpus_c.data_ptr<float>(),
        doc_lens_ptr,
        mask_ptr,
        N, L_q, L_d, dim,
        scores.data_ptr<float>(),
        stream
    );
    return scores;
}

std::pair<torch::Tensor, torch::Tensor> maxsim_token_search_cuda(
    torch::Tensor query_tokens,
    torch::Tensor corpus_tokens,
    std::optional<torch::Tensor> doc_lens,
    std::optional<torch::Tensor> bitmask,
    int K = 10
) {
    int N = corpus_tokens.size(0);
    int actual_k = std::min(K, N);

    auto scores = maxsim_token_score_cuda(query_tokens, corpus_tokens, doc_lens, bitmask);
    auto topk_res = torch::topk(scores, actual_k, /*dim=*/0);
    auto out_scores = std::get<0>(topk_res);
    auto out_ids = std::get<1>(topk_res).to(torch::kInt32);
    out_ids[out_scores <= -1e8] = -1;
    return {out_scores, out_ids};
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
    m.def("batched_search", &batched_search_cuda, "Batched vector search (CUDA)",
          py::arg("queries"), py::arg("corpus"), py::arg("original_ids") = py::none(), py::arg("k") = 10);
    m.def("batched_dense_score", &batched_dense_score_cuda, "Batched dense scoring (cuBLAS)",
          py::arg("queries"), py::arg("corpus"));
    m.def("batched_topk", &batched_topk_cuda, "Batched parallel Top-K reduction (CUDA)",
          py::arg("scores"), py::arg("original_ids") = py::none(), py::arg("k") = 10);
    m.def("flat_score_filtered", &flat_score_filtered_cuda, "Flat scoring with 64-bit bitmask filtering (CUDA)",
          py::arg("query"), py::arg("corpus"), py::arg("bitmask"));
    m.def("batched_search_filtered", &batched_search_filtered_cuda, "Batched vector search with bitmask filtering (CUDA)",
          py::arg("queries"), py::arg("corpus"), py::arg("bitmask"), py::arg("original_ids") = py::none(), py::arg("k") = 10);
    m.def("combine_bitmasks", &combine_bitmasks_cuda, "Combine two 64-bit packed bitmasks (CUDA)",
          py::arg("mask_a"), py::arg("mask_b"), py::arg("op") = "and");
    m.def("evaluate_compound_bitmask", &evaluate_compound_bitmask_cuda,
          "In-register CUDA compound bitmask RPN bytecode evaluator",
          py::arg("leaf_masks"), py::arg("bytecode"), py::arg("num_words") = -1, py::arg("N") = 0);
    m.def("scan_numeric_range", &scan_numeric_range_cuda,
          "Warp __ballot_sync scalar comparison bitmask scanner",
          py::arg("column"), py::arg("threshold"), py::arg("op"), py::arg("num_words") = -1);
    m.def("set_bitmask_bit", &set_bitmask_bit_cuda,
          "Atomic CUDA single bit mutation on packed bitmask",
          py::arg("bitmask"), py::arg("doc_id"), py::arg("valid") = true);
    m.def("set_bitmask_bits", &set_bitmask_bits_cuda,
          "Atomic CUDA batched bit mutation on packed bitmask",
          py::arg("bitmask"), py::arg("doc_ids"), py::arg("valid") = true);
    m.def("maxsim_token_score", &maxsim_token_score_cuda, "Late-interaction token MaxSim scoring (CUDA)",
          py::arg("query_tokens"), py::arg("corpus_tokens"), py::arg("doc_lens") = py::none(), py::arg("bitmask") = py::none());
    m.def("maxsim_token_search", &maxsim_token_search_cuda, "Late-interaction token MaxSim Top-K retrieval (CUDA)",
          py::arg("query_tokens"), py::arg("corpus_tokens"), py::arg("doc_lens") = py::none(), py::arg("bitmask") = py::none(), py::arg("k") = 10);
}
