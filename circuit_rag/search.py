from typing import Tuple
import torch

from circuit_rag.hierarchy.build import CircuitIndex

try:
    import circuit_cuda
except ImportError:  # no ext built / cpu only
    circuit_cuda = None


class SearchTrace:
    def __init__(self):
        self.corpus_size = 0
        self.seed_threshold = -1e9
        self.path = ""
        self.total_gpu_time_us = 0.0
        self.surviving_children = -1
        self.total_children = 0


# cheap lower bound on the final kth score
# exactly score every vec under the root closest to the query, take its kth best
# the real kth best over the whole corpus can only be >= this, so pruning anything
# whose upper bound is below it can never drop a true topk hit
def _seed_threshold(query: torch.Tensor, index: CircuitIndex, k: int) -> float:
    best_root = int(torch.argmax(index.root_centers @ query).item())
    cpr = index.num_total_children // index.num_roots
    kids = slice(best_root * cpr, (best_root + 1) * cpr)
    sizes = index.leaf_sizes[kids]
    offs = index.leaf_offsets[kids]
    nonempty = sizes > 0
    if not bool(nonempty.any()):
        return -1e9
    # a root's leaves are contiguous in leaf_vectors so one slice gets them all
    start = int(offs[nonempty].min().item())
    end = int((offs[nonempty] + sizes[nonempty]).max().item())
    if end - start < k:
        return -1e9
    scores = index.leaf_vectors[start:end] @ query
    return float(torch.topk(scores, k).values[-1].item()) - 1e-4


# exact topk inner product search, returns (scores, original_ids, trace)
def circuit_search(
    query: torch.Tensor, index: CircuitIndex, k: int = 10, min_hierarchy_n: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor, SearchTrace]:
    if query.ndim != 1 or query.numel() != index.dim:
        raise ValueError("query must have shape [index.dim]")
    query = query.float().contiguous().to(index.leaf_vectors.device)
    k = min(k, index.N)
    trace = SearchTrace()
    trace.corpus_size = index.N

    on_gpu = query.is_cuda and circuit_cuda is not None
    # smol corpus, one gemv is already faster than the hierarchy
    if not on_gpu or index.N < min_hierarchy_n:
        trace.path = "flat"
        scores = index.leaf_vectors @ query
        top_scores, top_idx = torch.topk(scores, k)
        return top_scores, index.original_ids[top_idx], trace

    trace.path = "hierarchical"
    trace.seed_threshold = _seed_threshold(query, index, k)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    circuit_cuda.hierarchical_search(
        query, index.root_centers, index.root_radii, index.bases,
        index.child_intervals_l, index.child_intervals_u, index.child_radii,
        index.child_parent_ids, index.leaf_vectors, index.leaf_offsets, index.leaf_sizes,
        index.ws_survivor_mask, index.ws_survivor_indices, index.ws_num_survivors,
        index.ws_root_dot_products, index.ws_projected_queries, index.ws_perp_norms,
        index.ws_child_survivor_mask, index.ws_child_survivor_indices, index.ws_num_child_survivors,
        index.ws_vector_scores, trace.seed_threshold, index.rank,
        float(torch.linalg.vector_norm(query).item()),
    )
    top_scores, top_idx = torch.topk(index.ws_vector_scores, k)
    end.record()
    end.synchronize()
    # how many child clusters the bounds couldnt rule out
    trace.surviving_children = int(index.ws_num_child_survivors.item())
    trace.total_children = index.num_total_children
    trace.total_gpu_time_us = start.elapsed_time(end) * 1000.0
    return top_scores, index.original_ids[top_idx], trace
