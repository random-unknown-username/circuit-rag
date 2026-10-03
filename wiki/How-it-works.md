# how it works

## the idea

brute force scores every vector. circuit groups vectors into clusters and tries to prove a whole cluster cant hold a top-k result, so it never has to score it. the proof is just an upper bound on the best score any vector in the cluster could get, if that bound is below the current kth best score the cluster is skipped. skipping on a real bound means the final answer is the same as brute force.

## the hierarchy

built once by `build_circuit_index`:

- **roots**: kmeans clusters. each stores a center `c` and a radius `r` (max distance of any member from `c`)
- **parent basis**: a rank-`r` pca basis `V` of the vectors under each root
- **children**: each root is split again. each child stores a box `[l, u]` in the `V` subspace, and a residual radius `rho` (how far members stray from that subspace)
- **leaves**: the actual vectors, laid out contiguous per child so a child is one slice of memory

## the 5 kernels (`native/kernels/`)

per query, in order:

1. **root_bounds**: for each root, upper bound = `q.c + |q| * r`. roots whose bound is under the threshold die
2. **parent_projection**: for surviving roots, project the query onto `V` (gives `qP`) and compute how much of the query is left outside the subspace (`perp_norm`, via pythagoras: `sqrt(|q|^2 - |qP|^2)`)
3. **child_bounds**: for each child of a surviving root, bound = `q.c + max over the box of qP.z + perp_norm * rho`. the box max is cheap since its per coordinate (pick `l` or `u` by sign of `qP`). children under the threshold die
4. **dense_score**: score every vector in surviving children (float4 loads, query cached in shared mem)
5. **topk**: two stage parallel top-k over the scores

all of this is launched on one stream with no cpu sync in between (`launch_hierarchical_search` in `native/src/dispatch.cu`), survivors are tracked via masks and device-side counters instead of copying counts to the host.

## why its exact

pruning needs a threshold: a score the final kth best is guaranteed to be at least. `circuit_search` gets it in `_seed_threshold` (`circuit_rag/search.py`):

1. find the root whose center is closest to the query
2. exactly score every vector under that root
3. take the kth best of those

the true kth best over the whole corpus can only be >= that, because those vectors are a subset of it. so anything whose upper bound is below the seed can never be in the true top-k. (theres a tiny `-1e-4` slack for float error.)

and the bounds themselves are real upper bounds (cauchy-schwarz on the radius terms), not heuristics.

## when it actually skips stuff

only when the bounds are tight enough. clustered low-dim-ish data: ~97% of children skipped. real 384-dim text embeddings: ~0%. see [limits + faq](Limits-and-FAQ.md).

## where to read the code

| thing | file |
|---|---|
| kernels | `native/kernels/*.cu` |
| launch order | `native/src/dispatch.cu` |
| torch bindings | `native/bindings/torch_extension.cpp` |
| index build | `circuit_rag/hierarchy/build.py` |
| search + seed threshold | `circuit_rag/search.py` |
| save/load | `circuit_rag/storage/format.py` |
