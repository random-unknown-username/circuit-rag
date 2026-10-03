# limits + faq

## known limits

- `rank` must be <= 32. higher crashes the kernel launch (cuda invalid argument), cause not found yet. `build_circuit_index` raises a `ValueError` for it
- `CircuitRAG.add_documents` rebuilds the whole index each call
- no deletes, metadata filters, reranking or hybrid/bm25 search in v1
- needs an nvidia gpu + nvcc to install
- float32 only, single gpu
- python path is not faster than a plain gemv yet (see [native bench](Native-bench.md))

## why does real_proof.py say 0% pruned?

the bounds couldnt rule out any cluster on scifact. real text embeddings are 384 dims and in that many dims most vectors are about the same distance from each other, so a rank-8 bound is too loose to separate clusters. i tried more roots and higher rank, best case skipped ~7%.

results are still exact (300/300 match brute force), circuit just ends up scoring everything on that data. on clustered data it skips ~97% (`tests/test_exact.py` data), so the math works, real embeddings just arent clustered enough. making the bounds tighter on real embeddings is the next thing.

## is it approximate?

no. every result is checked against brute force in the tests and in `real_proof.py`. pruning only happens on a proven bound, see [how it works](How-it-works.md).

## why does a small corpus not use the hierarchy?

under 10k vectors `circuit_search` does one gemv + topk because thats faster than launching the hierarchy. pass `min_hierarchy_n=0` to force it.

## does it do cosine or inner product?

inner product. embeddings from `CircuitRAG` are normalised so its cosine there. if u use the raw index normalise your vectors yourself.

## can i use a different embedding model?

yes, `CircuitRAG(embedding_model="...")` takes any sentence-transformers model. for the raw index its just a float32 array.

## whats on the roadmap

- bounds that actually prune on real embeddings
- faster python path (no host sync for the seed threshold)
- deletes + filters
- incremental adds instead of rebuild
