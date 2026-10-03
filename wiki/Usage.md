# usage

two ways in: `CircuitRAG` (text in, text out) or the raw index (u bring your own embeddings).

## CircuitRAG

```python
from circuit_rag import CircuitRAG

rag = CircuitRAG()   # BAAI/bge-small-en-v1.5, cuda if available
ids = rag.add_documents(
    ["mRNA vaccines make cells build an antigen protein.", "CRISPR edits dna."],
    metadatas=[{"topic": "medicine"}, {"topic": "genetics"}],   # optional
)

for r in rag.search("how do mrna vaccines work?", k=2):
    print(r.id, r.score, r.text, r.metadata)
```

- `CircuitRAG(embedding_model="BAAI/bge-small-en-v1.5", device=None)` any sentence-transformers model works
- `add_documents(texts, metadatas=None)` returns the new doc ids (0, 1, 2... in add order). **it rebuilds the whole index every call**, so add in big batches
- `search(query, k=5)` returns a list of `SearchResult(id, score, text, metadata)`. score is cosine since embeddings are normalised
- no deletes yet

## raw index

```python
import torch
from circuit_rag import build_circuit_index, circuit_search

# X: float32 numpy [N, D], rows unit-normalised
index = build_circuit_index(X, num_roots=64, children_per_root=16, rank=8)

q = torch.from_numpy(X[0]).cuda()                  # [D] tensor, same device as index
scores, ids, trace = circuit_search(q, index, k=10)
```

`build_circuit_index(X, num_roots=128, children_per_root=16, rank=8, random_state=42, device=None)`

- `num_roots` / `children_per_root`: size of the 2 level hierarchy. needs `N >= num_roots * children_per_root`.
- `rank`: dimensions of the low-rank bound. has to be **<= 32**. higher = tighter bounds but more work per cluster
- uses the gpu builder if cuda is there, sklearn kmeans/pca otherwise

`circuit_search(query, index, k=10, min_hierarchy_n=10000)`

- returns `(scores, original_ids, trace)`. `ids` are row numbers of the `X` u built from
- scores are inner products. normalise first if u want cosine
- under `min_hierarchy_n` vectors it just does one gemv + topk (faster than the hierarchy at that size). set it to `0` to force the hierarchy, `real_proof.py` does this

## the trace

```python
scores, ids, trace = circuit_search(q, index, k=10, min_hierarchy_n=0)
trace.path                 # "flat" or "hierarchical"
trace.seed_threshold       # lower bound on the kth score used for pruning
trace.surviving_children   # child clusters the bounds couldnt rule out
trace.total_children
trace.total_gpu_time_us
```

pruned % is `1 - surviving_children / total_children`. only filled in on the hierarchical path.

## save / load

```python
nbytes = index.save("my.circuit")

from circuit_rag.hierarchy.build import CircuitIndex
index = CircuitIndex.load("my.circuit")                       # mmap by default
index = CircuitIndex.load("my.circuit", device="cuda", use_mmap=False)
```

layout is in [file format](File-format.md).

## examples in the repo

- `tests/test_exact.py` 25 queries on fake clustered data, checks == brute force
- `examples/quickstart.py` 200k fake vectors, circuit vs brute force timing
- `examples/real_proof.py` real scifact text end to end
