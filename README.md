# circuit-rag

exact top-k vector search on the gpu. clusters get skipped only when a geometric bound proves none of their vectors can beat the current kth best score, so the results are identical to brute force, not approximate.

## fast try (reviewers start here)

```bash
pip install -e . --no-build-isolation   # 1. build (~1 min, needs nvcc + torch w cuda)
python tests/test_exact.py              # 2. ~5s, prints exact: 25/25 queries match brute force
python examples/real_proof.py           # 3. ~1 min, real scifact text, downloads its own data
```

pass = step 2 prints `exact: 25/25 ...` and step 3 prints `300/300`.

## try it (details)

needs an nvidia gpu, cuda toolkit (`nvcc`) and pytorch w cuda.

```bash
git clone <this repo> && cd circuit-rag
pip install -e . --no-build-isolation   # compiles the cuda ext, ~1 min
python tests/test_exact.py              # circuit == brute force
python examples/real_proof.py           # real scifact text, downloads its own data
```

`real_proof.py` pulls scifact from the public beir server, embeds it w `BAAI/bge-small-en-v1.5`, runs 300 labelled queries and checks every result against brute force. nothing precomputed. on my rtx 5050 laptop:

```
exact top-10 match vs brute force: 300/300
leaf groups pruned by bounds (avg): 0%
recall@10 vs SciFact labels: 0.840
```

## the 0% thing

yes thats real. on real text embeddings (384 dims) the bounds cant rule out any clusters, so circuit ends up checking everything. results are still exact, but theres no speedup from pruning there. i tried more roots / higher rank, best case skipped ~7%.

on clustered data it works: `tests/test_exact.py` (fake clustered vecs) skips ~97% of the child clusters. so the math holds, real embeddings are just too spread out in high dims for a rank-8 bound to separate them. thats whats next.

## speed (honest)

- native c++ bench (`circuit_bench`, scifact index, rtx 5050 laptop): ~38us p50 per query
- python path is about the same as a plain cublas gemv (~370us vs ~330us on 200k vecs), per-query python overhead + 1 host sync for the seed threshold
- under 10k vecs it just does a flat gemv

so ship #1 is about being exact and provably so, not about being faster yet.

## whats in here

- 5 cuda kernels in `native/kernels/`: root bounds, parent projection, child bounds, leaf scoring, topk
- torch extension `circuit_cuda` (`native/bindings/torch_extension.cpp`)
- python: `build_circuit_index`, `circuit_search`, `CircuitRAG` (text in, text out), `.circuit` file save/load w mmap
- `native/bin/circuit_bench`: standalone benchmark (`cmake -S native -B build && cmake --build build`)

## usage

```python
from circuit_rag import CircuitRAG

rag = CircuitRAG()   # bge-small-en-v1.5
rag.add_documents(["mRNA vaccines make cells build an antigen protein.", "CRISPR edits dna."])
for r in rag.search("how do mrna vaccines work?", k=2):
    print(r.score, r.text)
```

or the raw index:

```python
from circuit_rag import build_circuit_index, circuit_search

index = build_circuit_index(X, num_roots=64, children_per_root=16, rank=8)  # X: float32 [N, D], unit norm
scores, ids, trace = circuit_search(query, index, k=10)                     # query: torch tensor [D]
index.save("my.circuit")
```

scores are inner products so normalise ur embeddings if u want cosine.

## known limits

- `rank` has to be <= 32 (higher crashes the kernel launch, havent found why yet)
- `add_documents` rebuilds the whole index every call
- no deletes, filters, reranking or hybrid search yet
