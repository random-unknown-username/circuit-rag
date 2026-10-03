# circuit-rag

exact top-k vector search on the gpu. clusters get skipped only when a geometric bound proves none of their vectors can beat the current kth best score, so the results are identical to brute force, not approximate.

## fast try (reviewers start here)

needs an nvidia gpu, cuda toolkit (`nvcc`) and pytorch w cuda. no clone needed.

**1. install** (compiles the cuda part on your machine, ~1 min)

```bash
pip install circuit-rag --no-build-isolation
```

**2. save this as `try.py` and run it**

```python
from circuit_rag import CircuitRAG

rag = CircuitRAG()   # downloads bge-small-en-v1.5 the first time
rag.add_documents([
    "mRNA vaccines make your cells build a harmless antigen protein, which trains the immune system.",
    "CRISPR-Cas9 edits dna at a precise spot.",
    "Photosynthesis turns light into chemical energy inside chloroplasts.",
    "General relativity says gravity is curved spacetime.",
    "Mitochondria make atp through oxidative phosphorylation.",
])

for r in rag.search("how do mrna vaccines work?", k=3):
    print(f"{r.score:.3f}  {r.text}")
```

```
0.841  mRNA vaccines make your cells build a harmless antigen protein, which trains the immune system.
0.628  CRISPR-Cas9 edits dna at a precise spot.
0.554  Mitochondria make atp through oxidative phosphorylation.
```

top hit is the mrna doc, thats it working. want your own text? just swap the list.

## prove its exact

the results arent approximate, every result is checked against brute force. to run the checks clone the repo:

```bash
git clone https://github.com/random-unknown-username/circuit-rag && cd circuit-rag
pip install -e . --no-build-isolation
python tests/test_exact.py              # ~5s, prints exact: 25/25 queries match brute force
python examples/real_proof.py           # ~1 min, real scifact text, downloads its own data
```

![output of tests/test_exact.py and examples/real_proof.py on my rtx 5050 laptop](https://raw.githubusercontent.com/random-unknown-username/circuit-rag/main/assets/proof.png)

pass = `exact: 25/25 ...` and `300/300`.

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

## wiki

full docs (install, usage, how it works, file format, faq) are in [wiki/](https://github.com/random-unknown-username/circuit-rag/blob/main/wiki/Home.md).

## known limits

- `rank` has to be <= 32 (higher crashes the kernel launch, havent found why yet)
- `add_documents` rebuilds the whole index every call
- no deletes, filters, reranking or hybrid search yet
