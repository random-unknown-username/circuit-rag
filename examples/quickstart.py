# quick sanity run: 200k fake vecs, circuit vs brute force
# python examples/quickstart.py
import time

import numpy as np
import torch

from circuit_rag import build_circuit_index, circuit_search

N, DIM, QUERIES, K = 200_000, 128, 50, 10

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (cpu fallback)'}")

# clustered unit vecs, stand in for real embeddings
rng = np.random.default_rng(0)
centers = rng.normal(size=(256, DIM))
X = centers[rng.integers(0, 256, N)] + 0.3 * rng.normal(size=(N, DIM))
X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)

t = time.time()
index = build_circuit_index(X, num_roots=64, children_per_root=16, rank=8)
print(f"built index over {N:,} vectors in {time.time() - t:.1f}s")

corpus = torch.from_numpy(X).to(index.device)
qids = rng.integers(0, N, QUERIES)

def timed(fn):
    fn()  # warmup
    if corpus.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = [fn(q) for q in qids]
    if corpus.is_cuda:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t) / QUERIES * 1e6

def circuit(q=None):
    return circuit_search(corpus[qids[0] if q is None else q], index, k=K)

def brute(q=None):
    return torch.topk(corpus @ corpus[qids[0] if q is None else q], K)

res, circuit_us = timed(circuit)
truth, brute_us = timed(brute)

matches = sum(set(r[1].tolist()) == set(t.indices.tolist()) for r, t in zip(res, truth))
print(f"exact matches vs brute force: {matches}/{QUERIES}")
print(f"circuit: {circuit_us:.0f} us/query   brute-force GEMV: {brute_us:.0f} us/query")
assert matches == QUERIES, "results differ from brute force!"
