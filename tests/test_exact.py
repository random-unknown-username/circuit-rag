# checks circuit topk == brute force topk on clustered fake data
# (this one actually prunes, real text embeddings barely do, see real_proof.py)
import numpy as np
import torch

from circuit_rag import build_circuit_index, circuit_search


def _clustered(n=40000, dim=128, clusters=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    X = centers[rng.integers(0, clusters, n)] + 0.3 * rng.normal(size=(n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def test_matches_brute_force():
    X = _clustered()
    idx = build_circuit_index(X, num_roots=32, children_per_root=8, rank=8)
    corpus = torch.from_numpy(X).to(idx.device)
    k = 10
    rng = np.random.default_rng(1)
    for qi in rng.integers(0, len(X), 25):
        q = corpus[qi]
        scores, ids, _ = circuit_search(q, idx, k=k)
        truth = torch.topk(corpus @ q, k)
        assert set(ids.tolist()) == set(truth.indices.tolist())
        assert torch.allclose(scores.sort(descending=True).values, truth.values, atol=1e-4)


if __name__ == "__main__":
    test_matches_brute_force()
    print("exact: 25/25 queries match brute force")
