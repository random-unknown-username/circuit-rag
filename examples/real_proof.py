# real data proof on beir scifact (downloads ~3mb, embeds 5183 abstracts, 300 labelled queries)
# python examples/real_proof.py
# checks 3 things w the hierarchy forced on:
#   1. circuit top10 == brute force top10 for every query
#   2. how much of the index the bounds let us skip
#   3. recall@10 vs the human relevance labels
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from circuit_rag import build_circuit_index, circuit_search

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "scifact")
K = 10

if not os.path.exists(os.path.join(CACHE, "scifact", "corpus.jsonl")):
    print("downloading SciFact...")
    os.makedirs(CACHE, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).extractall(CACHE)
root = os.path.join(CACHE, "scifact")

corpus = [json.loads(l) for l in open(os.path.join(root, "corpus.jsonl"))]
queries = {q["_id"]: q["text"] for q in map(json.loads, open(os.path.join(root, "queries.jsonl")))}
qrels = {}
for line in list(open(os.path.join(root, "qrels", "test.tsv")))[1:]:
    qid, did, rel = line.split()
    if int(rel) > 0:
        qrels.setdefault(qid, set()).add(did)
qids = [q for q in queries if q in qrels]
doc_ids = [d["_id"] for d in corpus]
print(f"{len(corpus)} docs, {len(qids)} labelled queries")

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
X = model.encode([d.get("title", "") + ". " + d["text"] for d in corpus],
                 normalize_embeddings=True, batch_size=128, show_progress_bar=True).astype(np.float32)
Q = model.encode([queries[q] for q in qids], normalize_embeddings=True, convert_to_tensor=True).float()

index = build_circuit_index(X, num_roots=16, children_per_root=8, rank=8, device=dev)
corp = torch.from_numpy(X).to(index.device)

exact, hits, skipped = 0, 0, []
for i, qid in enumerate(qids):
    q = Q[i].to(index.device)
    scores, ids, tr = circuit_search(q, index, k=K, min_hierarchy_n=0)  # force hierarchy even tho N < 10k
    truth = torch.topk(corp @ q, K).indices.tolist()
    exact += set(ids.tolist()) == set(truth)
    if tr.surviving_children >= 0:
        skipped.append(1 - tr.surviving_children / tr.total_children)
    got = {doc_ids[j] for j in ids.tolist()}
    hits += len(got & qrels[qid]) / len(qrels[qid])

print(f"\nexact top-{K} match vs brute force: {exact}/{len(qids)}")
if skipped:
    print(f"leaf groups pruned by bounds (avg): {100 * np.mean(skipped):.0f}%")
print(f"recall@{K} vs SciFact labels: {hits / len(qids):.3f}")
assert exact == len(qids), "mismatch vs brute force"
