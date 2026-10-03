from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from circuit_rag.hierarchy.build import build_circuit_index
from circuit_rag.search import circuit_search


@dataclass
class SearchResult:
    id: int
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# text in, exact topk text out
# embeds w sentence-transformers then searches w circuit
class CircuitRAG:
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._vecs: Optional[np.ndarray] = None
        self.index = None

    # embed + add docs then rebuild the whole index, returns the new doc ids
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        metadatas = metadatas or [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("metadatas must be the same length as texts")
        vecs = self.embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        first = len(self.texts)
        self.texts += texts
        self.metadatas += metadatas
        self._vecs = vecs if self._vecs is None else np.vstack([self._vecs, vecs])
        self._rebuild()
        return list(range(first, first + len(texts)))

    def _rebuild(self):
        n = len(self.texts)
        # hierarchy needs enough vecs per leaf, tiny corpus just gets 1 root 1 child
        roots = int(max(1, min(64, n // 256)))
        children = int(max(1, min(16, n // (roots * 16))))
        rank = min(8, self._vecs.shape[1])
        self.index = build_circuit_index(
            self._vecs, num_roots=roots, children_per_root=children, rank=rank, device=self.device
        )

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self.index is None:
            return []
        q = self.embedder.encode(query, normalize_embeddings=True, convert_to_tensor=True).float()
        scores, ids, _ = circuit_search(q.to(self.index.device), self.index, k=k)
        return [
            SearchResult(int(i), float(s), self.texts[int(i)], self.metadatas[int(i)])
            for s, i in zip(scores.tolist(), ids.tolist())
        ]
