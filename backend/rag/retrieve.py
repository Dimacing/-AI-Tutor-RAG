from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import EmbeddingModel
from .store import IndexStore
from .ingest import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def retrieve(
    question: str,
    embedder: EmbeddingModel,
    store: IndexStore,
    top_k: int,
    min_score: float,
) -> list[RetrievedChunk]:
    embeddings = store.get_embeddings()
    chunks = store.get_chunks()
    if embeddings is None or not chunks:
        return []

    query_vec = embedder.encode([question])
    if query_vec.size == 0:
        return []

    query = query_vec[0]
    scores = embeddings @ query
    ranked = np.argsort(scores)[::-1]

    results: list[RetrievedChunk] = []
    for idx in ranked[: max(top_k, 1)]:
        score = float(scores[idx])
        if score < min_score:
            continue
        results.append(RetrievedChunk(chunk=chunks[idx], score=score))

    return results
