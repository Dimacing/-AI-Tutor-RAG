from __future__ import annotations

from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def encode_with_progress(
        self,
        texts: list[str],
        batch_size: int = 32,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        total = len(texts)
        batches: list[np.ndarray] = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]
            embeddings = self.model.encode(batch, normalize_embeddings=True)
            batches.append(np.asarray(embeddings, dtype="float32"))
            if progress_cb:
                progress_cb(end, total)
        if not batches:
            return np.zeros((0, 1), dtype="float32")
        return np.vstack(batches)
