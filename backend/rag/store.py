from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .ingest import Chunk


class IndexStore:
    _COLLECTION_NAME = "chunks"
    _UPSERT_BATCH_SIZE = 256

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._lock = Lock()
        self._chunks: list[Chunk] = []
        self._meta: dict[str, str] = {}
        self._vector_ready = False
        self._qdrant_path = self.index_dir / "qdrant"
        self._client = QdrantClient(path=str(self._qdrant_path))

    def is_ready(self) -> bool:
        return bool(self._chunks) and self._vector_ready

    def load(self) -> None:
        with self._lock:
            chunks_path = self.index_dir / "chunks.json"
            meta_path = self.index_dir / "index_meta.json"

            if not chunks_path.exists():
                self._chunks = []
                self._meta = {}
                self._vector_ready = False
                return

            raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**item) for item in raw_chunks]
            self._meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            self._vector_ready = self._collection_has_points()
            if not self._vector_ready:
                self._ensure_vectors_from_numpy()

    def save(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict[str, str]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.index_dir / "chunks.json"
        meta_path = self.index_dir / "index_meta.json"
        legacy_embeddings_path = self.index_dir / "embeddings.npy"

        with self._lock:
            for path in (chunks_path, meta_path, legacy_embeddings_path):
                if path.exists():
                    path.unlink()
            chunks_path.write_text(
                json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
            self._chunks = list(chunks)
            self._meta = dict(meta)
            self._write_vectors(embeddings)
            self._vector_ready = True

    def get_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get_meta(self) -> dict[str, str]:
        return dict(self._meta)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if not self._vector_ready:
            return []
        if query_vector.ndim != 1:
            query_vector = query_vector.reshape(-1)
        if query_vector.size == 0:
            return []
        try:
            if hasattr(self._client, "search"):
                hits = self._client.search(
                    collection_name=self._COLLECTION_NAME,
                    query_vector=query_vector.tolist(),
                    limit=max(top_k, 1),
                    with_payload=False,
                )
            elif hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self._COLLECTION_NAME,
                    query=query_vector.tolist(),
                    limit=max(top_k, 1),
                    with_payload=False,
                )
                hits = response.points
            else:
                return []
        except Exception:
            return []

        results: list[tuple[int, float]] = []
        max_index = len(self._chunks) - 1
        for hit in hits:
            try:
                idx = int(hit.id)
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx > max_index:
                continue
            results.append((idx, float(hit.score)))
        return results

    def _collection_exists(self) -> bool:
        try:
            exists = getattr(self._client, "collection_exists", None)
            if callable(exists):
                return bool(exists(self._COLLECTION_NAME))
            self._client.get_collection(self._COLLECTION_NAME)
            return True
        except Exception:
            return False

    def _collection_has_points(self) -> bool:
        if not self._collection_exists():
            return False
        try:
            count = self._client.count(self._COLLECTION_NAME, exact=True).count
        except Exception:
            return False
        return int(count) >= len(self._chunks) and len(self._chunks) > 0

    def _ensure_vectors_from_numpy(self) -> None:
        # One-time migration path from legacy embeddings.npy to Qdrant.
        embeddings_path = self.index_dir / "embeddings.npy"
        if not embeddings_path.exists() or not self._chunks:
            self._vector_ready = False
            return
        try:
            embeddings = np.load(embeddings_path, allow_pickle=False)
        except Exception:
            self._vector_ready = False
            return
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self._chunks):
            self._vector_ready = False
            return
        try:
            self._write_vectors(embeddings)
        except Exception:
            self._vector_ready = False
            return
        self._vector_ready = True

    def _write_vectors(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self._chunks):
            raise ValueError("Embeddings shape does not match chunks.")
        vector_size = int(embeddings.shape[1])
        if vector_size <= 0:
            raise ValueError("Embeddings must be non-empty.")
        self._client.recreate_collection(
            collection_name=self._COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        total = len(self._chunks)
        for start in range(0, total, self._UPSERT_BATCH_SIZE):
            end = min(start + self._UPSERT_BATCH_SIZE, total)
            points: list[PointStruct] = []
            for idx in range(start, end):
                chunk = self._chunks[idx]
                points.append(
                    PointStruct(
                        id=idx,
                        vector=embeddings[idx].tolist(),
                        payload={"chunk_id": chunk.chunk_id},
                    )
                )
            if points:
                self._client.upsert(collection_name=self._COLLECTION_NAME, points=points)
