from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import numpy as np

from .ingest import Chunk


class IndexStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self._lock = Lock()
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._meta: dict[str, str] = {}

    def is_ready(self) -> bool:
        return bool(self._chunks) and self._embeddings is not None

    def load(self) -> None:
        with self._lock:
            chunks_path = self.index_dir / "chunks.json"
            embeddings_path = self.index_dir / "embeddings.npy"
            meta_path = self.index_dir / "index_meta.json"

            if not chunks_path.exists() or not embeddings_path.exists():
                self._chunks = []
                self._embeddings = None
                self._meta = {}
                return

            raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**item) for item in raw_chunks]
            self._embeddings = np.load(embeddings_path, allow_pickle=False)
            self._meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    def save(self, chunks: list[Chunk], embeddings: np.ndarray, meta: dict[str, str]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.index_dir / "chunks.json"
        embeddings_path = self.index_dir / "embeddings.npy"
        meta_path = self.index_dir / "index_meta.json"

        with self._lock:
            for path in (chunks_path, embeddings_path, meta_path):
                if path.exists():
                    path.unlink()
            chunks_path.write_text(
                json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            np.save(embeddings_path, embeddings)
            meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
            self._chunks = list(chunks)
            self._embeddings = embeddings
            self._meta = dict(meta)

    def get_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get_embeddings(self) -> np.ndarray | None:
        return self._embeddings

    def get_meta(self) -> dict[str, str]:
        return dict(self._meta)
