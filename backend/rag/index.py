from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config import get_settings
from .embeddings import EmbeddingModel
from .ingest import build_chunks, expand_sources, load_document, load_sources
from .store import IndexStore


def build_index(
    sources_path: Path,
    chunk_size: int,
    overlap: int,
    allowlist: str,
    max_pages: int,
) -> None:
    settings = get_settings()
    sources = load_sources(sources_path)
    expanded = expand_sources(sources, allowlist, max_pages)

    documents = []
    for item in expanded:
        try:
            doc = load_document(item)
        except Exception:
            doc = None
        if doc:
            documents.append(doc)

    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    embedder = EmbeddingModel(settings.embedding_model)
    embeddings = embedder.encode([chunk.text for chunk in chunks])

    meta = {
        "embedding_model": settings.embedding_model,
        "num_chunks": str(len(chunks)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    store = IndexStore(settings.index_dir)
    store.save(chunks, embeddings, meta)
    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks.")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the RAG index")
    parser.add_argument("--sources", type=str, default=str(settings.sources_file))
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--allowlist", type=str, default=settings.crawl_allowlist)
    parser.add_argument("--max-pages", type=int, default=settings.crawl_max_pages)
    args = parser.parse_args()

    build_index(
        sources_path=Path(args.sources),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        allowlist=args.allowlist,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
