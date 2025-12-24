from __future__ import annotations

from dataclasses import dataclass
import re

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
    query_vec = embedder.encode([question])
    if query_vec.size == 0:
        return []

    chunks = store.get_chunks()
    if not chunks:
        return []
    search_k = max(top_k * 10, 40)
    hits = store.search(query_vec[0], top_k=search_k)

    question_tokens = _tokenize(question)
    question_lower = (question or "").lower()
    wants_sql = _question_wants_sql(question_lower)
    wants_wordpress = "wordpress" in question_lower
    scored: list[tuple[float, int, float]] = []
    for idx, score in hits:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        lex = _lexical_score(question_tokens, chunk)
        sql_boost = _sql_boost(chunk.text, wants_sql)
        wp_boost = 0.05 if wants_wordpress and _has_wordpress(chunk) else 0.0
        combined = _combine_scores(score, lex, sql_boost, wp_boost)
        scored.append((combined, idx, score))
    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[RetrievedChunk] = []
    for combined, idx, score in scored:
        if score < min_score:
            continue
        results.append(RetrievedChunk(chunk=chunks[idx], score=combined))
        if len(results) >= max(top_k, 1):
            break

    return results


_WORD_RE = re.compile(r"[A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u04510-9_]+")


def _tokenize(text: str) -> set[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    cleaned: list[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        cleaned.append(token)
    return set(cleaned)


def _lexical_score(question_tokens: set[str], chunk: Chunk) -> float:
    if not question_tokens:
        return 0.0
    haystack = f"{chunk.title} {chunk.text}"
    tokens = _tokenize(haystack)
    if not tokens:
        return 0.0
    overlap = len(question_tokens & tokens)
    return overlap / max(len(question_tokens), 1)


def _combine_scores(
    vector_score: float,
    lexical_score: float,
    sql_boost: float,
    wp_boost: float,
) -> float:
    vector_weight = 0.7
    lex_weight = 0.2
    base = vector_score * vector_weight + lexical_score * lex_weight
    return base + sql_boost + wp_boost


def _question_wants_sql(question: str) -> bool:
    if not question:
        return False
    markers = ("mysql", "sql", "database", "баз", "данн", "пользовател", "user")
    return any(marker in question for marker in markers)


def _sql_boost(text: str, enabled: bool) -> float:
    if not enabled:
        return 0.0
    haystack = text.lower()
    patterns = (
        "create database",
        "create user",
        "grant all",
        "flush privileges",
    )
    hits = sum(1 for pattern in patterns if pattern in haystack)
    if hits <= 0:
        return 0.0
    return min(0.2, 0.08 * hits)


def _has_wordpress(chunk: Chunk) -> bool:
    haystack = f"{chunk.title} {chunk.text}".lower()
    return "wordpress" in haystack
