from __future__ import annotations

import hashlib
import re
import secrets
import shutil
import time
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .logging_utils import get_logger, setup_logging
from .rag.answer import AnswerResult, LLMUnavailableError, generate_answer
from .rag.embeddings import EmbeddingModel
from .rag.ingest import (
    Document,
    SourceItem,
    build_chunks,
    expand_sources,
    extract_pdf_pages,
    load_document,
    load_sources,
)
from .rag.llm import LLMClient, create_llm
from .rag.retrieve import RetrievedChunk, retrieve
from .rag.store import IndexStore
from .safety import check_safety

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger("api")

app = FastAPI(title="AI Tutor RAG", version="0.1")
WEB_UI_DIR = settings.project_root / "web"
if WEB_UI_DIR.exists():
    app.mount("/site", StaticFiles(directory=WEB_UI_DIR), name="site")

_DEFAULT_CHUNK_SIZE = 1200
_DEFAULT_OVERLAP = 200


@app.get("/", response_model=None)
def root() -> FileResponse | JSONResponse:
    index_path = WEB_UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "ok"})

index_stores = {
    "cloud": IndexStore(settings.index_dir),
    "local": IndexStore(settings.local_index_dir),
    "web": IndexStore(settings.web_index_dir),
}
for store in index_stores.values():
    store.load()
_ALLOWED_LOCAL_EXTENSIONS = {".pdf", ".txt", ".docx"}
_metrics_lock = Lock()
_metrics = {
    "queries_total": 0,
    "queries_blocked": 0,
    "latency_sum_ms": 0.0,
    "last_latency_ms": 0.0,
}
_jobs_lock = Lock()
_jobs: dict[str, dict] = {}
_embedders_lock = Lock()
_embedders: dict[str, EmbeddingModel] = {}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = None
    min_score: float | None = None
    user_id: str | None = None
    subject: str | None = None
    mode: str | None = None
    provider: str | None = None
    model: str | None = None
    debug: bool = False


class IndexBuildRequest(BaseModel):
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    overlap: int | None = Field(default=None, ge=0, le=2000)
    allowlist: str | None = None
    max_pages: int | None = Field(default=None, ge=1, le=1000)
    embedding_model: str | None = Field(default=None, min_length=2, max_length=200)
    async_build: bool = False


class WebIndexRequest(IndexBuildRequest):
    url: str = Field(min_length=5, max_length=2000)
    crawl: bool = True


class CitationOut(BaseModel):
    source_id: int
    title: str
    url: str


class QuizOut(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    source_id: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    self_check_question: str
    recommended_resources: list[CitationOut]
    quiz: QuizOut | None = None
    quizzes: list[QuizOut] | None = None
    debug: dict | None = None


@app.get("/health")
def health() -> dict:
    cloud_store = index_stores["cloud"]
    local_store = index_stores["local"]
    web_store = index_stores["web"]
    defaults = {
        "chunk_size": _DEFAULT_CHUNK_SIZE,
        "overlap": _DEFAULT_OVERLAP,
        "top_k": settings.top_k,
        "min_score": settings.min_score,
        "crawl_max_pages": settings.crawl_max_pages,
        "crawl_allowlist": settings.crawl_allowlist,
    }
    return {
        "status": "ok",
        "index_ready": cloud_store.is_ready(),
        "num_chunks": len(cloud_store.get_chunks()),
        "local_index_ready": local_store.is_ready(),
        "local_num_chunks": len(local_store.get_chunks()),
        "web_index_ready": web_store.is_ready(),
        "web_num_chunks": len(web_store.get_chunks()),
        "defaults": defaults,
        "indexes": {
            "cloud": _index_meta_payload(cloud_store),
            "local": _index_meta_payload(local_store),
            "web": _index_meta_payload(web_store),
        },
    }


@app.get("/embedding-models")
def embedding_models() -> dict:
    models = list(settings.embedding_models)
    if settings.embedding_model not in models:
        models.insert(0, settings.embedding_model)
    return {"default": settings.embedding_model, "models": models}


@app.get("/metrics")
def metrics() -> dict:
    with _metrics_lock:
        avg = 0.0
        if _metrics["queries_total"] > 0:
            avg = _metrics["latency_sum_ms"] / _metrics["queries_total"]
        return {
            "queries_total": _metrics["queries_total"],
            "queries_blocked": _metrics["queries_blocked"],
            "avg_latency_ms": round(avg, 2),
            "last_latency_ms": round(_metrics["last_latency_ms"], 2),
        }


@app.post("/reload-index")
def reload_index(mode: str | None = None) -> dict:
    resolved_mode = _normalize_mode(mode)
    store = index_stores[resolved_mode]
    store.load()
    return {
        "status": "reloaded",
        "mode": resolved_mode,
        "index_ready": store.is_ready(),
    }


@app.get("/index-jobs/{job_id}")
def index_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(job)


@app.post("/cloud-index")
async def cloud_index(request: IndexBuildRequest | None = None) -> dict:
    payload = request or IndexBuildRequest()
    allowlist = payload.allowlist or settings.crawl_allowlist
    max_pages = payload.max_pages or settings.crawl_max_pages
    chunk_size, overlap = _resolve_chunking(payload.chunk_size, payload.overlap)
    embedding_model = _normalize_embedding_model(payload.embedding_model)

    if payload.async_build:
        job_id = _create_job("cloud")
        thread = Thread(
            target=_run_cloud_job,
            args=(job_id, allowlist, max_pages, chunk_size, overlap, embedding_model),
            daemon=True,
        )
        thread.start()
        return {"status": "started", "mode": "cloud", "job_id": job_id}

    num_documents, num_chunks = await run_in_threadpool(
        _build_cloud_index,
        allowlist,
        max_pages,
        chunk_size,
        overlap,
        embedding_model,
    )
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="Документы не были проиндексированы.")

    return {
        "status": "indexed",
        "mode": "cloud",
        "num_documents": num_documents,
        "num_chunks": num_chunks,
    }


@app.post("/web-index")
async def web_index(request: WebIndexRequest) -> dict:
    url = _normalize_url(request.url)
    allowlist = request.allowlist
    if allowlist is None:
        allowlist = _default_allowlist(url)
    max_pages = request.max_pages or settings.crawl_max_pages
    chunk_size, overlap = _resolve_chunking(request.chunk_size, request.overlap)
    embedding_model = _normalize_embedding_model(request.embedding_model)

    if request.async_build:
        job_id = _create_job("web")
        thread = Thread(
            target=_run_web_job,
            args=(
                job_id,
                url,
                request.crawl,
                allowlist,
                max_pages,
                chunk_size,
                overlap,
                embedding_model,
            ),
            daemon=True,
        )
        thread.start()
        return {"status": "started", "mode": "web", "job_id": job_id}

    num_documents, num_chunks = await run_in_threadpool(
        _build_web_index,
        url,
        request.crawl,
        allowlist,
        max_pages,
        chunk_size,
        overlap,
        embedding_model,
    )
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="Документы не были проиндексированы.")

    return {
        "status": "indexed",
        "mode": "web",
        "num_documents": num_documents,
        "num_chunks": num_chunks,
    }


@app.post("/local-index")
async def local_index(
    files: list[UploadFile] = File(...),
    async_build: bool = False,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не загружены.")

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for upload in files:
        filename = upload.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_LOCAL_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый тип файла: {suffix or '<none>'}.",
            )
        base_name = Path(filename).stem or "upload"
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
        target = settings.uploads_dir / f"{safe_base}-{secrets.token_hex(8)}{suffix}"
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        upload.file.close()
        saved_paths.append(target)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="Нет подходящих файлов.")

    resolved_embedding = _normalize_embedding_model(embedding_model)
    resolved_chunk_size, resolved_overlap = _resolve_chunking(chunk_size, overlap)

    if async_build:
        job_id = _create_job("local")
        _update_job(job_id, progress=5, message="файлы загружены")
        thread = Thread(
            target=_run_local_job,
            args=(job_id, saved_paths, resolved_chunk_size, resolved_overlap, resolved_embedding),
            daemon=True,
        )
        thread.start()
        return {"status": "started", "mode": "local", "job_id": job_id}

    num_documents, num_chunks = await run_in_threadpool(
        _build_local_index,
        saved_paths,
        resolved_chunk_size,
        resolved_overlap,
        resolved_embedding,
    )
    if num_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="Не удалось извлечь текст из загруженных файлов.",
        )

    return {
        "status": "indexed",
        "mode": "local",
        "num_files": len(saved_paths),
        "num_documents": num_documents,
        "num_chunks": num_chunks,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    mode = _normalize_mode(request.mode)
    store = index_stores[mode]
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Index '{mode}' is not ready. Run ingest first.",
        )

    safe, message = check_safety(request.question)
    if not safe:
        _record_blocked()
        raise HTTPException(status_code=400, detail=message)

    started = time.perf_counter()
    top_k = request.top_k if request.top_k is not None else settings.top_k
    min_score = request.min_score if request.min_score is not None else settings.min_score
    embedder = _get_embedder(_store_embedding_model(store))
    results = retrieve(
        request.question,
        embedder=embedder,
        store=store,
        top_k=top_k,
        min_score=min_score,
    )
    provider = (request.provider or settings.llm_provider).lower()
    llm, error_message, model = _resolve_llm(provider, request.model)
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM недоступна. Проверьте настройки.")

    try:
        answer = generate_answer(request.question, results, llm, strict_sources=True)
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail="LLM недоступна. Попробуйте позже.")
    _record_latency(time.perf_counter() - started)

    _log_query(request, results, answer, mode, provider, model, llm is not None)

    quizzes_out = [QuizOut(**quiz.__dict__) for quiz in answer.quizzes] if answer.quizzes else []
    quiz_out = quizzes_out[0] if quizzes_out else None
    response = QueryResponse(
        answer=answer.answer,
        citations=[CitationOut(**c.__dict__) for c in answer.citations],
        self_check_question=answer.self_check_question,
        recommended_resources=[CitationOut(**c.__dict__) for c in answer.recommended_resources],
        quiz=quiz_out,
        quizzes=quizzes_out or None,
        debug=_debug_payload(request, results, answer),
    )
    return response


def _normalize_mode(mode: str | None) -> str:
    if not mode:
        return "cloud"
    normalized = mode.strip().lower()
    if normalized not in index_stores:
        raise HTTPException(
            status_code=400,
            detail="Unknown mode. Use 'cloud', 'local', or 'web'.",
        )
    return normalized


def _hash_source(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _create_job(mode: str) -> str:
    job_id = secrets.token_hex(8)
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "mode": mode,
            "status": "running",
            "progress": 0.0,
            "message": "запуск",
            "result": None,
            "error": None,
            "started_at": time.time(),
        }
    return job_id


def _update_job(
    job_id: str,
    progress: float | None = None,
    message: str | None = None,
    status: str | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if progress is not None:
            job["progress"] = round(max(0.0, min(float(progress), 100.0)), 1)
        if message is not None:
            job["message"] = message
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error


def _normalize_embedding_model(value: str | None) -> str:
    cleaned = (value or "").strip()
    return cleaned or settings.embedding_model


def _get_embedder(model_name: str | None) -> EmbeddingModel:
    resolved = _normalize_embedding_model(model_name)
    with _embedders_lock:
        embedder = _embedders.get(resolved)
        if embedder is None:
            embedder = EmbeddingModel(resolved)
            _embedders[resolved] = embedder
    return embedder


def _store_embedding_model(store: IndexStore) -> str:
    meta = store.get_meta()
    return _normalize_embedding_model(meta.get("embedding_model"))


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _index_meta_payload(store: IndexStore) -> dict:
    meta = store.get_meta()
    return {
        "embedding_model": meta.get("embedding_model"),
        "chunk_size": _safe_int(meta.get("chunk_size")),
        "overlap": _safe_int(meta.get("overlap")),
        "max_pages": _safe_int(meta.get("max_pages")),
        "allowlist": meta.get("allowlist"),
        "num_documents": _safe_int(meta.get("num_documents")),
        "num_chunks": _safe_int(meta.get("num_chunks")),
        "created_at": meta.get("created_at"),
        "source": meta.get("source"),
        "url": meta.get("url"),
    }


def _resolve_chunking(
    chunk_size: int | None,
    overlap: int | None,
) -> tuple[int, int]:
    resolved_chunk_size = chunk_size or _DEFAULT_CHUNK_SIZE
    resolved_overlap = _DEFAULT_OVERLAP if overlap is None else overlap
    if resolved_overlap >= resolved_chunk_size:
        raise HTTPException(
            status_code=400,
            detail="Overlap must be smaller than chunk size.",
        )
    return resolved_chunk_size, resolved_overlap


def _normalize_url(value: str) -> str:
    cleaned = value.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL. Use http(s).")
    return parsed.geturl()


def _default_allowlist(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip()
    if not path or path == "/":
        return ""
    trimmed = path.rstrip("/")
    if "/" in trimmed:
        parent = trimmed.rsplit("/", 1)[0]
        return parent or trimmed
    return trimmed


def _load_documents(items: list[SourceItem]) -> list[Document]:
    documents: list[Document] = []
    for item in items:
        try:
            doc = load_document(item)
        except Exception:
            doc = None
        if doc:
            documents.append(doc)
    return documents


def _load_local_documents(paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for page_num, text in extract_pdf_pages(path):
                if not text:
                    continue
                source_url = f"{path}#page={page_num}"
                title = f"{path.name} (стр. {page_num})"
                documents.append(
                    Document(
                        source_id=_hash_source(source_url),
                        title=title,
                        source_url=source_url,
                        text=text,
                    )
                )
            continue

        item = SourceItem(title=path.name, url=None, path=str(path), crawl=False)
        try:
            doc = load_document(item)
        except Exception:
            doc = None
        if doc:
            documents.append(doc)
    return documents


def _progress_callback(job_id: str, start: float, end: float, message: str):
    def _callback(done: int, total: int) -> None:
        if total <= 0:
            progress = end
        else:
            progress = start + (done / total) * (end - start)
        _update_job(job_id, progress=progress, message=message)

    return _callback


def _run_cloud_job(
    job_id: str,
    allowlist: str,
    max_pages: int,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> None:
    try:
        _update_job(job_id, progress=2, message="загрузка источников")
        sources = load_sources(settings.sources_file)
        expanded = expand_sources(sources, allowlist, max_pages)
        _update_job(job_id, progress=10, message="загрузка документов")
        documents = _load_documents(expanded)
        if not documents:
            raise ValueError("Документы не были проиндексированы.")
        _update_job(job_id, progress=18, message="разбиение на чанки")
        chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("Не удалось создать чанки.")
        texts = [chunk.text for chunk in chunks]
        progress_cb = _progress_callback(job_id, 18, 90, "эмбеддинги")
        embedder = _get_embedder(embedding_model)
        embeddings = embedder.encode_with_progress(texts, progress_cb=progress_cb)
        _update_job(job_id, progress=95, message="сохранение индекса")
        meta = {
            "embedding_model": embedding_model,
            "num_chunks": str(len(chunks)),
            "chunk_size": str(chunk_size),
            "overlap": str(overlap),
            "allowlist": allowlist,
            "max_pages": str(max_pages),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "cloud_docs",
            "num_documents": str(len(documents)),
        }
        index_stores["cloud"].save(chunks, embeddings, meta)
        _update_job(
            job_id,
            progress=100,
            status="done",
            message="готово",
            result={"num_documents": len(documents), "num_chunks": len(chunks)},
        )
    except Exception as exc:
        _update_job(
            job_id,
            progress=100,
            status="error",
            message="ошибка",
            error=str(exc),
        )


def _run_web_job(
    job_id: str,
    url: str,
    crawl: bool,
    allowlist: str,
    max_pages: int,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> None:
    try:
        _update_job(job_id, progress=2, message="загрузка источников")
        sources = [SourceItem(title=url, url=url, path=None, crawl=crawl)]
        expanded = expand_sources(sources, allowlist, max_pages) if crawl else sources
        _update_job(job_id, progress=10, message="загрузка документов")
        documents = _load_documents(expanded)
        if not documents:
            raise ValueError("Документы не были проиндексированы.")
        _update_job(job_id, progress=18, message="разбиение на чанки")
        chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("Не удалось создать чанки.")
        texts = [chunk.text for chunk in chunks]
        progress_cb = _progress_callback(job_id, 18, 90, "эмбеддинги")
        embedder = _get_embedder(embedding_model)
        embeddings = embedder.encode_with_progress(texts, progress_cb=progress_cb)
        _update_job(job_id, progress=95, message="сохранение индекса")
        meta = {
            "embedding_model": embedding_model,
            "num_chunks": str(len(chunks)),
            "chunk_size": str(chunk_size),
            "overlap": str(overlap),
            "allowlist": allowlist,
            "max_pages": str(max_pages),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "web_url",
            "num_documents": str(len(documents)),
            "url": url,
        }
        index_stores["web"].save(chunks, embeddings, meta)
        _update_job(
            job_id,
            progress=100,
            status="done",
            message="готово",
            result={"num_documents": len(documents), "num_chunks": len(chunks)},
        )
    except Exception as exc:
        _update_job(
            job_id,
            progress=100,
            status="error",
            message="ошибка",
            error=str(exc),
        )


def _run_local_job(
    job_id: str,
    paths: list[Path],
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> None:
    try:
        _update_job(job_id, progress=10, message="загрузка документов")
        documents = _load_local_documents(paths)
        if not documents:
            raise ValueError("Не удалось извлечь текст из загруженных файлов.")
        _update_job(job_id, progress=18, message="разбиение на чанки")
        chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("Не удалось создать чанки.")
        texts = [chunk.text for chunk in chunks]
        progress_cb = _progress_callback(job_id, 18, 90, "эмбеддинги")
        embedder = _get_embedder(embedding_model)
        embeddings = embedder.encode_with_progress(texts, progress_cb=progress_cb)
        _update_job(job_id, progress=95, message="сохранение индекса")
        meta = {
            "embedding_model": embedding_model,
            "num_chunks": str(len(chunks)),
            "chunk_size": str(chunk_size),
            "overlap": str(overlap),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "local_upload",
            "num_documents": str(len(documents)),
        }
        index_stores["local"].save(chunks, embeddings, meta)
        _update_job(
            job_id,
            progress=100,
            status="done",
            message="готово",
            result={"num_documents": len(documents), "num_chunks": len(chunks)},
        )
    except Exception as exc:
        _update_job(
            job_id,
            progress=100,
            status="error",
            message="ошибка",
            error=str(exc),
        )


def _build_cloud_index(
    allowlist: str,
    max_pages: int,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> tuple[int, int]:
    sources = load_sources(settings.sources_file)
    expanded = expand_sources(sources, allowlist, max_pages)
    documents = _load_documents(expanded)
    if not documents:
        return 0, 0

    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return len(documents), 0

    embedder = _get_embedder(embedding_model)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    meta = {
        "embedding_model": embedding_model,
        "num_chunks": str(len(chunks)),
        "chunk_size": str(chunk_size),
        "overlap": str(overlap),
        "allowlist": allowlist,
        "max_pages": str(max_pages),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "cloud_docs",
        "num_documents": str(len(documents)),
    }
    index_stores["cloud"].save(chunks, embeddings, meta)
    return len(documents), len(chunks)


def _build_web_index(
    url: str,
    crawl: bool,
    allowlist: str,
    max_pages: int,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> tuple[int, int]:
    sources = [SourceItem(title=url, url=url, path=None, crawl=crawl)]
    expanded = expand_sources(sources, allowlist, max_pages) if crawl else sources
    documents = _load_documents(expanded)
    if not documents:
        return 0, 0

    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return len(documents), 0

    embedder = _get_embedder(embedding_model)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    meta = {
        "embedding_model": embedding_model,
        "num_chunks": str(len(chunks)),
        "chunk_size": str(chunk_size),
        "overlap": str(overlap),
        "allowlist": allowlist,
        "max_pages": str(max_pages),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "web_url",
        "num_documents": str(len(documents)),
        "url": url,
    }
    index_stores["web"].save(chunks, embeddings, meta)
    return len(documents), len(chunks)


def _build_local_index(
    paths: list[Path],
    chunk_size: int,
    overlap: int,
    embedding_model: str,
) -> tuple[int, int]:
    documents = _load_local_documents(paths)

    if not documents:
        return 0, 0

    chunks = build_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return len(documents), 0

    embedder = _get_embedder(embedding_model)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    meta = {
        "embedding_model": embedding_model,
        "num_chunks": str(len(chunks)),
        "chunk_size": str(chunk_size),
        "overlap": str(overlap),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "local_upload",
        "num_documents": str(len(documents)),
    }
    index_stores["local"].save(chunks, embeddings, meta)
    return len(documents), len(chunks)


def _record_blocked() -> None:
    with _metrics_lock:
        _metrics["queries_blocked"] += 1


def _record_latency(duration_s: float) -> None:
    with _metrics_lock:
        _metrics["queries_total"] += 1
        duration_ms = duration_s * 1000.0
        _metrics["latency_sum_ms"] += duration_ms
        _metrics["last_latency_ms"] = duration_ms


def _log_query(
    request: QueryRequest,
    results: list[RetrievedChunk],
    answer: AnswerResult,
    mode: str,
    provider: str,
    model: str,
    llm_used: bool,
) -> None:
    user_hash = None
    if request.user_id:
        user_hash = hashlib.sha256(request.user_id.encode("utf-8")).hexdigest()[:10]

    question_preview = "<hidden>"
    if settings.log_user_text:
        question_preview = request.question[:200]

    logger.info(
        "query",
        extra={
            "user": user_hash,
            "question": question_preview,
            "top_k": request.top_k,
            "matches": len(results),
            "mode": mode,
            "llm_used": llm_used,
            "provider": provider,
            "model": model,
        },
    )


def _resolve_llm(
    provider: str,
    model_override: str | None,
) -> tuple[LLMClient | None, str | None, str]:
    provider = provider.lower()
    if provider == "openai":
        model = model_override or settings.openai_model
        llm = create_llm("openai", model=model, api_key=settings.openai_api_key)
        if llm is None:
            return None, "OPENAI_API_KEY is not set.", model
        return llm, None, model
    if provider == "gemini":
        model = model_override or settings.gemini_model
        llm = create_llm("gemini", model=model, api_key=settings.gemini_api_key)
        if llm is None:
            return None, "GEMINI_API_KEY is not set.", model
        return llm, None, model
    if provider in {"deepseek", "deepsick"}:
        model = model_override or settings.deepseek_model
        llm = create_llm(
            "deepseek",
            model=model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        if llm is None:
            return None, "DEEPSEEK_API_KEY is not set.", model
        return llm, None, model
    if provider == "ollama":
        model = model_override or settings.ollama_model
        llm = create_llm(
            "ollama",
            model=model,
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
        )
        if llm is None:
            return None, "OLLAMA_BASE_URL is not set.", model
        return llm, None, model
    if provider == "gigachat":
        model = model_override or settings.gigachat_model
        llm = create_llm(
            "gigachat",
            model=model,
            access_token=settings.gigachat_access_token,
            auth_key=settings.gigachat_auth_key,
            base_url=settings.gigachat_base_url,
            ca_bundle=settings.gigachat_ca_bundle,
            verify_ssl=settings.gigachat_verify_ssl,
        )
        if llm is None:
            return None, "GIGACHAT_AUTH_KEY or GIGACHAT_ACCESS_TOKEN is not set.", model
        return llm, None, model
    return None, f"Unknown LLM provider: {provider}.", model_override or ""


def _debug_payload(
    request: QueryRequest,
    results: list[RetrievedChunk],
    answer: AnswerResult,
) -> dict | None:
    if not request.debug:
        return None
    return {
        "scores": [r.score for r in results],
        "raw_model_output": answer.raw_model_output,
    }
