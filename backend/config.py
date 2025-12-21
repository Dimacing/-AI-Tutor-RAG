from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or default


def _dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@dataclass(frozen=True)
class Settings:
    project_root: Path
    index_dir: Path
    local_index_dir: Path
    web_index_dir: Path
    uploads_dir: Path
    sources_file: Path
    top_k: int
    min_score: float
    llm_provider: str
    openai_api_key: str | None
    openai_model: str
    gemini_api_key: str | None
    gemini_model: str
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout: int
    gigachat_access_token: str | None
    gigachat_auth_key: str | None
    gigachat_model: str
    gigachat_base_url: str
    gigachat_ca_bundle: str | None
    gigachat_verify_ssl: bool
    embedding_model: str
    embedding_models: list[str]
    log_level: str
    log_user_text: bool
    telegram_bot_token: str | None
    crawl_max_pages: int
    crawl_allowlist: str


def get_settings() -> Settings:
    root = PROJECT_ROOT
    default_embedding = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    default_embedding_models = _dedupe_list(
        [
            default_embedding,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]
    )
    embedding_models = _parse_list(os.getenv("EMBEDDING_MODELS"), default_embedding_models)
    if default_embedding not in embedding_models:
        embedding_models.insert(0, default_embedding)
    return Settings(
        project_root=root,
        index_dir=_resolve_path(os.getenv("INDEX_DIR", "data/index"), root),
        local_index_dir=_resolve_path(os.getenv("LOCAL_INDEX_DIR", "data/index_local"), root),
        web_index_dir=_resolve_path(os.getenv("WEB_INDEX_DIR", "data/index_web"), root),
        uploads_dir=_resolve_path(os.getenv("UPLOADS_DIR", "data/uploads"), root),
        sources_file=_resolve_path(os.getenv("SOURCES_FILE", "data/sources.json"), root),
        top_k=int(os.getenv("TOP_K", "4")),
        min_score=float(os.getenv("MIN_SCORE", "0.2")),
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
        gigachat_access_token=os.getenv("GIGACHAT_ACCESS_TOKEN"),
        gigachat_auth_key=os.getenv("GIGACHAT_AUTH_KEY"),
        gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat"),
        gigachat_base_url=os.getenv(
            "GIGACHAT_BASE_URL",
            "https://gigachat.devices.sberbank.ru",
        ),
        gigachat_ca_bundle=os.getenv("GIGACHAT_CA_BUNDLE"),
        gigachat_verify_ssl=_parse_bool(os.getenv("GIGACHAT_VERIFY_SSL"), True),
        embedding_model=default_embedding,
        embedding_models=embedding_models,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_user_text=_parse_bool(os.getenv("LOG_USER_TEXT"), False),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        crawl_max_pages=int(os.getenv("CRAWL_MAX_PAGES", "30")),
        crawl_allowlist=os.getenv("CRAWL_ALLOWLIST", "/docs/"),
    )
