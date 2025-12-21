from __future__ import annotations

import asyncio
import hashlib
import html
import re
from threading import Lock

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.filters import Command
from aiogram.utils.markdown import hbold, hlink

from .config import get_settings
from .logging_utils import get_logger, setup_logging
from .rag.answer import AnswerResult, Citation, LLMUnavailableError, generate_answer
from .rag.embeddings import EmbeddingModel
from .rag.llm import LLMClient, create_llm
from .rag.retrieve import RetrievedChunk, retrieve
from .rag.store import IndexStore
from .safety import check_safety

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger("telegram_bot")


def _resolve_llm() -> tuple[LLMClient | None, str, str]:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        model = settings.openai_model
        return (
            create_llm("openai", model=model, api_key=settings.openai_api_key),
            provider,
            model,
        )
    if provider == "gemini":
        model = settings.gemini_model
        return (
            create_llm("gemini", model=model, api_key=settings.gemini_api_key),
            provider,
            model,
        )
    if provider in {"deepseek", "deepsick"}:
        model = settings.deepseek_model
        return (
            create_llm(
                "deepseek",
                model=model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            ),
            provider,
            model,
        )
    if provider == "ollama":
        model = settings.ollama_model
        return (
            create_llm(
                "ollama",
                model=model,
                base_url=settings.ollama_base_url,
                timeout=settings.ollama_timeout,
            ),
            provider,
            model,
        )
    if provider == "gigachat":
        model = settings.gigachat_model
        return (
            create_llm(
                "gigachat",
                model=model,
                access_token=settings.gigachat_access_token,
                auth_key=settings.gigachat_auth_key,
                base_url=settings.gigachat_base_url,
                ca_bundle=settings.gigachat_ca_bundle,
                verify_ssl=settings.gigachat_verify_ssl,
            ),
            provider,
            model,
        )
    return None, provider, ""


router = Router()
index_store = IndexStore(settings.index_dir)
index_store.load()
_embedders_lock = Lock()
_embedders: dict[str, EmbeddingModel] = {}


def _get_embedder(model_name: str | None) -> EmbeddingModel:
    resolved = (model_name or "").strip() or settings.embedding_model
    with _embedders_lock:
        embedder = _embedders.get(resolved)
        if embedder is None:
            embedder = EmbeddingModel(resolved)
            _embedders[resolved] = embedder
    return embedder


def _store_embedding_model() -> str:
    meta = index_store.get_meta()
    return str(meta.get("embedding_model") or settings.embedding_model)
llm_client, llm_provider, llm_model = _resolve_llm()


def format_llm_response(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
    return escaped


def _chunk_text(text: str, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= limit:
            final.append(chunk)
        else:
            for idx in range(0, len(chunk), limit):
                final.append(chunk[idx : idx + limit])
    return final


def _hash_user(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:10]


def _log_query(
    question: str,
    results: list[RetrievedChunk],
    answer: AnswerResult,
    user_id: str | None,
    llm_used: bool,
) -> None:
    question_preview = "<hidden>"
    if settings.log_user_text:
        question_preview = question[:200]
    logger.info(
        "telegram_query",
        extra={
            "user": _hash_user(user_id),
            "question": question_preview,
            "matches": len(results),
            "llm_used": llm_used,
            "provider": llm_provider,
            "model": llm_model,
        },
    )


def _format_sources(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = [hbold("Sources:")]
    for citation in citations:
        title = html.escape(citation.title or "source")
        url = citation.url or ""
        if url.startswith("http"):
            safe_url = html.escape(url, quote=True)
            lines.append(f"- {hlink(title, safe_url)}")
        else:
            lines.append(f"- {title}")
    return "\n\n" + "\n".join(lines)


def _build_answer(question: str, user_id: str | None) -> tuple[str, list[Citation]]:
    if not index_store.is_ready():
        return "Index is not ready. Run ingest first.", []
    safe, message = check_safety(question)
    if not safe:
        return message, []
    if llm_client is None:
        raise LLMUnavailableError("LLM unavailable.")
    embedder = _get_embedder(_store_embedding_model())
    results = retrieve(
        question,
        embedder=embedder,
        store=index_store,
        top_k=settings.top_k,
        min_score=settings.min_score,
    )
    answer = generate_answer(question, results, llm_client, append_sources=False, strict_sources=True)
    _log_query(question, results, answer, user_id, llm_client is not None)
    return answer.answer, answer.citations


@router.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    await message.answer(
        f"Hello, {hbold(message.from_user.full_name)}!\n\n"
        "I am your AI tutor for the Cloud.ru materials. Ask a technical question.",
        parse_mode="HTML",
    )


@router.message(F.text)
async def handle_question(message: types.Message) -> None:
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    user_id = str(message.from_user.id) if message.from_user else None
    try:
        answer_text, citations = await asyncio.to_thread(
            _build_answer, message.text, user_id
        )
        response = format_llm_response(answer_text) + _format_sources(citations)
        for chunk in _chunk_text(response):
            await message.answer(
                chunk,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
    except LLMUnavailableError:
        await message.answer("LLM недоступна. Попробуйте позже.")
    except Exception:
        logger.exception("telegram_error", extra={"user": _hash_user(user_id)})
        await message.answer("Sorry, I ran into an error while processing your request.")


async def main() -> None:
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        return
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info("telegram_bot_started", extra={"provider": llm_provider, "model": llm_model})
    await bot.delete_webhook(drop_pending_updates=True)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
