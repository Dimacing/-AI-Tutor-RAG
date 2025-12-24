from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field

from .llm import LLMClient
from .prompt import SourceContext, build_system_prompt, build_user_prompt
from .retrieve import RetrievedChunk

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Citation:
    source_id: int
    title: str
    url: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    self_check_question: str
    recommended_resources: list[Citation]
    quizzes: list["Quiz"] = field(default_factory=list)
    raw_model_output: str | None = None


@dataclass(frozen=True)
class Quiz:
    question: str
    options: list[str]
    correct_index: int
    source_id: int


_NO_ANSWER_TEXT = (
    "\u0412 \u0431\u0430\u0437\u0435 \u0437\u043d\u0430\u043d\u0438\u0439 "
    "\u043d\u0435\u0442 \u0442\u043e\u0447\u043d\u043e\u0433\u043e "
    "\u043e\u0442\u0432\u0435\u0442\u0430 \u043d\u0430 \u044d\u0442\u043e\u0442 "
    "\u0432\u043e\u043f\u0440\u043e\u0441."
)
_NO_ANSWER_CHECK = (
    "\u041f\u043e\u043f\u0440\u043e\u0431\u0443\u0439\u0442\u0435 "
    "\u0443\u0442\u043e\u0447\u043d\u0438\u0442\u044c \u0437\u0430\u043f\u0440\u043e\u0441 "
    "\u0438\u043b\u0438 \u043f\u0435\u0440\u0435\u0444\u043e\u0440\u043c\u0443\u043b\u0438"
    "\u0440\u043e\u0432\u0430\u0442\u044c \u0435\u0433\u043e."
)
_QUIZ_SYSTEM_PROMPT = (
    "You are an AI tutor. Use only the provided sources. "
    "Create multiple-choice questions that are realistic and non-trivial. "
    "Do not use fill-in-the-blank placeholders like ____ or missing words. "
    "Each question must be answerable using a single source. "
    "Return JSON with key 'quizzes'."
)


def _no_answer_result() -> AnswerResult:
    return AnswerResult(
        answer=_NO_ANSWER_TEXT,
        citations=[],
        self_check_question=_NO_ANSWER_CHECK,
        recommended_resources=[],
        quizzes=[],
    )


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    llm: LLMClient | None,
    append_sources: bool = True,
    strict_sources: bool = True,
) -> AnswerResult:
    if llm is None:
        raise LLMUnavailableError("LLM unavailable.")
    if not retrieved:
        if strict_sources:
            return _no_answer_result()
        fallback = "I could not find relevant sources for this question."
        if append_sources:
            fallback = f"{fallback}\n\nSources:\n- none"
        return AnswerResult(
            answer=fallback,
            citations=[],
            self_check_question="Try rephrasing the question to be more specific.",
            recommended_resources=[],
            quizzes=[],
        )

    sources = _build_sources(retrieved)
    if strict_sources and not _has_overlap(question, sources):
        if not _passes_vector_gate(retrieved):
            return _no_answer_result()

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(question, sources)

    model_output = ""
    parsed = None
    answer_from_fallback = False
    try:
        model_output = llm.complete(system_prompt, user_prompt)
        parsed = _try_parse_json(model_output)
    except Exception as exc:
        logger.warning("llm_complete_failed", exc_info=exc)
        raise LLMUnavailableError("LLM unavailable.") from exc

    answer_text = ""
    self_check = ""
    recommended = _to_citations(sources)

    if parsed:
        answer_text = _render_answer_payload(parsed.get("answer", ""), question)
        self_check = _coerce_text(parsed.get("self_check_question", ""))
        recommended_items = parsed.get("recommended_resources", [])
        if not isinstance(recommended_items, list):
            recommended_items = []
        if strict_sources:
            recommended = _filter_recommended(recommended_items, sources) or recommended
        else:
            recommended = _from_resource_list(recommended_items) or recommended
    else:
        candidate = (model_output or "").strip()
        if candidate and not _looks_like_json(candidate):
            answer_text = candidate

    if answer_text:
        answer_text = _focus_answer_on_sql(answer_text, question)

    if strict_sources and llm is not None and answer_text:
        if not _is_grounded_answer(answer_text, sources):
            answer_text = ""
            self_check = ""
        else:
            answer_text = _ensure_citations(answer_text, sources)

    if not answer_text:
        answer_text = _fallback_answer(sources)
        answer_from_fallback = True

    if not self_check:
        self_check = (
            "\u041a\u0430\u043a\u043e\u0439 \u043a\u043b\u044e\u0447\u0435\u0432\u043e\u0439 "
            "\u0432\u044b\u0432\u043e\u0434 \u0438\u0437 \u0438\u0441\u0442\u043e\u0447\u043d\u0438"
            "\u043a\u043e\u0432 \u0432\u044b\u0448\u0435?"
        )

    if strict_sources and not answer_from_fallback and not _answer_mentions_question(answer_text, question):
        return _no_answer_result()

    if strict_sources and not _is_grounded_answer(answer_text, sources):
        return _no_answer_result()

    if answer_text:
        answer_text = _renumber_lists(answer_text)

    if append_sources:
        answer_text = _append_sources(answer_text, _to_citations(sources))

    quizzes = _build_quizzes(question, sources, llm)

    return AnswerResult(
        answer=answer_text,
        citations=_to_citations(sources),
        self_check_question=self_check,
        recommended_resources=recommended,
        quizzes=quizzes,
        raw_model_output=model_output or None,
    )


def _build_sources(retrieved: list[RetrievedChunk]) -> list[SourceContext]:
    grouped: dict[str, list[RetrievedChunk]] = {}
    for item in retrieved:
        key = item.chunk.source_url or item.chunk.title or item.chunk.chunk_id
        grouped.setdefault(key, []).append(item)

    collapsed: list[tuple[float, str, str, str]] = []
    max_chunks_per_source = 3
    for items in grouped.values():
        items.sort(key=lambda x: x.score, reverse=True)
        top_items = items[:max_chunks_per_source]
        combined_text = "\n\n".join([it.chunk.text for it in top_items])
        collapsed.append(
            (
                items[0].score,
                top_items[0].chunk.title,
                top_items[0].chunk.source_url,
                combined_text,
            )
        )

    collapsed.sort(key=lambda x: x[0], reverse=True)
    sources: list[SourceContext] = []
    for idx, (_, title, url, text) in enumerate(collapsed, start=1):
        sources.append(
            SourceContext(
                source_id=idx,
                title=title,
                url=url,
                text=text,
            )
        )
    return sources


def _append_sources(answer: str, citations: list[Citation]) -> str:
    lines = [answer.strip(), "", "Sources:"]
    for citation in citations:
        lines.append(f"[{citation.source_id}] {citation.title} - {citation.url}")
    return "\n".join(lines).strip()


def _to_citations(sources: list[SourceContext]) -> list[Citation]:
    return [Citation(source_id=s.source_id, title=s.title, url=s.url) for s in sources]


def _from_resource_list(items: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title", "resource"))
        url = str(item.get("url", ""))
        citations.append(Citation(source_id=idx, title=title, url=url))
    return citations


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _normalize_command(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([;:,)])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    return cleaned


def _extract_sql_commands(text: str) -> list[str]:
    collapsed = _collapse_whitespace(text)
    patterns = [
        r"CREATE DATABASE [^;]+?;?",
        r"CREATE USER [^;]+?;?",
        r"GRANT [^;]+?;?",
        r"FLUSH PRIVILEGES\s*;?",
        r"USE [^;]+?;?",
        r"EXIT\s*;?",
    ]
    commands: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, collapsed, flags=re.IGNORECASE):
            cmd = _normalize_command(match.group(0))
            upper = cmd.upper()
            if upper.startswith(("CREATE", "GRANT", "FLUSH", "USE", "EXIT")) and not cmd.endswith(";"):
                cmd = f"{cmd};"
            commands.append(cmd)
    return _dedupe_preserve_order(commands)


def _extract_shell_commands(text: str) -> list[str]:
    collapsed = _collapse_whitespace(text)
    patterns = [
        r"(?:sudo\s+)?mysql\s+-u\s+\S+(?:\s+-p)?",
    ]
    commands: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, collapsed, flags=re.IGNORECASE):
            cmd = _normalize_command(match.group(0))
            commands.append(cmd)
    return _dedupe_preserve_order(commands)


def _collect_summary_sentences(
    sources: list[SourceContext],
    max_items: int = 3,
) -> list[str]:
    sentences: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for sentence in _split_sentences(source.text):
            cleaned = _clean_text(sentence)
            if len(cleaned) < 40:
                continue
            if len(cleaned) > 240:
                cleaned = cleaned[:240].rstrip() + "..."
            first = cleaned[0]
            if first.isalpha() and first.islower():
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(cleaned)
            if len(sentences) >= max_items:
                return sentences
    return sentences


def _fallback_answer(sources: list[SourceContext]) -> str:
    if not sources:
        return (
            "\u041d\u0435\u0442 \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d"
            "\u044b\u0445 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u043e\u0432 "
            "\u0434\u043b\u044f \u043e\u0442\u0432\u0435\u0442\u0430 \u043d\u0430 \u0432"
            "\u043e\u043f\u0440\u043e\u0441."
        )
    shell_commands: list[str] = []
    sql_commands: list[str] = []
    for source in sources[:3]:
        shell_commands.extend(_extract_shell_commands(source.text))
        sql_commands.extend(_extract_sql_commands(source.text))
    shell_commands = _dedupe_preserve_order(shell_commands)
    sql_commands = _dedupe_preserve_order(sql_commands)

    parts: list[str] = []
    if shell_commands:
        parts.append("\u041f\u043e\u0434\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435 \u043a MySQL:")
        parts.append("```bash")
        parts.extend(shell_commands)
        parts.append("```")
    if sql_commands:
        parts.append("\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0431\u0430\u0437\u044b \u0438 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044f:")
        parts.append("```sql")
        parts.extend(sql_commands)
        parts.append("```")

    if parts:
        return "\n".join(parts).strip()

    summaries = _collect_summary_sentences(sources)
    if summaries:
        parts.append("\u041a\u0440\u0430\u0442\u043a\u043e \u043f\u043e \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0430\u043c:")
        for sentence in summaries:
            parts.append(f"- {sentence}")
        return "\n".join(parts).strip()
    return (
        "\u041d\u0430 \u043e\u0441\u043d\u043e\u0432\u0435 \u0438\u0441\u0442\u043e"
        "\u0447\u043d\u0438\u043a\u043e\u0432 \u043d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c "
        "\u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u0438\u0442\u044c "
        "\u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 "
        "\u043e\u0442\u0432\u0435\u0442."
    )


def _try_parse_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").lstrip()
    return stripped.startswith("{") and stripped.rstrip().endswith("}")


def _coerce_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _render_answer_payload(value: object, question: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        steps = value.get("steps")
        if isinstance(steps, list):
            filtered = _filter_steps_for_question(question, steps)
            return _render_steps(filtered)
        text = value.get("text")
        if isinstance(text, str):
            return text
        return ""
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            filtered = _filter_steps_for_question(question, value)
            return _render_steps(filtered)
        return "\n".join(_coerce_text(item) for item in value if _coerce_text(item))
    return ""


def _focus_answer_on_sql(answer: str, question: str) -> str:
    if not _question_wants_sql_commands(question):
        return answer

    sql_block = _extract_code_block(answer, "sql", aliases=("mysql",))
    sql_commands: list[str] = []
    if not sql_block:
        sql_commands = _extract_sql_commands(answer)
        if sql_commands:
            sql_block = "\n".join(sql_commands)

    if not sql_block:
        return answer
    if sql_commands and not _has_create_or_grant(sql_commands):
        return answer

    mysql_cmd = _find_mysql_command(answer)
    if not mysql_cmd:
        shell_cmds = _extract_shell_commands(answer)
        mysql_cmd = shell_cmds[0] if shell_cmds else ""

    parts: list[str] = []
    if mysql_cmd:
        parts.append("\u041f\u043e\u0434\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435 \u043a MySQL:")
        parts.append("```bash")
        parts.append(mysql_cmd)
        parts.append("```")
    parts.append("\u0421\u043e\u0437\u0434\u0430\u043d\u0438\u0435 \u0431\u0430\u0437\u044b \u0438 \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044f:")
    parts.append("```sql")
    parts.extend(sql_block.splitlines())
    parts.append("```")
    return "\n".join(parts).strip()


def _extract_code_block(text: str, lang: str, aliases: tuple[str, ...] = ()) -> str:
    for match in _CODE_BLOCK_RE.finditer(text or ""):
        block_lang = (match.group(1) or "").lower()
        content = match.group(2) or ""
        if block_lang == lang or (aliases and block_lang in aliases):
            return content.strip()
        if not block_lang and lang == "sql" and _looks_like_sql_block(content):
            return content.strip()
    return ""


def _find_mysql_command(text: str) -> str:
    for match in _CODE_BLOCK_RE.finditer(text or ""):
        content = match.group(2) or ""
        for line in content.splitlines():
            cleaned = _strip_list_prefix(line.strip())
            if "mysql" in cleaned.lower() and cleaned.startswith(("sudo", "mysql")):
                return cleaned
    for line in (text or "").splitlines():
        cleaned = _strip_list_prefix(line.strip())
        if "mysql" in cleaned.lower() and cleaned.startswith(("sudo", "mysql")):
            return cleaned
    return ""


_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\b\d{2,}(?:\.\d{1,2})?\b")
_WORD_RAW_RE = re.compile(r"[A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451]+")
_WORD_RE = re.compile(r"[A-Za-z\u0410-\u042f\u0430-\u044f0-9_]+")
_CODE_BLOCK_RE = re.compile(r"^[ \t]*```(\w+)?[ \t]*\r?\n(.*?)[ \t]*```", re.DOTALL | re.MULTILINE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+")
_QUIZ_BLOCKLIST = {
    "\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0435",
    "\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u044c",
    "\u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u044c\u0442\u0435",
    "\u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u0438\u0442\u044c",
    "\u0441\u043e\u0437\u0434\u0430\u0439\u0442\u0435",
    "\u0441\u043e\u0437\u0434\u0430\u0442\u044c",
    "\u0432\u044b\u0431\u0435\u0440\u0438\u0442\u0435",
    "\u0432\u044b\u0431\u0440\u0430\u0442\u044c",
    "\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435",
    "\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c",
    "\u0432\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u0435",
    "\u0432\u044b\u043f\u043e\u043b\u043d\u0438\u0442\u044c",
    "\u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0438\u0442\u0435\u0441\u044c",
    "\u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0438\u0442\u044c\u0441\u044f",
    "\u043d\u0430\u0441\u0442\u0440\u043e\u0439\u0442\u0435",
    "\u043d\u0430\u0441\u0442\u0440\u043e\u0438\u0442\u044c",
    "\u043e\u0442\u043a\u0440\u043e\u0439\u0442\u0435",
    "\u043e\u0442\u043a\u0440\u044b\u0442\u044c",
    "\u043d\u0430\u0436\u043c\u0438\u0442\u0435",
    "\u043f\u0440\u043e\u0432\u0435\u0440\u044c\u0442\u0435",
    "\u043f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c",
    "install",
    "create",
    "select",
    "run",
    "open",
    "check",
}
_STOPWORDS = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "\u044d\u0442\u043e",
    "\u0447\u0442\u043e",
    "\u043a\u0430\u043a",
    "\u043b\u0438",
    "\u043a\u0442\u043e",
    "\u0433\u0434\u0435",
    "\u043a\u043e\u0433\u0434\u0430",
    "\u043f\u043e\u0447\u0435\u043c\u0443",
    "\u0437\u0430\u0447\u0435\u043c",
    "\u043b\u0438\u0431\u043e",
    "\u043a\u0430\u043a\u043e\u0439",
    "\u043a\u0430\u043a\u0430\u044f",
    "\u043a\u0430\u043a\u0438\u0435",
    "\u043a\u0430\u043a\u043e\u0432",
    "\u043a\u0430\u043a\u043e\u0432\u044b",
    "\u0438\u043b\u0438",
    "\u0438",
    "\u0430",
    "\u043d\u043e",
    "\u0432",
    "\u043d\u0430",
    "\u043f\u043e",
    "\u0438\u0437",
    "\u0443",
    "\u043e",
    "\u043e\u0431",
    "\u0437\u0430",
    "\u0434\u043b\u044f",
    "\u043f\u0440\u0438",
    "\u043f\u0440\u043e",
    "\u043d\u0430\u0434",
    "\u043f\u043e\u0434",
    "\u0431\u0435\u0437",
    "\u0442\u0430\u043a\u043e\u0435",
    "\u0442\u043e\u0442",
    "\u0442\u0430",
    "\u0442\u043e",
    "\u0442\u0435",
    "\u044d\u0442\u043e\u0442",
    "\u044d\u0442\u0430",
    "\u044d\u0442\u0438",
    "\u044d\u0442\u043e\u0433\u043e",
    "\u044d\u0442\u043e\u0439",
    "\u044d\u0442\u043e\u043c",
}


def _answer_mentions_question(answer: str, question: str) -> bool:
    question_tokens = _extract_tokens(question)
    if not question_tokens:
        return True
    answer_tokens = _extract_tokens(answer)
    if not answer_tokens:
        return False
    q_set = set(question_tokens)
    a_set = set(answer_tokens)
    matched = len(q_set & a_set)
    if len(q_set) <= 2:
        return matched >= len(q_set)
    required = max(1, int(math.ceil(len(q_set) * 0.3)))
    return matched >= required


def _has_valid_citations(text: str, max_id: int) -> bool:
    if not text or max_id <= 0:
        return False
    for match in _CITATION_RE.findall(text):
        try:
            idx = int(match)
        except ValueError:
            continue
        if 1 <= idx <= max_id:
            return True
    return False


def _ensure_citations(answer: str, sources: list[SourceContext]) -> str:
    if not answer or not sources:
        return answer
    if _has_valid_citations(answer, len(sources)):
        return answer
    suffix = f"[{sources[0].source_id}]"
    trimmed = answer.rstrip()
    if trimmed.endswith("```"):
        return f"{trimmed}\n{suffix}"
    return f"{trimmed} {suffix}"


def _filter_recommended(
    items: list[dict],
    sources: list[SourceContext],
) -> list[Citation]:
    if not items:
        return []
    allowed_titles = {source.title.lower() for source in sources if source.title}
    allowed_urls = {source.url for source in sources if source.url}
    filtered: list[Citation] = []
    for item in items:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if (title and title.lower() in allowed_titles) or (url and url in allowed_urls):
            filtered.append(Citation(source_id=len(filtered) + 1, title=title or "resource", url=url))
    return filtered


def _build_quizzes(
    question: str,
    sources: list[SourceContext],
    llm: LLMClient | None,
    max_items: int = 3,
) -> list[Quiz]:
    if llm is None:
        return _build_quizzes_heuristic(question, sources, max_items=max_items)

    prompt = _build_quiz_prompt(question, sources, max_items=max_items)
    try:
        raw = llm.complete(_QUIZ_SYSTEM_PROMPT, prompt)
    except Exception:
        raw = ""
    parsed = _try_parse_json(raw)
    quizzes = _parse_quiz_payload(parsed, sources, max_items=max_items)
    if quizzes:
        return quizzes
    return _build_quizzes_heuristic(question, sources, max_items=max_items)


def _build_quiz_prompt(
    question: str,
    sources: list[SourceContext],
    max_items: int,
) -> str:
    parts = [
        "Sources:",
    ]
    for source in sources:
        text = _clean_text(source.text)
        if len(text) > 1200:
            text = text[:1200] + "..."
        parts.append(f"[{source.source_id}] {source.title} - {source.url}\n{text}")
    parts.append("\nQuestion:")
    parts.append(question)
    parts.append(
        "\nTask:\n"
        f"- Create {max_items} multiple-choice questions in the same language as the question.\n"
        "- Avoid fill-in-the-blank or missing word formats.\n"
        "- Each question must be answerable using a single source.\n"
        "- Provide 3-4 options per question.\n"
        "- Include correct_index (0-based) and source_id.\n"
        "Return JSON only."
    )
    parts.append(
        '\nExample JSON:\n{"quizzes":[{"question":"...","options":["A","B","C"],'
        '"correct_index":0,"source_id":1}]}\n'
    )
    return "\n".join(parts)


def _parse_quiz_payload(
    payload: dict | None,
    sources: list[SourceContext],
    max_items: int,
) -> list[Quiz]:
    if not payload or not isinstance(payload, dict):
        return []
    raw_items = payload.get("quizzes")
    if not isinstance(raw_items, list):
        return []
    result: list[Quiz] = []
    source_count = len(sources)

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question or "____" in question:
            continue
        if len(question) < 8 or len(question) > 240:
            continue
        options_raw = item.get("options")
        if not isinstance(options_raw, list):
            continue
        options = [str(opt).strip() for opt in options_raw if str(opt).strip()]
        options = _dedupe_options(options)
        if len(options) < 2:
            continue

        correct_index = item.get("correct_index")
        try:
            idx = int(correct_index)
        except (TypeError, ValueError):
            continue

        if len(options) > 4:
            if idx < 0 or idx >= len(options):
                continue
            correct_value = options[idx]
            trimmed = [correct_value]
            for opt in options:
                if opt == correct_value:
                    continue
                trimmed.append(opt)
                if len(trimmed) >= 4:
                    break
            options = trimmed
            idx = 0

        if idx < 0 or idx >= len(options):
            continue

        source_id = item.get("source_id")
        try:
            source_id = int(source_id)
        except (TypeError, ValueError):
            source_id = 1
        if source_id < 1 or source_id > source_count:
            source_id = 1

        if not _has_overlap(question, [sources[source_id - 1]]):
            continue

        result.append(
            Quiz(
                question=question,
                options=options,
                correct_index=idx,
                source_id=source_id,
            )
        )
        if len(result) >= max_items:
            break
    return result


def _dedupe_options(options: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for option in options:
        key = option.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(option.strip())
    return result


def _build_quizzes_heuristic(
    question: str,
    sources: list[SourceContext],
    max_items: int = 3,
) -> list[Quiz]:
    if not sources:
        return []
    question_tokens = set(_extract_tokens(question))
    prefer_numbers = _question_prefers_numbers(question_tokens)
    candidates = _collect_sentence_candidates(sources, question_tokens, prefer_numbers)
    quizzes: list[Quiz] = []
    seen_questions: set[str] = set()
    for _, sentence, source_id, source_text in candidates:
        quiz = _quiz_from_sentence(
            sentence=sentence,
            source_id=source_id,
            source_text=source_text,
            question_tokens=question_tokens,
            prefer_numbers=prefer_numbers,
            question=question,
        )
        if not quiz:
            continue
        if quiz.question in seen_questions:
            continue
        seen_questions.add(quiz.question)
        quizzes.append(quiz)
        if len(quizzes) >= max_items:
            break
    return quizzes


def _collect_sentence_candidates(
    sources: list[SourceContext],
    question_tokens: set[str],
    prefer_numbers: bool,
) -> list[tuple[float, str, int, str]]:
    candidates: list[tuple[float, str, int, str]] = []
    seen: set[str] = set()

    def _push(sentence: str, source: SourceContext, score: float) -> None:
        if sentence in seen:
            return
        seen.add(sentence)
        candidates.append((score, sentence, source.source_id, source.text))

    for source in sources:
        for sentence in _split_sentences(source.text):
            if len(sentence) < 20 or len(sentence) > 260:
                continue
            tokens = _extract_tokens(sentence)
            overlap = len(set(tokens) & question_tokens) if question_tokens else 0
            if question_tokens and overlap == 0:
                continue
            has_number = bool(_NUMBER_RE.search(sentence))
            score = overlap * 2
            if has_number:
                score += 2 if prefer_numbers else 1
            score += min(len(tokens), 10) * 0.05
            _push(sentence, source, score)

    if not candidates and question_tokens:
        for source in sources:
            for sentence in _split_sentences(source.text):
                if len(sentence) < 20 or len(sentence) > 260:
                    continue
                tokens = _extract_tokens(sentence)
                has_number = bool(_NUMBER_RE.search(sentence))
                score = (1 if has_number else 0) + min(len(tokens), 10) * 0.05
                _push(sentence, source, score)

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _split_sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]


def _renumber_lists(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_code_block = False
    counter = 1
    in_list = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            continue
        if in_code_block:
            out.append(line)
            continue
        if in_list and _is_bullet_line(stripped):
            out.append(line)
            continue
        match = re.match(r"^(\d+)[.)]\s+", stripped)
        if match:
            content = stripped[match.end():]
            if content:
                prefix = line[: len(line) - len(stripped)]
                out.append(f"{prefix}{counter}. {content}")
                counter += 1
                in_list = True
            else:
                out.append(line)
            continue
        if not stripped:
            out.append(line)
            continue
        if in_list and (line.startswith(" ") or line.startswith("\t")):
            out.append(line)
            continue
        if in_list:
            counter = 1
            in_list = False
        out.append(line)
    return "\n".join(out)


def _render_steps(steps: list[dict]) -> str:
    lines: list[str] = []
    auto_step = 1
    for item in steps:
        if not isinstance(item, dict):
            continue
        step_value = item.get("step")
        try:
            step_num = int(step_value)
        except (TypeError, ValueError):
            step_num = auto_step
        auto_step = step_num + 1

        description = _coerce_text(item.get("description", "")).strip()
        if description:
            lines.append(f"{step_num}. {description}")
        else:
            lines.append(f"{step_num}.")

        substeps = item.get("substeps")
        if isinstance(substeps, list):
            for sub in substeps:
                text = _coerce_text(sub).strip()
                if text:
                    lines.append(f"- {text}")

        commands = item.get("commands")
        if isinstance(commands, list):
            rendered = _render_commands(commands)
            if rendered:
                lines.append(rendered)

    return "\n".join([line for line in lines if line]).strip()


def _render_commands(commands: list[object]) -> str:
    cleaned: list[str] = []
    for cmd in commands:
        text = _coerce_text(cmd).strip()
        if not text:
            continue
        cleaned.append(text)
    if not cleaned:
        return ""

    blocks: list[tuple[str, list[str]]] = []
    current_lang = ""
    current: list[str] = []
    for cmd in cleaned:
        lang = _command_lang(cmd)
        if not current:
            current_lang = lang
            current.append(cmd)
            continue
        if lang != current_lang:
            blocks.append((current_lang, current))
            current_lang = lang
            current = [cmd]
        else:
            current.append(cmd)
    if current:
        blocks.append((current_lang, current))

    rendered: list[str] = []
    for lang, cmds in blocks:
        rendered.append(f"```{lang}")
        rendered.extend(cmds)
        rendered.append("```")
    return "\n".join(rendered)


def _command_lang(command: str) -> str:
    upper = command.strip().upper()
    if upper.startswith(("CREATE", "GRANT", "FLUSH", "USE", "EXIT")):
        return "sql"
    return "bash"


def _filter_steps_for_question(question: str, steps: list[dict]) -> list[dict]:
    if not _question_wants_sql_commands(question):
        return steps

    filtered: list[dict] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        commands = item.get("commands")
        if isinstance(commands, list):
            if any(_command_lang(_coerce_text(cmd)) == "sql" for cmd in commands):
                filtered.append(item)
                continue
            if any("mysql" in _coerce_text(cmd).lower() for cmd in commands):
                filtered.append(item)
                continue

    return filtered or steps


def _question_wants_sql_commands(question: str) -> bool:
    question_lower = (question or "").lower()
    wants_db = any(
        marker in question_lower
        for marker in (
            "mysql",
            "sql",
            "database",
            "\u0431\u0430\u0437",
            "\u0434\u0430\u043d\u043d",
            "\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b",
            "user",
            "wordpress",
        )
    )
    wants_commands = any(
        marker in question_lower for marker in ("\u043a\u043e\u043c\u0430\u043d\u0434", "command")
    )
    return wants_db and wants_commands


def _looks_like_sql_block(text: str) -> bool:
    return bool(re.search(r"\b(CREATE|GRANT|FLUSH|USE|EXIT)\b", text or "", re.IGNORECASE))


def _strip_list_prefix(text: str) -> str:
    return _LIST_PREFIX_RE.sub("", text)


def _is_bullet_line(text: str) -> bool:
    return bool(re.match(r"^[-*\u2022]\s+", text))


def _has_create_or_grant(commands: list[str]) -> bool:
    for cmd in commands:
        upper = cmd.strip().upper()
        if upper.startswith(("CREATE", "GRANT")):
            return True
    return False


def _quiz_from_sentence(
    sentence: str,
    source_id: int,
    source_text: str,
    question_tokens: set[str],
    prefer_numbers: bool,
    question: str,
) -> Quiz | None:
    sentence = _clean_text(sentence)
    if not sentence:
        return None

    if prefer_numbers:
        number = _select_number_target(sentence)
        if number:
            blanked = _blank_value(sentence, number)
            options, correct_index = _build_number_options(number, source_text, question)
            if blanked and len(options) >= 2:
                return Quiz(
                    question=blanked,
                    options=options,
                    correct_index=correct_index,
                    source_id=source_id,
                )

    target = _select_word_target(sentence, question_tokens)
    if target:
        blanked = _blank_sentence(sentence, target)
        if blanked:
            options, correct_index = _build_word_options(target, sentence, question)
            if len(options) >= 2:
                return Quiz(
                    question=blanked,
                    options=options,
                    correct_index=correct_index,
                    source_id=source_id,
                )

    if not prefer_numbers:
        number = _select_number_target(sentence)
        if number:
            blanked = _blank_value(sentence, number)
            options, correct_index = _build_number_options(number, source_text, question)
            if blanked and len(options) >= 2:
                return Quiz(
                    question=blanked,
                    options=options,
                    correct_index=correct_index,
                    source_id=source_id,
                )

    if 20 <= len(sentence) <= 220:
        return Quiz(
            question=(
                "\u0412\u0435\u0440\u043d\u043e \u043b\u0438 \u0443\u0442\u0432\u0435\u0440"
                "\u0436\u0434\u0435\u043d\u0438\u0435: "
                + sentence
            ),
            options=[
                "\u0412\u0435\u0440\u043d\u043e",
                "\u041d\u0435\u0432\u0435\u0440\u043d\u043e",
            ],
            correct_index=0,
            source_id=source_id,
        )
    return None


def _question_prefers_numbers(question_tokens: set[str]) -> bool:
    markers = {
        "\u043f\u043e\u0440\u0442",
        "port",
        "tcp",
        "udp",
        "ip",
        "ipv4",
        "ipv6",
        "\u0432\u0435\u0440\u0441\u0438\u044f",
        "version",
        "release",
        "build",
    }
    return any(token in markers for token in question_tokens)


def _select_number_target(sentence: str) -> str | None:
    numbers = _NUMBER_RE.findall(sentence)
    if not numbers:
        return None
    return numbers[0]


def _select_word_target(
    sentence: str,
    question_tokens: set[str],
) -> tuple[str, str] | None:
    words = _sentence_words(sentence)
    if not words:
        return None
    for word in words:
        if word.lower() in question_tokens:
            return word.lower(), word
    longest = max(words, key=len)
    return longest.lower(), longest


def _sentence_words(sentence: str) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for word in _WORD_RAW_RE.findall(sentence):
        lower = word.lower()
        if len(lower) < 4:
            continue
        if lower in _STOPWORDS:
            continue
        if lower in _QUIZ_BLOCKLIST:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        words.append(word)
    return words


def _blank_value(sentence: str, value: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(value)}\b")
    match = pattern.search(sentence)
    if not match:
        return None
    return sentence[: match.start()] + "____" + sentence[match.end() :]


def _blank_sentence(sentence: str, target: tuple[str, str]) -> str | None:
    token, original = target
    pattern = re.compile(rf"\b{re.escape(original)}\b", re.IGNORECASE)
    if not pattern.search(sentence):
        pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
    match = pattern.search(sentence)
    if not match:
        return None
    return sentence[: match.start()] + "____" + sentence[match.end() :]


def _build_number_options(
    target: str,
    source_text: str,
    question: str,
) -> tuple[list[str], int]:
    numbers: list[str] = []
    for value in _NUMBER_RE.findall(source_text):
        if value == target:
            continue
        if value not in numbers:
            numbers.append(value)
        if len(numbers) >= 3:
            break

    options = [target]
    for value in numbers:
        options.append(value)
        if len(options) >= 3:
            break
    if len(options) < 2:
        synthetic = _synthetic_number_options(target)
        for value in synthetic:
            if value not in options:
                options.append(value)
            if len(options) >= 3:
                break
    if len(options) < 2:
        return [], -1
    return _shuffle_options(options, question, target)


def _build_word_options(
    target: tuple[str, str],
    sentence: str,
    question: str,
) -> tuple[list[str], int]:
    token, original = target
    options = [original]
    for word in _sentence_words(sentence):
        if word.lower() == token:
            continue
        if word.lower() in [opt.lower() for opt in options]:
            continue
        options.append(word)
        if len(options) >= 3:
            break
    if len(options) < 2:
        return [], -1
    return _shuffle_options(options, question, original)


def _shuffle_options(options: list[str], question: str, correct_value: str) -> tuple[list[str], int]:
    seed = abs(hash((question, correct_value))) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(options)
    correct_index = options.index(correct_value)
    return options, correct_index


def _extract_tokens(text: str) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    cleaned = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in _STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def _has_overlap(question: str, sources: list[SourceContext]) -> bool:
    tokens = _extract_tokens(question)
    if not tokens:
        return True
    haystack = " ".join([s.title + " " + s.text for s in sources]).lower()
    matched = sum(1 for token in tokens if token in haystack)
    if len(tokens) <= 2:
        return matched >= len(tokens)
    if len(tokens) <= 4:
        return matched >= 2
    required = max(2, int(math.ceil(len(tokens) * 0.3)))
    return matched >= required


def _passes_vector_gate(
    retrieved: list[RetrievedChunk],
    threshold: float = 0.3,
) -> bool:
    if not retrieved:
        return False
    top_score = max(item.score for item in retrieved)
    return top_score >= threshold


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


def _is_grounded_answer(answer: str, sources: list[SourceContext]) -> bool:
    if not answer:
        return False
    tokens = _extract_tokens(_strip_citations(answer))
    if not tokens:
        return False
    haystack = " ".join([s.title + " " + s.text for s in sources]).lower()
    matched = sum(1 for token in tokens if token in haystack)
    if len(tokens) <= 2:
        required = len(tokens)
        ratio = matched / max(len(tokens), 1)
        return matched >= required and ratio >= 0.6
    if len(tokens) <= 4:
        ratio = matched / max(len(tokens), 1)
        return matched >= 2 and ratio >= 0.5
    ratio = matched / max(len(tokens), 1)
    return matched >= max(2, int(math.ceil(len(tokens) * 0.25))) and ratio >= 0.3


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    return cleaned


def _synthetic_number_options(value: str) -> list[str]:
    if "." in value:
        parts = value.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            major = int(parts[0])
            minor = parts[1]
            options = []
            if major >= 2:
                options.append(f"{major - 2}.{minor}")
            options.append(f"{major + 2}.{minor}")
            return options
    if value.isdigit():
        num = int(value)
        return [str(num - 1), str(num + 1)]
    return []
