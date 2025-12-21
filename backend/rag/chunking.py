from __future__ import annotations

import re


_LETTER_RE = re.compile(r"[A-Za-z\u0410-\u042f\u0430-\u044f]")


def normalize_whitespace(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        fixed = _fix_spaced_letters(cleaned)
        if fixed:
            lines.append(fixed)
    return "\n".join(lines)


def _fix_spaced_letters(line: str) -> str:
    parts = line.split(" ")
    if not parts:
        return line
    single_letters = 0
    for part in parts:
        if len(part) == 1 and _LETTER_RE.fullmatch(part):
            single_letters += 1
    if single_letters < 5:
        return line

    result: list[str] = []
    buffer: list[str] = []
    for part in parts:
        if len(part) == 1 and _LETTER_RE.fullmatch(part):
            buffer.append(part)
            continue
        if buffer:
            if len(buffer) >= 3:
                result.append("".join(buffer))
            else:
                result.extend(buffer)
            buffer = []
        result.append(part)
    if buffer:
        if len(buffer) >= 3:
            result.append("".join(buffer))
        else:
            result.extend(buffer)
    return " ".join(result)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 1 > chunk_size:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if overlap > 0:
                tail = chunk[-overlap:]
                current = [tail] if tail else []
                current_len = len(tail)
            else:
                current = []
                current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + 1

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks
