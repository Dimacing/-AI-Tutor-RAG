from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceContext:
    source_id: int
    title: str
    url: str
    text: str


def build_system_prompt() -> str:
    return (
        "You are an AI tutor. Use only the provided sources. "
        "Explain clearly and simply. Do not invent facts. "
        "If the sources describe a procedure or steps, reproduce the steps "
        "that are relevant to the question in order without omitting sub-steps. "
        "Do not include unrelated setup steps. Preserve order and hierarchy; "
        "renumber sequentially if it improves clarity. "
        "Include commands, manifests, and configs in code blocks as written. "
        "Always include citations as [n] matching the sources. "
        "Return JSON only with keys: answer, self_check_question, recommended_resources. "
        "The answer value must be a single string in Markdown."
    )


def build_user_prompt(question: str, sources: list[SourceContext]) -> str:
    parts = ["Sources:"]
    for source in sources:
        parts.append(
            f"[{source.source_id}] {source.title} - {source.url}\n{source.text}"
        )
    parts.append("\nQuestion:")
    parts.append(question)
    parts.append(
        "\nRequirements:\n"
        "- Answer in the same language as the question.\n"
        "- Use simple explanations, but do not summarize away steps.\n"
        "- If sources contain steps, include only those relevant to the question.\n"
        "- Keep structure; renumber lists sequentially if needed for clarity.\n"
        "- Include commands, YAML, and code in fenced code blocks.\n"
        "- Include citations like [1], [2] for each factual claim.\n"
        "- Provide one self-check question.\n"
        "- Provide 1-3 recommended resources as title+url.\n"
        "- Return JSON with answer as a Markdown string, not nested objects.\n"
    )
    return "\n".join(parts)
