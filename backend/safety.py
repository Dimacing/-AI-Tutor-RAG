import re

PROFANITY = {
    "fuck",
    "shit",
    "bitch",
    "asshole",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\+?\d[\d\s()\-]{7,}\d\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def check_safety(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    if any(word in lowered for word in PROFANITY):
        return False, "Please avoid offensive language."

    if EMAIL_RE.search(text) or PHONE_RE.search(text) or CARD_RE.search(text):
        return False, "Please avoid sharing personal data in questions."

    return True, ""
