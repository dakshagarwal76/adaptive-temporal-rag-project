"""Short-answer stage for the heuristic Adaptive pipeline.

Retrieval, the date window, and the sufficiency judge stay as they are.
This stage only shortens an answer with general clause shapes. Dataset-specific
status phrases are left in the heuristic sentence.
"""

from __future__ import annotations

import re

_WHO = re.compile(
    r"\b(who|whom|whose)\b|\bname the\b|\bname a\b",
    re.IGNORECASE,
)
_RESULT = re.compile(
    r"\b(winner|champion|result|outcome|title|prize|medal)\b",
    re.IGNORECASE,
)
_NAME = r"((?:[A-ZÀ-Ý][\w'’\-]*\.?\s+){0,4}[A-ZÀ-Ý][\w'’\-]*)"

# Who-questions can use every pattern. Result questions use only the event patterns.
_WHO_SPANS = (
    re.compile(r"^(.+?)\s+\([^)]+\)$"),
    re.compile(r"\bidentify\s+(.+?)\s+as\b", re.IGNORECASE),
    re.compile(r"\blist\s+(.+?)\s+as\b", re.IGNORECASE),
    re.compile(r"^(.+?)\s+held the office\b", re.IGNORECASE),
    re.compile(_NAME + r"\s+won\b"),
    re.compile(_NAME + r"\s+was crowned\b"),
    re.compile(r"\bawarded to\s+" + _NAME),
    re.compile(r"\bhosted by\s+" + _NAME),
)
_RESULT_SPANS = _WHO_SPANS[4:]

_NOT_A_SPAN = re.compile(
    r"^(official|dated|on|as of|the|records|before|after|in)\b",
    re.IGNORECASE,
)
_YES_NO = re.compile(r"^(is|are|was|were|do|does|did|can|could)\b", re.IGNORECASE)


def asks_who(query: str) -> bool:
    return _WHO.search(query) is not None


def project_answer(query: str, answer: str) -> str:
    """Return a shorter grounded span, or the original heuristic answer."""
    text = (answer or "").strip()
    if not text:
        return text
    if _YES_NO.search(query) and re.match(r"^yes\b", text, re.IGNORECASE):
        return "Yes"
    if asks_who(query):
        patterns = _WHO_SPANS
    elif _RESULT.search(query):
        patterns = _RESULT_SPANS
    else:
        return text
    return _first_span(text, patterns) or text


def _first_span(text: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        span = match.group(1).strip(" ,.:;-")
        if pattern is _WHO_SPANS[0] and not _person_name(span):
            continue
        if not _acceptable(span, text):
            continue
        return span
    return ""


def _person_name(span: str) -> bool:
    """Parenthetical shortening is for a person, not a titled subject."""
    lowered = span.lower()
    if " of " in lowered or lowered.startswith("article"):
        return False
    return True


def _acceptable(span: str, answer: str) -> bool:
    if len(span) < 2 or len(span) >= len(answer):
        return False
    if _NOT_A_SPAN.match(span):
        return False
    return span.lower() in answer.lower()
