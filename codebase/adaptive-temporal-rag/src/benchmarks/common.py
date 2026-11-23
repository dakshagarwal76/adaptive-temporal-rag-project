"""Shared chunking and Hugging Face dataset-viewer paging."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from typing import Any

from ..data_schema import CorpusDoc


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def chunk_text(text: str, size: int = 900, overlap: int = 120, limit: int = 8) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < limit:
        end = min(len(text), start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def window_around(text: str, needle: str, size: int = 900) -> str | None:
    if not text or not needle:
        return None
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return None
    half = size // 2
    start = max(0, idx - half)
    end = min(len(text), idx + len(needle) + half)
    return " ".join(text[start:end].split())


def hf_rows(dataset: str, config: str, split: str, offset: int, length: int = 100) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-temporal-rag"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for item in payload.get("rows", []):
        row = item.get("row") if isinstance(item, dict) else None
        if isinstance(row, dict):
            rows.append(row)
    return rows


def page_rows(dataset: str, config: str, split: str, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < n:
        batch = hf_rows(dataset, config, split, offset=offset, length=min(100, n - len(rows)))
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < 100 and len(rows) < n:
            break
    return rows[:n]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def make_doc(doc_id: str, text: str, timestamp: str, title: str, source: str, entity: str = "") -> CorpusDoc:
    return CorpusDoc(
        doc_id=doc_id,
        text=text,
        timestamp=timestamp or "1970-01-01",
        title=title,
        source=source,
        entity=entity,
        topic=source,
    )
