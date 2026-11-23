"""TimeQA: time-scoped questions over Wikipedia pages, with unanswerable targets."""

from __future__ import annotations

import random
import re

from ..data_schema import CorpusDoc, QueryItem
from ..temporal_tagger import first_date, parse_temporal
from .common import as_list, chunk_text, make_doc, page_rows, stable_id, window_around


def _answers(row: dict) -> list[str]:
    for key in ("targets", "answers", "answer"):
        if key in row:
            return [item for item in as_list(row[key]) if len(item.strip()) > 1]
    return []


def load_timeqa_rows(split: str, n: int) -> list[dict]:
    return page_rows("hugosousa/TimeQA", "default", split, n)


def build_timeqa(
    rows: list[dict],
    split: str,
    reference_year: int = 2026,
) -> tuple[list[CorpusDoc], list[QueryItem]]:
    docs: list[CorpusDoc] = []
    queries: list[QueryItem] = []
    seen_docs: set[str] = set()
    for index, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "")
        if not question or not context:
            continue
        answers = _answers(row)
        unanswerable = len(answers) == 0
        parsed = parse_temporal(question)
        hint = None
        if parsed.target_start is not None:
            hint = parsed.target_start.isoformat()
        year = parsed.target_start.year if parsed.target_start else None
        temporal = "historical"
        if year is not None and year >= reference_year - 1:
            temporal = "current"
        title = str(row.get("idx") or f"timeqa-{split}-{index}")[:80]
        pieces = chunk_text(context, limit=6)
        for answer in answers[:2]:
            window = window_around(context, answer)
            if window and window not in pieces:
                pieces.append(window)
        gold_ids: list[str] = []
        for piece_i, piece in enumerate(pieces):
            doc_id = stable_id("tq", split, title, str(piece_i), piece[:80])
            stamp = first_date(piece) or "1970-01-01"
            if doc_id not in seen_docs:
                docs.append(make_doc(doc_id, piece, stamp, title, "timeqa"))
                seen_docs.add(doc_id)
            lowered = piece.lower()
            if any(answer.lower() in lowered for answer in answers):
                gold_ids.append(doc_id)
        queries.append(
            QueryItem(
                query_id=stable_id("tqq", split, str(index), question[:80]),
                query=question,
                needs_retrieval=1,
                temporal_class=temporal,  # type: ignore[arg-type]
                time_hint=hint,
                gold_answer="" if unanswerable else answers[0],
                gold_answers=[] if unanswerable else answers,
                gold_doc_ids=gold_ids[:8],
                split=split if split in ("train", "val", "test") else "test",  # type: ignore[arg-type]
                unanswerable=unanswerable,
            )
        )
    return docs, queries


def sample_timeqa(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    easy = [row for row in rows if str(row.get("level", "")).lower() == "easy"]
    hard = [row for row in rows if str(row.get("level", "")).lower() == "hard"]
    other = [row for row in rows if row not in easy and row not in hard]
    rng.shuffle(easy)
    rng.shuffle(hard)
    rng.shuffle(other)
    half = n // 2
    picked = easy[:half] + hard[: n - half] + other
    if len(picked) < n:
        rest = [row for row in rows if row not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def question_has_time(question: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", question))
