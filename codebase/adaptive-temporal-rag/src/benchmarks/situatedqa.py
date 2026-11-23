"""SituatedQA temporal split: questions plus a date, and no passages.

Used to train and test the router, not the retriever.
"""

from __future__ import annotations

from datetime import date

import json
import urllib.request

from ..data_schema import QueryItem
from .common import as_list, stable_id


def load_situatedqa_rows(split: str, n: int) -> list[dict]:
    """Stream the official JSONL. The dataset viewer returns 500 for this config."""
    name = {"train": "temp.train.jsonl", "val": "temp.dev.jsonl", "validation": "temp.dev.jsonl", "dev": "temp.dev.jsonl", "test": "temp.test.jsonl"}[
        split
    ]
    url = f"https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-temporal-rag"})
    rows: list[dict] = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if len(rows) >= n:
                break
            text = line.decode("utf-8").strip()
            if text:
                rows.append(json.loads(text))
    return rows


def situatedqa_queries(rows: list[dict], split: str, reference: date | None = None) -> list[QueryItem]:
    reference = reference or date.today()
    queries: list[QueryItem] = []
    for index, row in enumerate(rows):
        question = str(row.get("edited_question") or row.get("question") or "").strip()
        if not question:
            continue
        date_type = str(row.get("date_type") or "")
        raw_date = str(row.get("date") or "").strip()
        answers = as_list(row.get("answer"))
        if date_type == "orig":
            temporal = "timeless"
            needs = 0
            hint = None
        else:
            year = None
            for token in raw_date.replace(",", " ").split():
                if len(token) == 4 and token.isdigit():
                    year = int(token)
                    break
            temporal = "current" if year is not None and year >= reference.year - 1 else "historical"
            needs = 1
            hint = raw_date or None
        queries.append(
            QueryItem(
                query_id=stable_id("sqa", split, str(row.get("id") or index), question[:60]),
                query=question,
                needs_retrieval=needs,
                temporal_class=temporal,  # type: ignore[arg-type]
                time_hint=hint,
                gold_answer=answers[0] if answers else "",
                gold_answers=answers,
                gold_doc_ids=[],
                split=split if split in ("train", "val", "test") else "train",  # type: ignore[arg-type]
                unanswerable=not answers,
            )
        )
    return queries
