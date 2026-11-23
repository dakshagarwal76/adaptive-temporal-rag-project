"""JSONL schemas and I/O helpers for corpus and queries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional

TemporalClass = Literal["current", "historical", "timeless"]
Split = Literal["train", "val", "test"]


@dataclass
class CorpusDoc:
    doc_id: str
    text: str
    timestamp: str  # YYYY-MM-DD
    title: str = ""
    source: str = ""
    entity: str = ""
    topic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryItem:
    query_id: str
    query: str
    needs_retrieval: int  # 0 or 1
    temporal_class: TemporalClass
    time_hint: Optional[str] = None  # year or YYYY-MM-DD / range
    gold_answer: str = ""
    gold_doc_ids: list[str] = field(default_factory=list)
    split: Split = "train"
    is_paraphrase: bool = False
    seed_query_id: Optional[str] = None
    gold_answers: list[str] = field(default_factory=list)
    unanswerable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any] | CorpusDoc | QueryItem]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, (CorpusDoc, QueryItem)):
                row = row.to_dict()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _take(row: dict[str, Any], cls: type) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in row.items() if k in known}


def load_corpus(path: str | Path) -> list[CorpusDoc]:
    return [CorpusDoc(**_take(row, CorpusDoc)) for row in read_jsonl(path)]


def load_queries(path: str | Path) -> list[QueryItem]:
    return [QueryItem(**_take(row, QueryItem)) for row in read_jsonl(path)]
