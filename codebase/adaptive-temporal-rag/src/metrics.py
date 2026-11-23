"""Evaluation metrics for Adaptive Temporal RAG."""

from __future__ import annotations

import re
import string
from collections import Counter
from datetime import date
from typing import Any, Optional

from .data_schema import CorpusDoc, QueryItem
from .temporal_rank import parse_date


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def contains_answer(pred: str, gold: str) -> float:
    p, g = normalize_text(pred), normalize_text(gold)
    if not g:
        return 0.0
    return float(g in p or p in g)


_ABSTAIN = (
    "",
    "insufficient evidence",
    "i dont know",
    "i do not know",
    "no answer",
    "cannot answer",
    "unanswerable",
)


def is_abstention(pred: str) -> bool:
    text = normalize_text(pred)
    return text in _ABSTAIN or text.startswith("insufficient evidence")


def gold_strings(query: QueryItem) -> list[str]:
    found: list[str] = []
    for item in [*(query.gold_answers or []), query.gold_answer]:
        if item and item not in found:
            found.append(item)
    return found


def best_exact_match(pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(exact_match(pred, gold) for gold in golds)


def best_token_f1(pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(token_f1(pred, gold) for gold in golds)


def best_contains(pred: str, golds: list[str]) -> float:
    if not golds:
        return 0.0
    return max(contains_answer(pred, gold) for gold in golds)


def abstention_correct(pred: str, unanswerable: bool) -> float:
    """1 when the system abstains on unanswerable questions and answers the others."""
    abstained = is_abstention(pred)
    return float(abstained if unanswerable else not abstained)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    if not gold_ids:
        return float("nan")
    top = retrieved_ids[:k]
    hits = sum(1 for i in top if i in set(gold_ids))
    return hits / max(len(top), 1)


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int = 5) -> float:
    if not gold_ids:
        return float("nan")
    top = set(retrieved_ids[:k])
    hits = sum(1 for g in gold_ids if g in top)
    return hits / len(gold_ids)


def temporal_correctness(
    retrieved_docs: list[CorpusDoc],
    query: QueryItem,
    reference_date: date | None = None,
) -> float:
    """Heuristic temporal accuracy from top-1 evidence date."""
    if not retrieved_docs:
        # For no-retrieval timeless, count as correct if model skipped retrieval
        return 1.0 if query.temporal_class == "timeless" and query.needs_retrieval == 0 else 0.0

    reference_date = reference_date or date(2025, 10, 1)
    top = retrieved_docs[0]
    try:
        top_date = parse_date(top.timestamp)
    except ValueError:
        return 0.0

    if query.temporal_class == "current":
        # Top doc should be among the newest for that entity when possible
        age_years = (reference_date - top_date).days / 365.25
        return float(age_years <= 3.0)

    if query.temporal_class == "historical":
        year: Optional[int] = None
        if query.time_hint and re.match(r"^\d{4}", query.time_hint):
            year = int(query.time_hint[:4])
        else:
            years = re.findall(r"\b(?:19|20)\d{2}\b", query.query)
            if years:
                year = int(years[0])
        if year is None:
            return 0.5
        return float(abs(top_date.year - year) <= 2)

    # timeless: any evidence is fine
    return 1.0


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    import math

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r and r[key] == r[key] and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return sum(vals) / len(vals) if vals else float("nan")

    def median(key: str) -> float:
        vals = [
            r[key]
            for r in rows
            if key in r and r[key] == r[key] and not (isinstance(r[key], float) and math.isnan(r[key]))
        ]
        if not vals:
            return float("nan")
        vals = sorted(vals)
        mid = len(vals) // 2
        if len(vals) % 2:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2.0)

    return {
        "n": float(len(rows)),
        "em": mean("em"),
        "token_f1": mean("token_f1"),
        "contains": mean("contains"),
        "precision_at_k": mean("precision_at_k"),
        "recall_at_k": mean("recall_at_k"),
        "temporal_acc": mean("temporal_acc"),
        "abstain_acc": mean("abstain_acc"),
        "router_temporal_acc": mean("router_temporal_acc"),
        "router_needs_acc": mean("router_needs_acc"),
        "latency_s": mean("latency_s"),
        "latency_s_median": median("latency_s"),
        "retrieval_rate": mean("did_retrieve"),
    }
