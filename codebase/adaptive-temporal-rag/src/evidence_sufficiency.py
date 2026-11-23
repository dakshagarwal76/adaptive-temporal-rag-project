"""Evidence sufficiency scoring + repair suggestions for Adaptive Temporal RAG."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

logger = logging.getLogger("atr.sufficiency")

# Hard caps for the Adaptive repair loop (Checkpoint B freeze).
MAX_REPAIR_ROUNDS = 3
QUERY_TIMEOUT_S = 60.0

from .data_schema import CorpusDoc
from .temporal_rank import extract_full_date_hint, extract_year_hint, lexical_overlap_boost, parse_date

RepairAction = Literal[
    "widen_window",
    "expand_candidates",
    "raise_topk",
    "relax_to_semantic",
    "boost_recency",
    "none",
]


@dataclass
class SufficiencyResult:
    sufficient: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    suggested_repairs: list[RepairAction] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "suggested_repairs": list(self.suggested_repairs),
            "details": self.details,
        }


@dataclass
class QueryBudget:
    """Wall-clock and iteration budget for one Adaptive query."""

    timeout_s: float = QUERY_TIMEOUT_S
    max_rounds: int = MAX_REPAIR_ROUNDS
    t0: float = field(default_factory=time.perf_counter)
    rounds: int = 0
    timed_out: bool = False
    capped: bool = False

    def expired(self) -> bool:
        if (time.perf_counter() - self.t0) >= self.timeout_s:
            self.timed_out = True
            return True
        return False

    def allow_round(self) -> bool:
        if self.rounds >= self.max_rounds:
            self.capped = True
            return False
        if self.expired():
            return False
        self.rounds += 1
        return True

    def log_stop(self, query: str) -> None:
        if not (self.timed_out or self.capped):
            return
        logger.warning(
            "sufficiency stop query=%r rounds=%s timed_out=%s capped=%s elapsed=%.2fs",
            query[:120],
            self.rounds,
            self.timed_out,
            self.capped,
            time.perf_counter() - self.t0,
        )

    def as_meta(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "timed_out": self.timed_out,
            "capped": self.capped,
            "elapsed_s": round(time.perf_counter() - self.t0, 3),
            "timeout_s": self.timeout_s,
            "max_rounds": self.max_rounds,
        }


_STOP = {
    "the", "a", "an", "of", "in", "on", "who", "is", "was", "were", "what", "which",
    "current", "currently", "now", "today", "please", "tell", "me", "name", "serving",
    "as", "most", "recent", "latest", "during", "year", "and", "or", "for", "to",
    "a", "an", "did", "does", "do", "how", "when", "where", "whom",
}


def _query_tokens(query: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 2 and t not in _STOP and not t.isdigit()
    }


def _doc_blob(doc: CorpusDoc) -> str:
    return f"{doc.entity} {doc.title} {doc.text}".lower()


def _coverage(query_toks: set[str], docs: list[CorpusDoc]) -> float:
    if not query_toks or not docs:
        return 0.0
    blob = " ".join(_doc_blob(d) for d in docs)
    hits = sum(1 for t in query_toks if t in blob)
    return hits / len(query_toks)


def _temporal_fit(
    docs: list[CorpusDoc],
    temporal_class: str,
    query: str,
    time_hint: Optional[str],
) -> float:
    if not docs:
        return 0.0
    if temporal_class in ("timeless", "semantic"):
        return 0.7
    top = docs[0]
    try:
        d = parse_date(top.timestamp)
    except ValueError:
        return 0.2

    if temporal_class == "current":
        # Prefer recent (within ~3y of corpus reference 2025-10)
        age = max(0, (2025 - d.year))
        return float(max(0.0, 1.0 - age / 8.0))

    # historical
    from .temporal_tagger import closest_passage_date, parse_temporal, passage_distance_days, years_from_target

    full = extract_full_date_hint(time_hint, query)
    year = extract_year_hint(time_hint, query)
    best_fit = 0.0
    if full is not None:
        dist = abs((d - full).days) / 365.25
        best_fit = max(best_fit, float(max(0.0, 1.0 - dist / 2.0)))
        p_dist = passage_distance_days(top.text, top.timestamp, full)
        if p_dist is not None:
            best_fit = max(best_fit, float(max(0.0, 1.0 - (p_dist / 365.25) / 2.0)))
    elif year is not None:
        best_fit = max(best_fit, float(max(0.0, 1.0 - abs(d.year - year) / 4.0)))
        p_dist = passage_distance_days(top.text, top.timestamp, parse_date(str(year)))
        if p_dist is not None:
            best_fit = max(best_fit, float(max(0.0, 1.0 - (p_dist / 365.25) / 4.0)))
    else:
        best_fit = 0.55

    parsed = parse_temporal(query)
    if parsed.target_start is not None:
        moment = closest_passage_date(top.text, top.timestamp, parsed)
        if moment is not None:
            y_dist = years_from_target(moment, parsed)
            best_fit = max(best_fit, float(max(0.0, 1.0 - y_dist / 4.0)))
    return best_fit


def _weak_answer_text(answer: str) -> bool:
    a = (answer or "").lower()
    markers = [
        "insufficient",
        "could not find",
        "no local evidence",
        "not enough evidence",
        "i don't know",
        "i do not know",
        "cannot answer",
        "no evidence",
        "unable to",
    ]
    return any(m in a for m in markers) or len(a.strip()) < 2


def assess_evidence_sufficiency(
    query: str,
    docs: list[CorpusDoc],
    scores: list[float] | None = None,
    temporal_class: str = "timeless",
    time_hint: Optional[str] = None,
    answer: Optional[str] = None,
    min_score: float = 0.55,
    entity_matched: bool = True,
) -> SufficiencyResult:
    """Heuristic sufficiency judge (fast, offline, always available)."""
    scores = scores or []
    reasons: list[str] = []
    repairs: list[RepairAction] = []
    q_toks = _query_tokens(query)

    if not docs:
        return SufficiencyResult(
            sufficient=False,
            score=0.0,
            reasons=["no_documents"],
            suggested_repairs=["expand_candidates", "relax_to_semantic", "widen_window"],
            details={"n_docs": 0},
        )

    coverage = _coverage(q_toks, docs)
    top_score = float(scores[0]) if scores else 0.0
    mean_score = float(sum(scores) / len(scores)) if scores else 0.0
    overlap_boost = lexical_overlap_boost(query, docs[0])
    temp_fit = _temporal_fit(docs, temporal_class, query, time_hint)

    # Weighted sufficiency score in [0,1]
    score = (
        0.35 * coverage
        + 0.25 * min(1.0, max(0.0, top_score))
        + 0.15 * min(1.0, max(0.0, mean_score))
        + 0.15 * temp_fit
        + 0.10 * min(1.0, overlap_boost / 0.25)
    )

    if coverage < 0.35:
        reasons.append("low_query_term_coverage")
        repairs.append("expand_candidates")
        repairs.append("raise_topk")
    if top_score < 0.25:
        reasons.append("weak_top_retrieval_score")
        repairs.append("expand_candidates")
    if not entity_matched:
        reasons.append("wrong_entity")
        repairs.append("expand_candidates")
        repairs.append("relax_to_semantic")
        score = min(score, max(0.0, min_score - 0.01))
    if temporal_class == "historical" and temp_fit < 0.45:
        reasons.append("poor_temporal_fit")
        repairs.append("widen_window")
    if temporal_class == "current" and temp_fit < 0.45:
        reasons.append("stale_evidence_for_current")
        repairs.append("boost_recency")
        repairs.append("expand_candidates")
    if answer is not None and _weak_answer_text(answer):
        reasons.append("weak_generated_answer")
        score = min(score, 0.4)
        repairs.extend(["expand_candidates", "widen_window", "relax_to_semantic"])

    # Deduplicate repairs preserving order
    seen: set[str] = set()
    uniq_repairs: list[RepairAction] = []
    for r in repairs:
        if r not in seen:
            seen.add(r)
            uniq_repairs.append(r)
    if not uniq_repairs:
        uniq_repairs = ["none"]

    sufficient = (
        entity_matched
        and score >= min_score
        and "weak_generated_answer" not in reasons
        and "wrong_entity" not in reasons
    )
    if sufficient and not reasons:
        reasons.append("ok")

    return SufficiencyResult(
        sufficient=sufficient,
        score=score,
        reasons=reasons,
        suggested_repairs=uniq_repairs,
        details={
            "n_docs": len(docs),
            "coverage": round(coverage, 3),
            "top_score": round(top_score, 3),
            "mean_score": round(mean_score, 3),
            "temporal_fit": round(temp_fit, 3),
            "threshold": min_score,
        },
    )
