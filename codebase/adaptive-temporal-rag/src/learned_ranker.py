"""Feature-based temporal ranker. Replaces window, prefer_time, latest-tie, and alpha."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .data_schema import CorpusDoc
from .temporal_rank import lexical_overlap_boost
from .temporal_tagger import TemporalParse, closest_passage_date, inside_target, passage_dates, years_from_target

FEATURE_NAMES = [
    "ce",
    "faiss",
    "overlap",
    "has_date",
    "year_distance",
    "inside_interval",
    "recency",
    "p_current",
    "p_historical",
    "p_timeless",
    "p_retrieve",
    "entity_overlap",
    "current_recency",
    "historical_fit",
]


@dataclass
class RouterProbs:
    p_current: float = 0.33
    p_historical: float = 0.33
    p_timeless: float = 0.34
    p_retrieve: float = 1.0


def featurize(
    query: str,
    doc: CorpusDoc,
    faiss_score: float,
    ce_score: float,
    parsed: TemporalParse,
    router: RouterProbs,
    recency: float,
) -> list[float]:
    moment = closest_passage_date(doc.text, doc.timestamp, parsed)
    entity_overlap = 0.0
    if doc.entity:
        entity_overlap = lexical_overlap_boost(query, doc, ignore_years=True, entity_only=True)
    text_overlap = lexical_overlap_boost(query, doc, ignore_years=True, entity_only=False)
    y_dist = float(years_from_target(moment, parsed))
    hist_prox = 1.0 / (1.0 + y_dist)
    return [
        float(ce_score),
        float(faiss_score),
        float(text_overlap),
        1.0 if passage_dates(doc.text, doc.timestamp) else 0.0,
        y_dist,
        1.0 if inside_target(moment, parsed) else 0.0,
        float(recency),
        float(router.p_current),
        float(router.p_historical),
        float(router.p_timeless),
        float(router.p_retrieve),
        float(entity_overlap),
        float(router.p_current) * float(recency),
        float(router.p_historical) * hist_prox,
    ]


def recency_scores(docs: list[CorpusDoc]) -> list[float]:
    stamped = []
    for doc in docs:
        found = passage_dates(doc.text, doc.timestamp)
        stamped.append(max(found).toordinal() if found else 0)
    order = sorted(range(len(docs)), key=lambda index: stamped[index], reverse=True)
    ranks = [0.0] * len(docs)
    last = max(len(docs) - 1, 1)
    for place, index in enumerate(order):
        ranks[index] = 1.0 - place / last
    return ranks


class LearnedRanker:
    def __init__(self, model: GradientBoostingClassifier) -> None:
        self.model = model

    def score_pairs(
        self,
        query: str,
        docs: list[CorpusDoc],
        faiss_scores: list[float],
        ce_scores: list[float],
        parsed: TemporalParse,
        router: RouterProbs,
    ) -> list[float]:
        if not docs:
            return []
        faiss_norm = _minmax(faiss_scores)
        ce_norm = _minmax(ce_scores)
        recency = recency_scores(docs)
        rows = [
            featurize(query, doc, faiss_norm[i], ce_norm[i], parsed, router, recency[i])
            for i, doc in enumerate(docs)
        ]
        probs = self.model.predict_proba(np.asarray(rows, dtype=np.float32))
        positive = 0
        classes = list(self.model.classes_)
        if 1 in classes and probs.shape[1] > 1:
            positive = classes.index(1)
        gbdt_scores = [float(value) for value in probs[:, positive]]

        # Blend GBDT probability with router-calibrated temporal fit on subject-matched candidates
        has_entity = any((doc.entity or "").strip() for doc in docs)
        overlaps = [
            lexical_overlap_boost(query, doc, ignore_years=True, entity_only=has_entity)
            for doc in docs
        ]
        best_ov = max(overlaps) if overlaps else 0.0
        final_scores: list[float] = []
        for i, doc in enumerate(docs):
            y_dist = rows[i][4]
            hist_prox = 1.0 / (1.0 + y_dist)
            temp_prior = (
                router.p_current * recency[i]
                + router.p_historical * hist_prox
                + router.p_timeless * ce_norm[i]
            )
            subject_gate = 1.0 if (best_ov <= 0 or overlaps[i] >= best_ov - 0.01 or (not has_entity and overlaps[i] >= 0.05)) else 0.25
            score = subject_gate * (0.35 * gbdt_scores[i] + 0.60 * temp_prior + 0.05 * ce_norm[i])
            final_scores.append(float(score))
        return final_scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRanker":
        payload = joblib.load(path)
        return cls(payload["model"])


def train_ranker(rows: list[list[float]], labels: list[int]) -> LearnedRanker:
    model = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        random_state=42,
    )
    model.fit(np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64))
    return LearnedRanker(model)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return [0.5 for _ in values]
    return [(value - lo) / span for value in values]


def contains_gold(text: str, golds: list[str]) -> bool:
    lowered = text.lower()
    return any(gold.lower() in lowered for gold in golds if len(gold.strip()) > 1)
