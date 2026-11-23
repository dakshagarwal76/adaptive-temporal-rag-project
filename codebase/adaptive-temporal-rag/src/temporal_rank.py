"""Temporal scoring, adaptive windows, and re-ranking utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional

from .data_schema import CorpusDoc

TemporalMode = Literal["current", "historical", "timeless", "semantic"]


def parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.date()
        except ValueError:
            continue
    # fallback: first 4 digits as year
    if len(value) >= 4 and value[:4].isdigit():
        return date(int(value[:4]), 6, 15)
    raise ValueError(f"Unrecognized date: {value}")


def extract_year_hint(time_hint: Optional[str], query: str = "") -> Optional[int]:
    if time_hint:
        digits = "".join(ch if ch.isdigit() else " " for ch in time_hint).split()
        for tok in digits:
            if len(tok) == 4:
                y = int(tok)
                if 1900 <= y <= 2100:
                    return y
    # scan query for a year (ignoring parenthetical birth years)
    cleaned_query = re.sub(r"\([^)]*\bborn\b[^)]*\)", " ", query or "", flags=re.I)
    for tok in "".join(ch if ch.isdigit() else " " for ch in cleaned_query).split():
        if len(tok) == 4:
            y = int(tok)
            if 1900 <= y <= 2100:
                return y
    return None


def extract_full_date_hint(time_hint: Optional[str], query: str = "") -> Optional[date]:
    """Prefer an explicit YYYY-MM-DD if present in hint or query."""
    blob = f"{time_hint or ''} {query}"
    m = re.search(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", blob)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


@dataclass
class AdaptiveWindow:
    """Adaptive historical date window (half-width in years around a target)."""

    target_year: Optional[int]
    target_date: Optional[date]
    initial_years: float
    max_years: float
    min_docs: int = 3
    final_years: float = 0.0
    expansions: int = 0
    reason: str = ""

    def as_meta(self) -> dict:
        return {
            "target_year": self.target_year,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "initial_years": self.initial_years,
            "final_years": self.final_years or self.initial_years,
            "max_years": self.max_years,
            "expansions": self.expansions,
            "reason": self.reason,
        }


def infer_adaptive_window(
    query: str,
    time_hint: Optional[str] = None,
    reference_date: Optional[date] = None,
    min_docs: int = 3,
) -> AdaptiveWindow:
    """Choose initial/max window width from how specific the time cue is."""
    reference_date = reference_date or date(2025, 10, 1)
    q = (query or "").lower()
    hint = (time_hint or "").lower()
    blob = f"{hint} {q}"

    full = extract_full_date_hint(time_hint, query)
    year = extract_year_hint(time_hint, query)

    # Decade: 1990s / 2010s
    decade = re.search(r"\b((?:19|20)\d)0s\b", blob)
    if decade:
        y0 = int(decade.group(1) + "0")
        return AdaptiveWindow(
            target_year=y0 + 5,
            target_date=date(y0 + 5, 6, 15),
            initial_years=5.0,
            max_years=8.0,
            min_docs=min_docs,
            reason="decade",
        )

    if full is not None:
        return AdaptiveWindow(
            target_year=full.year,
            target_date=full,
            initial_years=0.35,  # ~4 months
            max_years=1.5,
            min_docs=min_docs,
            reason="exact_date",
        )

    # early / mid / late YEAR
    if year is not None and re.search(rf"\b(early|mid|late)\b.*\b{year}\b|\b{year}\b.*(early|mid|late)", blob):
        return AdaptiveWindow(
            target_year=year,
            target_date=date(year, 6, 15),
            initial_years=1.5,
            max_years=3.5,
            min_docs=min_docs,
            reason="year_phase",
        )

    if year is not None:
        # Bare year ("in 2018") — start tight, expand if sparse
        return AdaptiveWindow(
            target_year=year,
            target_date=date(year, 6, 15),
            initial_years=1.0,
            max_years=4.0,
            min_docs=min_docs,
            reason="year",
        )

    # Historical with no year: soft mid-past target, wider search
    target_year = max(1900, reference_date.year - 5)
    return AdaptiveWindow(
        target_year=target_year,
        target_date=date(target_year, 6, 15),
        initial_years=3.0,
        max_years=8.0,
        min_docs=min_docs,
        reason="vague_historical",
    )


def _within_window(doc: CorpusDoc, window: AdaptiveWindow, half_years: float) -> bool:
    from .temporal_tagger import passage_distance_days

    target = window.target_date or date(window.target_year or 2000, 6, 15)
    distance = passage_distance_days(doc.text, doc.timestamp, target)
    if distance is None:
        return False
    return distance <= half_years * 365.25


def filter_by_year_window(
    candidates: list[tuple[CorpusDoc, float]],
    year: Optional[int],
    window_years: int = 2,
) -> list[tuple[CorpusDoc, float]]:
    """Fixed ±window_years filter (used by B3 baseline)."""
    if year is None:
        return candidates
    kept: list[tuple[CorpusDoc, float]] = []
    for doc, score in candidates:
        try:
            y = parse_date(doc.timestamp).year
        except ValueError:
            continue
        if abs(y - year) <= window_years:
            kept.append((doc, score))
    return kept if kept else candidates


def _is_year_token(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and token[:2] in ("19", "20")


def _minmax_norm(values: list[float]) -> list[float]:
    """Map raw cross-encoder scores onto [0, 1] so they can blend with a date score."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


def _doc_date(doc: CorpusDoc) -> date:
    try:
        return parse_date(doc.timestamp)
    except ValueError:
        return date.min


def _subject_surname(query: str) -> str:
    """Extract the last capitalized token of the first proper name in the query."""
    cleaned = re.sub(r"\([^)]*\)", " ", query or "")
    skip = {
        "who", "what", "which", "when", "where", "whom", "whose", "name", "as",
        "in", "on", "the", "a", "an", "between", "before", "after", "from", "until",
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
        "september", "oct", "october", "nov", "november", "dec", "december",
    }
    for match in re.finditer(r"\b([A-ZÀ-Ý][\w'’\-]*\.?(?:\s+[A-ZÀ-Ý][\w'’\-]*\.?)*)", cleaned):
        tokens = [
            tok.strip(".,'’-")
            for tok in match.group(1).split()
            if tok.strip(".,'’-").lower() not in skip and len(tok.strip(".,'’-")) >= 3
        ]
        if tokens:
            return tokens[-1].lower()
    return ""


def filter_subject(
    candidates: list[tuple[CorpusDoc, float]],
    query: str = "",
) -> list[tuple[CorpusDoc, float]]:
    """Keep documents of the asked entity (tagged) or surname (untagged prose)."""
    if not candidates or not query:
        return candidates
    eps = 0.01
    has_entity = any((doc.entity or "").strip() for doc, _ in candidates)
    if has_entity:
        best = max(
            lexical_overlap_boost(query, d, ignore_years=True, entity_only=True)
            for d, _ in candidates
        )
        if best > 0:
            matched = [
                (d, s)
                for d, s in candidates
                if lexical_overlap_boost(query, d, ignore_years=True, entity_only=True) >= best - eps
            ]
            if matched:
                return matched
        return candidates
    overlaps = [
        lexical_overlap_boost(query, doc, ignore_years=True, entity_only=False)
        for doc, _ in candidates
    ]
    best = max(overlaps) if overlaps else 0.0
    surname = _subject_surname(query)
    matched = [
        item
        for item, overlap in zip(candidates, overlaps)
        if overlap >= 0.05
        and (
            not surname
            or surname in f"{item[0].title} {item[0].text}".lower()
            or overlap >= best - eps
        )
    ]
    return matched or list(candidates)


def adaptive_filter_by_window(
    candidates: list[tuple[CorpusDoc, float]],
    window: AdaptiveWindow,
    query: str = "",
    top_k: int = 0,
    use_entity_cut: bool = True,
) -> tuple[list[tuple[CorpusDoc, float]], AdaptiveWindow]:
    """Keep the best-matching entity, then widen the date window until one of its docs is inside.

    Overlap is against ``doc.entity`` only, so template words like "held" cannot
    prefer a later holder. A window that already contains the right entity
    (Parag Agrawal on 2022-03-31) does not grow out to the next appointment.
    """
    if not candidates:
        window.final_years = window.initial_years
        return candidates, window

    pool = filter_subject(candidates, query) if use_entity_cut else list(candidates)
    has_entity = any((doc.entity or "").strip() for doc, _ in candidates)

    half = float(window.initial_years)
    expansions = 0
    kept: list[tuple[CorpusDoc, float]] = []

    while True:
        kept = [(d, s) for d, s in pool if _within_window(d, window, half)]
        at_max = half >= window.max_years - 1e-9
        # One in-window hit is enough: growing further pulls in the next holder.
        if kept or at_max:
            break
        # Expand: grow by ~50% + 0.25y each step
        next_half = min(window.max_years, half * 1.5 + 0.25)
        if next_half <= half + 1e-9:
            break
        half = next_half
        expansions += 1

    window.final_years = half
    window.expansions = expansions

    fill_to = top_k if top_k > 0 else 0
    if not has_entity and fill_to and len(kept) < fill_to:
        kept = _backfill_untagged(kept, pool, window, fill_to)

    if kept:
        return kept, window
    # Right entity never fell inside the cap.
    if pool is not candidates:
        return list(pool), window
    return list(candidates), window


def _backfill_untagged(
    kept: list[tuple[CorpusDoc, float]],
    pool: list[tuple[CorpusDoc, float]],
    window: AdaptiveWindow,
    fill_to: int,
) -> list[tuple[CorpusDoc, float]]:
    """Add subject-matched passages when the date window returned too few."""
    from .temporal_tagger import passage_distance_days

    chosen = list(kept)
    seen = {doc.doc_id for doc, _ in chosen}
    target = window.target_date or date(window.target_year or 2000, 6, 15)
    ranked: list[tuple[float, CorpusDoc, float]] = []
    for doc, sem in pool:
        if doc.doc_id in seen:
            continue
        distance = passage_distance_days(doc.text, doc.timestamp, target)
        temporal = 0.0 if distance is None else 1.0 / (1.0 + distance / 365.25)
        ranked.append((float(sem) + temporal, doc, sem))
    ranked.sort(key=lambda item: item[0], reverse=True)
    for _, doc, sem in ranked:
        if len(chosen) >= fill_to:
            break
        chosen.append((doc, sem))
    return chosen


def temporal_score(
    doc: CorpusDoc,
    mode: TemporalMode,
    query: str = "",
    time_hint: Optional[str] = None,
    reference_date: Optional[date] = None,
    window_years: Optional[float] = None,
) -> float:
    """Return temporal affinity in [0, 1]."""
    reference_date = reference_date or date(2025, 10, 1)
    try:
        doc_date = parse_date(doc.timestamp)
    except ValueError:
        return 0.0

    if mode in ("timeless", "semantic"):
        return 0.5  # neutral

    if mode == "current":
        # Prefer newer documents. Map age in days to [0,1] over ~10 years.
        age_days = max((reference_date - doc_date).days, 0)
        return float(max(0.0, 1.0 - (age_days / (3650.0))))

    # historical: prefer docs near query year/window
    from .temporal_tagger import closest_passage_date, parse_temporal, passage_distance_days, years_from_target

    if not (doc.entity or "").strip() and query:
        parsed = parse_temporal(query, reference=reference_date)
        if parsed.target_start is not None:
            moment = closest_passage_date(doc.text, doc.timestamp, parsed)
            if moment is not None:
                dist = years_from_target(moment, parsed) * 365.25
                half_life_days = max(90.0, (window_years if window_years is not None else 5.0) * 365.25)
                return float(max(0.0, 1.0 - (dist / half_life_days)))

    full = extract_full_date_hint(time_hint, query)
    year = extract_year_hint(time_hint, query)
    if full is not None:
        target = full
    elif year is not None:
        target = date(year, 6, 15)
    else:
        target = date(reference_date.year - 5, 6, 15)

    dist = passage_distance_days(doc.text, doc.timestamp, target)
    if dist is None:
        dist = abs((doc_date - target).days)
    # Decay scale tracks adaptive half-window (default ~5y if unspecified)
    half_life_days = max(90.0, (window_years if window_years is not None else 5.0) * 365.25)
    return float(max(0.0, 1.0 - (dist / half_life_days)))


def combine_scores(
    semantic: float,
    temporal: float,
    alpha: float = 0.6,
) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * semantic + (1.0 - alpha) * temporal


def lexical_overlap_boost(
    query: str,
    doc: CorpusDoc,
    ignore_years: bool = False,
    entity_only: bool = False,
) -> float:
    """Small bonus when query tokens overlap entity/title/text (keeps temporal re-rank on-topic)."""
    stop = {
        "the", "a", "an", "of", "in", "on", "who", "is", "was", "what", "current",
        "currently", "now", "today", "please", "tell", "me", "name", "serving", "as",
        "most", "recent", "latest", "during", "year", "and", "or", "for", "to",
        "between", "before", "after", "until", "since", "from", "which", "did",
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
        "september", "oct", "october", "nov", "november", "dec", "december",
        "position", "team", "organization", "association", "occupation", "player",
        "member", "work", "played", "took", "hold", "held",
    }
    q_toks = {
        t
        for t in "".join(ch.lower() if ch.isalnum() else " " for ch in query).split()
        if len(t) > 2 and t not in stop and not (ignore_years and _is_year_token(t))
    }
    if not q_toks:
        return 0.0
    blob = doc.entity.lower() if entity_only else f"{doc.entity} {doc.title} {doc.text}".lower()
    hits = sum(1 for t in q_toks if t in blob)
    return min(0.25, 0.05 * hits)


def rerank(
    candidates: list[tuple[CorpusDoc, float]],
    mode: TemporalMode,
    query: str = "",
    time_hint: Optional[str] = None,
    alpha: float = 0.6,
    top_k: int = 5,
    reference_date: Optional[date] = None,
    semantic_floor_ratio: float = 0.75,
    window_years: Optional[float] = None,
    prefer_time: bool = False,
) -> list[tuple[CorpusDoc, float, float, float]]:
    """Return list of (doc, final, semantic, temporal).

    Semantic scores are min-max normalized before the alpha blend so a large
    cross-encoder logit cannot swamp a date in [0, 1]. Current mode then keeps
    only the best entity overlap and orders those passages by timestamp.
    """
    if not candidates:
        return []
    norms = _minmax_norm([sem for _, sem in candidates])
    normalized = [(doc, norm) for (doc, _), norm in zip(candidates, norms)]

    # Current questions: same-entity passages tie on overlap; the latest one is the holder.
    if mode == "current" and query:
        overlaps = [lexical_overlap_boost(query, doc, ignore_years=True) for doc, _ in normalized]
        best = max(overlaps) if overlaps else 0.0
        if best > 0:
            eps = 0.01
            tied = [item for item, ov in zip(normalized, overlaps) if ov >= best - eps]
            tied.sort(key=lambda item: (_doc_date(item[0]), item[1]), reverse=True)
            ranked: list[tuple[CorpusDoc, float, float, float]] = []
            n_tied = len(tied)
            for i, (doc, sem) in enumerate(tied):
                temp = temporal_score(
                    doc,
                    mode,
                    query=query,
                    time_hint=time_hint,
                    reference_date=reference_date,
                    window_years=window_years,
                )
                # Rank order is date, then semantic. Stay near 1 so sufficiency does not retry.
                final = 1.0 - (i / max(n_tied, 1)) * 0.05
                ranked.append((doc, final, sem, temp))
            return ranked[:top_k]

    max_sem = max(sem for _, sem in normalized) or 1.0
    floor = max_sem * semantic_floor_ratio
    # Date-first ranking is safe on tagged timelines. On untagged prose a shared
    # year must not outrank the subject, so the semantic floor stays on.
    tagged = any((doc.entity or "").strip() for doc, _ in candidates)
    time_first = bool(prefer_time and tagged and mode == "historical")
    if time_first or (not tagged and len(normalized) <= top_k + 3):
        pooled = list(normalized)
    else:
        pooled = [(d, s) for d, s in normalized if s >= floor] or list(normalized)
        if len(pooled) < top_k:
            seen_ids = {d.doc_id for d, _ in pooled}
            for d, s in sorted(normalized, key=lambda x: x[1], reverse=True):
                if d.doc_id not in seen_ids:
                    pooled.append((d, s))
                    if len(pooled) >= top_k:
                        break

    ranked = []
    for doc, sem in pooled:
        temp = temporal_score(
            doc,
            mode,
            query=query,
            time_hint=time_hint,
            reference_date=reference_date,
            window_years=window_years,
        )
        boost = lexical_overlap_boost(query, doc)
        if mode in ("timeless", "semantic"):
            final = sem + boost
        elif time_first:
            # Asked-date distance decides. Semantic only breaks ties.
            final = temp + 0.01 * sem + boost
        elif not tagged and mode == "historical" and prefer_time:
            # Subject-filtered untagged chunks: keep semantic floor >= 0.5 so body
            # paragraphs with the target date are not zeroed out by min-max scaling.
            sem_adj = 0.5 + 0.5 * sem
            final = combine_scores(sem_adj, temp, alpha=0.45) + boost
        else:
            # Gate temporal by semantic so wrong-entity recency cannot dominate
            final = combine_scores(sem, temp * sem, alpha=alpha) + boost
        ranked.append((doc, final, sem, temp))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
