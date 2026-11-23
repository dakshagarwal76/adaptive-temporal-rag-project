"""Date and interval extraction for questions and passages.

Normalization is rule-based. Ranking decisions that use these spans are learned elsewhere.
The reference date defaults to today so a hardcoded exam date is not required.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from dateutil import parser as date_parser

_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})

_YEAR = r"(?:19|20)\d{2}"
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"


@dataclass
class TemporalParse:
    target_start: date | None = None
    target_end: date | None = None
    dates: list[date] = field(default_factory=list)
    relative: str = ""

    @property
    def has_date(self) -> bool:
        return bool(self.dates) or self.target_start is not None


def _clamp_day(year: int, month: int, day: int) -> date:
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, last))


def _month_index(token: str) -> int | None:
    return _MONTHS.get(token.lower().rstrip("."))


def _add_unique(found: list[date], value: date | None) -> None:
    if value is not None and value not in found:
        found.append(value)


def _parse_loose(text: str) -> date | None:
    try:
        parsed = date_parser.parse(text, fuzzy=True, default=datetime(2000, 1, 1))
        if isinstance(parsed, datetime):
            return parsed.date()
        if isinstance(parsed, date):
            return parsed
        return None
    except (ValueError, OverflowError, TypeError):
        return None


def extract_dates(text: str) -> list[date]:
    """Every explicit date mentioned in text, earliest first."""
    if not text:
        return []
    found: list[date] = []
    covered_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            _add_unique(found, date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
            covered_spans.append(match.span())
        except ValueError:
            continue
    for match in re.finditer(rf"\b({_MONTH})\.?\s+(\d{{1,2}}),?\s+({_YEAR})\b", text, flags=re.I):
        month = _month_index(match.group(1))
        if month:
            _add_unique(found, _clamp_day(int(match.group(3)), month, int(match.group(2))))
            covered_spans.append(match.span())
    for match in re.finditer(rf"\b({_MONTH})\.?\s+({_YEAR})\b", text, flags=re.I):
        if any(s <= match.start() and match.end() <= e for s, e in covered_spans):
            continue
        month = _month_index(match.group(1))
        if month:
            _add_unique(found, date(int(match.group(2)), month, 1))
            covered_spans.append(match.span())
    for match in re.finditer(rf"\b({_YEAR})\b", text):
        if any(s <= match.start() and match.end() <= e for s, e in covered_spans):
            continue
        _add_unique(found, date(int(match.group(1)), 6, 15))
    return sorted(found)


def first_date(text: str) -> str | None:
    found = extract_dates(text)
    if not found:
        return None
    # Prefer a fully specified date over a year-only midpoint when both exist.
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso:
        return iso.group(1)
    month = re.search(rf"\b({_MONTH})\.?\s+({_YEAR})\b", text, flags=re.I)
    if month and _month_index(month.group(1)):
        return date(int(month.group(2)), _month_index(month.group(1)) or 1, 1).isoformat()
    return found[0].isoformat()


def parse_temporal(text: str, reference: date | None = None) -> TemporalParse:
    """Target interval for a question, plus every explicit date in the text."""
    reference = reference or date.today()
    raw = re.sub(r"\([^)]*\bborn\b[^)]*\)", " ", text or "", flags=re.I)
    lowered = raw.lower()
    parsed = TemporalParse(dates=extract_dates(raw))

    if re.search(r"\b(today|now|currently|as of now|at present|right now)\b", lowered):
        parsed.relative = "now"
        parsed.target_start = reference
        parsed.target_end = reference
        return parsed
    if re.search(r"\blast year\b|\bprevious year\b", lowered):
        parsed.relative = "last_year"
        parsed.target_start = date(reference.year - 1, 1, 1)
        parsed.target_end = date(reference.year - 1, 12, 31)
        return parsed
    if re.search(r"\bthis year\b", lowered):
        parsed.relative = "this_year"
        parsed.target_start = date(reference.year, 1, 1)
        parsed.target_end = reference
        return parsed

    between = re.search(rf"\b(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\?|$)", raw, flags=re.I)
    if between:
        start = _parse_loose(between.group(1)) or (extract_dates(between.group(1)) or [None])[0]
        end = _parse_loose(between.group(2)) or (extract_dates(between.group(2)) or [None])[0]
        if start and end:
            if end < start:
                start, end = end, start
            parsed.target_start = start
            parsed.target_end = end
            return parsed

    since = re.search(rf"\b(?:since|after)\s+({_MONTH}\.?\s+)?({_YEAR})\b", raw, flags=re.I)
    if since:
        year = int(since.group(2))
        month = _month_index(since.group(1)) if since.group(1) else 1
        parsed.relative = "since"
        parsed.target_start = date(year, month or 1, 1)
        parsed.target_end = reference
        return parsed

    before = re.search(rf"\bbefore\s+({_MONTH}\.?\s+)?({_YEAR})\b", raw, flags=re.I)
    if before:
        year = int(before.group(2))
        month = _month_index(before.group(1)) if before.group(1) else 1
        end_d = date(year, month or 1, 1)
        parsed.relative = "before"
        parsed.target_start = date(max(1900, year - 3), 1, 1)
        parsed.target_end = end_d
        return parsed

    if parsed.dates:
        parsed.target_start = parsed.dates[0]
        parsed.target_end = parsed.dates[-1]
    return parsed


def extract_intervals(text: str) -> list[tuple[date, date]]:
    """Inclusive ranges stated in a passage, such as 1934–1936 or from X to Y."""
    if not text:
        return []
    found: list[tuple[date, date]] = []

    def _add(start: date | None, end: date | None) -> None:
        if start is None or end is None:
            return
        if end < start:
            start, end = end, start
        # Ignore full human lifespan ranges such as (1907–1997) in biography intros.
        if (end.year - start.year) > 45:
            return
        span = (start, end)
        if span not in found:
            found.append(span)

    for match in re.finditer(r"\b((?:19|20)\d{2})\s*(?:[\u2013\u2014\-]|to)\s*((?:19|20)\d{2})\b", text, flags=re.I):
        _add(date(int(match.group(1)), 1, 1), date(int(match.group(2)), 12, 31))
    for match in re.finditer(r"\b((?:19|20)\d{2})\s*[\u2013\u2014\-]\s*(\d{2})\b", text):
        y1 = int(match.group(1))
        y2 = int(match.group(1)[:2] + match.group(2))
        if y2 >= y1 and (y2 - y1) <= 15:
            _add(date(y1, 1, 1), date(y2, 12, 31))
    for match in re.finditer(
        r"\b(?:from|between)\s+(.{3,48}?)\s+(?:to|and)\s+(.{3,48}?)(?:[.;,]|$)",
        text,
        flags=re.I,
    ):
        _add(_parse_loose(match.group(1)), _parse_loose(match.group(2)))
    for match in re.finditer(
        rf"\b({_YEAR})\b.{{0,200}}?\b(?:until|till|through)\s+({_YEAR})\b",
        text,
        flags=re.I,
    ):
        y1, y2 = int(match.group(1)), int(match.group(2))
        if 0 <= (y2 - y1) <= 30:
            _add(date(y1, 1, 1), date(y2, 12, 31))
    for match in re.finditer(
        rf"\bfor\s+(?:a\s+period\s+of\s+)?(\d{{1,2}})\s+years\s+(?:beginning|starting)\s+in\s+({_YEAR})\b",
        text,
        flags=re.I,
    ):
        years_n = int(match.group(1))
        y1 = int(match.group(2))
        if 0 < years_n <= 40:
            _add(date(y1, 1, 1), date(y1 + years_n, 12, 31))
    return found


def passage_distance_days(text: str, timestamp: str, target: date) -> float | None:
    """Days from target to the closest passage date or interval. 0 when target is inside an interval."""
    best: float | None = None
    for start, end in extract_intervals(text):
        if start <= target <= end:
            return 0.0
        edge = start if target < start else end
        best = abs((target - edge).days) if best is None else min(best, abs((target - edge).days))
    for moment in passage_dates(text, timestamp):
        dist = abs((moment - target).days)
        best = dist if best is None else min(best, dist)
    return best


def years_from_target(moment: date | None, parsed: TemporalParse) -> float:
    """0 when moment falls inside the target interval, else distance in years."""
    if moment is None or parsed.target_start is None:
        return 99.0
    end = parsed.target_end or parsed.target_start
    if parsed.target_start <= moment <= end:
        return 0.0
    edge = parsed.target_start if moment < parsed.target_start else end
    return abs((moment - edge).days) / 365.25


def inside_target(moment: date | None, parsed: TemporalParse) -> bool:
    if moment is None or parsed.target_start is None:
        return False
    end = parsed.target_end or parsed.target_start
    return parsed.target_start <= moment <= end


def passage_dates(text: str, timestamp: str = "") -> list[date]:
    found = extract_dates(text)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", timestamp or "") and timestamp != "1970-01-01":
        try:
            stamp = date.fromisoformat(timestamp)
        except ValueError:
            stamp = None
        if stamp and stamp not in found:
            found.append(stamp)
    return found


def closest_passage_date(text: str, timestamp: str, parsed: TemporalParse) -> date | None:
    if parsed.target_start is not None:
        end = parsed.target_end or parsed.target_start
        for start, stop in extract_intervals(text):
            overlap_start = max(start, parsed.target_start)
            overlap_end = min(stop, end)
            if overlap_start <= overlap_end:
                return overlap_start
    found = passage_dates(text, timestamp)
    if not found:
        return None
    if parsed.target_start is None:
        return max(found)
    return min(found, key=lambda item: years_from_target(item, parsed))
