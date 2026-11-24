"""Large-router dataset helpers and A100 training.

This module does not change DistilBERT loading. `load_query_analyzer` still
reads `artifacts/analyzer.pt`. DeBERTa bundles are loaded only through
`load_large_router`, from whatever directory the A100 notebook saved.

Label schema matches `src/query_analyzer.py`:
  needs_retrieval in {0, 1}
  temporal_class in {current, historical, timeless}
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

TEMPORAL_LABELS = ("current", "historical", "timeless")

_PREFIXES = (
    "please tell me",
    "quick question",
    "i need to know",
    "can you answer",
    "kindly answer",
    "please answer",
    "quickly",
    "tell me",
    "as asked",
    "question",
)
_SUFFIXES = ("please be brief", "answer briefly")

_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
)
_MONTH_RE = re.compile(r"\b(?:" + "|".join(_MONTHS) + r")\b")
_YEAR_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
_DATE_RE = re.compile(r"\b(\d{4}) (\d{2}) (\d{2})\b")

_CURRENT_RE = re.compile(
    r"\b(?:currently|nowadays|today|tonight|right now|as of now|at present|"
    r"at the moment|most recent|this year|these days)\b|\blatest\b|"
    r"\bcurrent\s+(?:president|prime minister|premier|chancellor|ceo|governor|mayor|"
    r"monarch|pope|government|administration|champion|officeholder|holder|leader|"
    r"season|year|status|events)\b"
)
_REL_PAST_RE = re.compile(
    r"\b(?:last year|previous year|last time|formerly|former|previously|used to)\b"
)
_WHEN_RE = re.compile(
    r"\b(?:when did|when was|when were|what year|which year|in what year|what date|"
    r"which date)\b"
)
_OFFICE = (
    r"president|prime minister|premier|chancellor|ceo|chief executive|governor|mayor|"
    r"monarch|king of|queen of|pope|secretary general|chief justice|speaker of|"
    r"champion|head coach|managing director"
)
_OFFICE_NOW_RE = re.compile(
    rf"\b(?:who is|whos|name the|who'?s)\b.{{0,40}}\b(?:current|present|sitting|incumbent|{_OFFICE})\b"
)
_OFFICE_PAST_RE = re.compile(rf"\b(?:who was|who were)\b.{{0,40}}\b(?:{_OFFICE})\b")
_STABLE_RE = re.compile(
    r"\b(?:who wrote|who invented|who discovered|who painted|who composed|author of|"
    r"chemical formula|chemical symbol|speed of light)\b"
)

_ENTITY_SLOT = (
    r"(?:(?!(?:in|on|at|during|before|after|from|between|when|while|which|that|"
    r"because|into|by|as|who|what|where|how|hold)\b)[a-z0-9]+)"
    r"(?:\s+(?:(?:of|the|for|de|and|la|le|von|van)\s+)?"
    r"(?!(?:in|on|at|during|before|after|from|between|when|while|which|that|"
    r"because|into|by|as|who|what|where|how|hold)\b)[a-z0-9]+){0,5}"
)
_SLOT_RE = re.compile(r"\{(entity|year|date|mon|num)\}")
_SLOT_PATTERN = {
    "entity": _ENTITY_SLOT,
    "year": r"(?:1[0-9]{3}|20[0-9]{2})",
    "date": r"\d{4} \d{2} \d{2}",
    "mon": r"\b(?:" + "|".join(_MONTHS) + r")\b",
    "num": r"\d+",
}

_STOP_ENTITIES = set(_MONTHS) | {
    "what", "which", "when", "where", "who", "how", "from", "that", "this", "with",
    "have", "were", "was", "the", "and", "for", "his", "her", "its", "their", "not",
    "you", "are", "but", "can", "did", "does", "has", "had", "been", "will", "would",
    "about", "into", "over", "after", "before", "between", "during", "under", "while",
}

TEMPORAL_SOURCES = {
    "timeqa", "timeqa_local", "situatedqa", "situatedqa_local", "timequestions",
    "tempquestions", "temporalwiki", "tot", "mintaka",
}
CURRENT_SOURCES = {"freshqa", "realtimeqa"}
EXAM_SOURCES = {"arc", "commonsense_qa", "openbookqa", "sciq", "qasc", "squad_v2"}

_TIMEQA_SUBJECTS = (
    re.compile(
        r"\b(?:did|does)\s+([A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,5})\s+"
        r"(?:hold|take|have|serve)\b"
    ),
    re.compile(
        r"\bof\s+([A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,5})\s+"
        r"(?:from|in|during|between|hold)\b"
    ),
    re.compile(
        r"\b([A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,5})['’]s\s+"
        r"(?:spouse|position|wife|husband|office)\b"
    ),
)


def canonicalize(text: str) -> str:
    """Lowercase, strip accents and wrappers used by the old paraphraser."""
    raw = unicodedata.normalize("NFKD", text or "")
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = raw.lower().replace("'", "").replace("’", "")
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    changed = True
    while changed and raw:
        changed = False
        for prefix in _PREFIXES:
            if raw.startswith(prefix + " "):
                raw = raw[len(prefix):].strip()
                changed = True
        for suffix in _SUFFIXES:
            if raw.endswith(" " + suffix):
                raw = raw[: -(len(suffix) + 1)].strip()
                changed = True
    return raw


def extract_timeqa_subjects(question: str) -> list[str]:
    found: list[str] = []
    for pattern in _TIMEQA_SUBJECTS:
        for match in pattern.finditer(question or ""):
            name = " ".join(match.group(1).replace(".", " ").split())
            if name and name not in found:
                found.append(name)
    return found


def wiki_title(idx: str) -> str:
    if not idx or "/wiki/" not in idx:
        return ""
    title = idx.split("/wiki/", 1)[1].split("#", 1)[0]
    title = title.replace("_", " ")
    try:
        from urllib.parse import unquote
        title = unquote(title)
    except Exception:
        pass
    return " ".join(title.split())


def label_question(
    question: str,
    source: str,
    *,
    date_type: str | None = None,
    reference_year: int = 2026,
) -> dict[str, Any] | None:
    """Map a real question onto the analyzer label schema.

    Years and explicit current wording decide the class before the source
    default. Timeless general-QA questions are non-retrieval negatives.
    Current and historical questions need retrieval.
    """
    text = " ".join((question or "").split())
    if len(text.split()) < 3:
        return None
    canonical = canonicalize(text)
    if len(canonical) < 8:
        return None
    years = [int(match.group(0)) for match in _YEAR_RE.finditer(canonical)]
    hint = str(max(years)) if years else None

    def pack(temporal: str, needs: int, confidence: float) -> dict[str, Any]:
        return {
            "needs_retrieval": int(needs),
            "temporal_class": temporal,
            "time_hint": hint,
            "label_confidence": float(confidence),
        }

    if years:
        latest = max(years)
        if latest >= reference_year - 1:
            return pack("current", 1, 0.95)
        return pack("historical", 1, 0.95)
    if _CURRENT_RE.search(canonical):
        return pack("current", 1, 0.95)
    if _REL_PAST_RE.search(canonical):
        return pack("historical", 1, 0.85)
    if _WHEN_RE.search(canonical):
        return pack("historical", 1, 0.7)
    if _OFFICE_NOW_RE.search(canonical) and not _OFFICE_PAST_RE.search(canonical):
        return pack("current", 1, 0.7)
    if _OFFICE_PAST_RE.search(canonical):
        return pack("historical", 1, 0.65)

    source_key = (source or "").lower()
    if source_key in CURRENT_SOURCES:
        return pack("current", 1, 0.85)
    if source_key in {"situatedqa", "situatedqa_local"} and (date_type or "") == "orig":
        if _STABLE_RE.search(canonical):
            return pack("timeless", 0, 0.55)
        return pack("current", 1, 0.8)
    if source_key in {"timeqa", "timeqa_local"}:
        return pack("historical", 1, 0.9)
    if source_key in TEMPORAL_SOURCES:
        return pack("historical", 1, 0.6)
    if source_key in EXAM_SOURCES or source_key in {"nq_open", "triviaqa", "squad_v2"}:
        confidence = 0.8 if source_key in EXAM_SOURCES else 0.75 if source_key == "squad_v2" else 0.6
        return pack("timeless", 0, confidence)
    return None


def _mask_time(canonical: str) -> str:
    text = _DATE_RE.sub(" {date} ", canonical)
    text = _MONTH_RE.sub(" {mon} ", text)
    text = _YEAR_RE.sub(" {year} ", text)
    text = re.sub(r"\b\d+\b", " {num} ", text)
    return re.sub(r"\s+", " ", text).strip()


def _skeleton(canonical: str, entities: Iterable[str]) -> str | None:
    text = canonical
    replaced = False
    for entity in sorted({e for e in entities if e}, key=len, reverse=True):
        if entity and entity in text:
            text = text.replace(entity, " {entity} ", 1)
            replaced = True
            break
    text = _mask_time(text)
    text = re.sub(r"\s+", " ", text).strip()
    if "{entity}" not in text and "{year}" not in text and "{date}" not in text and "{mon}" not in text and "{num}" not in text:
        return None
    literals = _SLOT_RE.split(text)
    literal_chunks = literals[::2]
    if not any(len(chunk.strip()) >= 6 for chunk in literal_chunks):
        return None
    if not replaced and "{entity}" in text:
        return None
    return text


def _compile_skeleton(skeleton: str) -> re.Pattern[str] | None:
    parts = _SLOT_RE.split(skeleton)
    if len(parts) < 2:
        return None
    chunks: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            chunks.append(re.escape(part))
        else:
            chunks.append(_SLOT_PATTERN[part])
    try:
        return re.compile("^" + "".join(chunks) + "$")
    except re.error:
        return None


class _Template:
    def __init__(self, skeleton: str, pattern: re.Pattern[str]) -> None:
        self.skeleton = skeleton
        self.pattern = pattern
        prefix = skeleton.split("{", 1)[0]
        self.prefix = prefix

    def matches(self, canonical: str) -> bool:
        if self.prefix and not canonical.startswith(self.prefix):
            return False
        return self.pattern.fullmatch(canonical) is not None


class LeakageIndex:
    """Exact, entity, and template overlap against the external test sets."""

    def __init__(self) -> None:
        self.exact: set[str] = set()
        self.entities: set[str] = set()
        self.templates: list[_Template] = []
        self._template_keys: set[str] = set()
        self._ent_re: re.Pattern[str] | None = None

    def add_exact(self, questions: Iterable[str]) -> int:
        added = 0
        for question in questions:
            canonical = canonicalize(question)
            if canonical and canonical not in self.exact:
                self.exact.add(canonical)
                added += 1
        return added

    def add_entities(self, entities: Iterable[str]) -> int:
        added = 0
        for entity in entities:
            canonical = canonicalize(entity)
            if not self._keep_entity(canonical):
                continue
            if canonical not in self.entities:
                self.entities.add(canonical)
                added += 1
        if added:
            self._compile_entities()
        return added

    def add_question_templates(self, questions: Iterable[str]) -> int:
        added = 0
        for question in questions:
            canonical = canonicalize(question)
            skeleton = _skeleton(canonical, self.entities)
            if not skeleton or skeleton in self._template_keys:
                continue
            pattern = _compile_skeleton(skeleton)
            if pattern is None:
                continue
            self._template_keys.add(skeleton)
            self.templates.append(_Template(skeleton, pattern))
            added += 1
        return added

    @staticmethod
    def _keep_entity(canonical: str) -> bool:
        if not canonical or len(canonical) < 3:
            return False
        if canonical in _STOP_ENTITIES:
            return False
        tokens = canonical.split()
        if len(tokens) > 8:
            return False
        if len(tokens) == 1 and canonical in _STOP_ENTITIES:
            return False
        return True

    def _compile_entities(self) -> None:
        parts = sorted(self.entities, key=len, reverse=True)
        if not parts:
            self._ent_re = None
            return
        body = "|".join(re.escape(part) for part in parts)
        self._ent_re = re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")

    def classify(self, question: str, entity: str | None = None) -> tuple[str, str] | None:
        canonical = canonicalize(question)
        if not canonical:
            return ("exact", "")
        if canonical in self.exact:
            return ("exact", canonical)
        if entity:
            ent = canonicalize(entity)
            if ent and (ent in self.entities or (self._ent_re and self._ent_re.search(ent))):
                return ("entity", ent)
        if self._ent_re is not None:
            match = self._ent_re.search(canonical)
            if match:
                return ("entity", match.group(0))
        for template in self.templates:
            if template.matches(canonical):
                return ("template", template.skeleton)
        return None

    def reason(self, question: str, entity: str | None = None) -> str | None:
        found = self.classify(question, entity=entity)
        if found is None:
            return None
        return found[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "forbidden_questions": sorted(self.exact),
            "forbidden_entities": sorted(self.entities),
            "forbidden_templates": [template.skeleton for template in self.templates],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeakageIndex":
        index = cls()
        index.exact = set(payload.get("forbidden_questions") or [])
        index.entities = set(payload.get("forbidden_entities") or [])
        index._compile_entities()
        for skeleton in payload.get("forbidden_templates") or []:
            pattern = _compile_skeleton(skeleton)
            if pattern is None or skeleton in index._template_keys:
                continue
            index._template_keys.add(skeleton)
            index.templates.append(_Template(skeleton, pattern))
        return index


def class_targets(counts: dict[str, int], min_total: int, max_total: int) -> dict[str, int]:
    """Most balanced per-class caps that can reach min_total when the data allows it."""
    labels = list(TEMPORAL_LABELS)
    counts = {label: int(counts.get(label, 0)) for label in labels}
    available = sum(counts.values())
    if available <= min_total:
        return counts
    minority = min(counts.values())
    if minority * 3 >= min_total:
        cap = min(minority, max_total // 3)
        if cap * 3 < min_total:
            cap = min(minority, math.ceil(min_total / 3))
        return {label: min(counts[label], cap) for label in labels}
    lo, hi = 0, max(counts.values())
    best: dict[str, int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = {label: min(counts[label], mid) for label in labels}
        if sum(cand.values()) >= min_total:
            best = cand
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        return counts
    while sum(best.values()) > max_total:
        label = max(labels, key=lambda name: best[name])
        if best[label] <= 0:
            break
        best[label] -= 1
    return best


def select_balanced_rows(
    rows: list[dict[str, Any]],
    min_total: int = 50000,
    max_total: int = 75000,
    max_synthetic_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prefer real rows. Synthetic rows fill starved classes only, and stay <= 20%."""
    rng = random.Random(seed)
    labels = list(TEMPORAL_LABELS)
    real: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    synthetic: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for row in rows:
        label = row.get("temporal_class")
        if label not in real:
            continue
        bucket = synthetic if row.get("is_synthetic") else real
        bucket[label].append(row)
    for bucket in (real, synthetic):
        for items in bucket.values():
            rng.shuffle(items)
    available = {label: len(real[label]) + len(synthetic[label]) for label in labels}
    targets = class_targets(available, min_total, max_total)
    chosen: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    real_taken = {label: 0 for label in labels}
    for label in labels:
        take = min(targets[label], len(real[label]))
        chosen[label].extend(real[label][:take])
        real_taken[label] = take
    deficits = {label: targets[label] - len(chosen[label]) for label in labels}
    synthetic_taken = 0
    synthetic_index = {label: 0 for label in labels}
    while True:
        options = [
            label for label in labels
            if deficits[label] > 0 and synthetic_index[label] < len(synthetic[label])
        ]
        if not options:
            break
        label = max(options, key=lambda name: deficits[name])
        new_total = sum(len(items) for items in chosen.values()) + 1
        if (synthetic_taken + 1) / new_total > max_synthetic_fraction:
            break
        chosen[label].append(synthetic[label][synthetic_index[label]])
        synthetic_index[label] += 1
        synthetic_taken += 1
        deficits[label] -= 1
    spare = {label: real[label][real_taken[label]:] for label in labels}
    spare_index = {label: 0 for label in labels}
    while sum(len(items) for items in chosen.values()) < min_total:
        options = [label for label in labels if spare_index[label] < len(spare[label])]
        if not options:
            break
        if sum(len(items) for items in chosen.values()) >= max_total:
            break
        label = min(options, key=lambda name: len(chosen[name]))
        chosen[label].append(spare[label][spare_index[label]])
        spare_index[label] += 1
    selected = [row for label in labels for row in chosen[label]]
    rng.shuffle(selected)
    synthetic_n = sum(1 for row in selected if row.get("is_synthetic"))
    total = len(selected)
    report = {
        "targets": targets,
        "available": available,
        "real_available": {label: len(real[label]) for label in labels},
        "synthetic_available": {label: len(synthetic[label]) for label in labels},
        "selected": {label: sum(1 for row in selected if row["temporal_class"] == label) for label in labels},
        "synthetic_selected": synthetic_n,
        "synthetic_fraction": (synthetic_n / total) if total else 0.0,
    }
    return selected, report


def assign_splits(
    rows: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    """Keep every group_id inside one split, stratified by temporal class."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("group_id") or canonicalize(row["query"]))
        grouped[(row["temporal_class"], key)].append(row)
    buckets: dict[str, list[list[dict[str, Any]]]] = {label: [] for label in TEMPORAL_LABELS}
    for (label, _key), items in grouped.items():
        if label in buckets:
            buckets[label].append(items)
    assigned: list[dict[str, Any]] = []
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    for label, groups in buckets.items():
        rng.shuffle(groups)
        count = len(groups)
        n_test = int(round(count * test_ratio))
        n_val = int(round(count * val_ratio))
        if count >= 10:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= count:
            n_test = count // 10
            n_val = count // 10
        train_cut = count - n_test - n_val
        val_cut = count - n_test
        for index, items in enumerate(groups):
            if index < train_cut:
                split = "train"
            elif index < val_cut:
                split = "val"
            else:
                split = "test"
            for row in items:
                copied = dict(row)
                copied["split"] = split
                assigned.append(copied)
    return assigned


def scan_leakage(rows: Iterable[dict[str, Any]], index: LeakageIndex) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    total = 0
    for row in rows:
        total += 1
        found = index.classify(row.get("query", ""), entity=row.get("entity") or None)
        if found is None:
            continue
        counts[found[0]] += 1
        if len(examples) < 12:
            examples.append({"query": row.get("query", ""), "reason": found[0], "detail": found[1][:180]})
    return {
        "rows": total,
        "leaks": int(sum(counts.values())),
        "by_reason": dict(counts),
        "examples": examples,
    }


def expected_calibration_error(
    confidences: list[float],
    correct: list[int],
    n_bins: int = 15,
) -> float:
    if not confidences:
        return float("nan")
    totals = [0] * n_bins
    acc = [0.0] * n_bins
    conf = [0.0] * n_bins
    for score, ok in zip(confidences, correct):
        clipped = min(max(float(score), 0.0), 1.0)
        bucket = min(n_bins - 1, int(clipped * n_bins))
        if clipped == 1.0:
            bucket = n_bins - 1
        totals[bucket] += 1
        acc[bucket] += float(ok)
        conf[bucket] += clipped
    n = len(confidences)
    ece = 0.0
    for bucket in range(n_bins):
        if totals[bucket] == 0:
            continue
        ece += (totals[bucket] / n) * abs(acc[bucket] / totals[bucket] - conf[bucket] / totals[bucket])
    return float(ece)


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def score_predictions(
    necessity_gold: list[int],
    necessity_pred: list[int],
    necessity_conf: list[float],
    temporal_gold: list[int],
    temporal_pred: list[int],
    temporal_conf: list[float],
    head_conf: list[float] | None = None,
) -> dict[str, Any]:
    """Macro-F1, per-class F1, confusion, and ECE. Labels are integer ids."""
    n = len(necessity_gold)
    nec_cm = [[0, 0], [0, 0]]
    temp_cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for gold, pred in zip(necessity_gold, necessity_pred):
        nec_cm[int(gold)][int(pred)] += 1
    for gold, pred in zip(temporal_gold, temporal_pred):
        temp_cm[int(gold)][int(pred)] += 1

    def class_report(cm: list[list[int]], names: list[str]) -> dict[str, Any]:
        per_class = {}
        supported = []
        all_labels = []
        for index, name in enumerate(names):
            tp = cm[index][index]
            fp = sum(cm[row][index] for row in range(len(names))) - tp
            fn = sum(cm[index]) - tp
            support = sum(cm[index])
            metrics = _prf(tp, fp, fn)
            metrics["support"] = int(support)
            per_class[name] = metrics
            if support > 0:
                supported.append(metrics["f1"])
            all_labels.append(metrics["f1"] if support > 0 else 0.0)
        correct = sum(cm[i][i] for i in range(len(names)))
        total = sum(sum(row) for row in cm)
        return {
            "per_class": per_class,
            "macro_f1": float(sum(supported) / len(supported)) if supported else float("nan"),
            "macro_f1_all_labels": float(sum(all_labels) / len(all_labels)) if all_labels else float("nan"),
            "accuracy": float(correct / total) if total else float("nan"),
        }

    necessity = class_report(nec_cm, ["no_retrieval", "needs_retrieval"])
    temporal = class_report(temp_cm, list(TEMPORAL_LABELS))
    nec_correct = [int(g == p) for g, p in zip(necessity_gold, necessity_pred)]
    temp_correct = [int(g == p) for g, p in zip(temporal_gold, temporal_pred)]
    both_correct = [int(a == 1 and b == 1) for a, b in zip(nec_correct, temp_correct)]
    macro = 0.5 * necessity["macro_f1"] + 0.5 * temporal["macro_f1"]
    report: dict[str, Any] = {
        "n": n,
        "val_macro_f1": float(macro) if macro == macro else float("nan"),
        "necessity": necessity,
        "temporal": temporal,
        "necessity_confusion": nec_cm,
        "temporal_confusion": temp_cm,
        "necessity_ece": expected_calibration_error(necessity_conf, nec_correct),
        "temporal_ece": expected_calibration_error(temporal_conf, temp_correct),
    }
    if head_conf is not None:
        report["confidence_head_ece"] = expected_calibration_error(head_conf, both_correct)
    return report


def self_test() -> None:
    """Fast checks that do not download data or load a model."""
    current = label_question("Who is the current secretary general of the UN?", "nq_open", reference_year=2026)
    historical = label_question("Which position did Ada Lovelace hold from 1833 to 1840?", "timeqa", reference_year=2026)
    timeless = label_question("Which of the following is a chemical change?", "arc", reference_year=2026)
    when = label_question("When did the Berlin Wall fall?", "squad_v2", reference_year=2026)
    recent = label_question("Who won the election in 2026?", "nq_open", reference_year=2026)
    assert current and current["temporal_class"] == "current" and current["needs_retrieval"] == 1
    assert historical and historical["temporal_class"] == "historical" and historical["needs_retrieval"] == 1
    assert timeless and timeless["temporal_class"] == "timeless" and timeless["needs_retrieval"] == 0
    assert when and when["temporal_class"] == "historical"
    assert recent and recent["temporal_class"] == "current"
    electric = label_question(
        "Which set of conditions will produce an electric current in a closed loop of wire?",
        "arc",
        reference_year=2026,
    )
    lion = label_question("who is the voice of scar in lion king", "nq_open", reference_year=2026)
    governor = label_question("who is running for governor of florida", "nq_open", reference_year=2026)
    assert electric and electric["temporal_class"] == "timeless" and electric["needs_retrieval"] == 0
    assert lion and lion["temporal_class"] == "timeless"
    assert governor and governor["temporal_class"] == "current" and governor["needs_retrieval"] == 1

    index = LeakageIndex()
    questions = [
        "As of now, who is the Prime Minister of India?",
        "What is Photosynthesis?",
        "Who was the President of the United States in 2020?",
        "Which position did Knox Cunningham hold from May 1955 to Apr 1956?",
    ]
    entities = [
        "Prime Minister of India",
        "Photosynthesis",
        "President of the United States",
        "Knox Cunningham",
    ]
    index.add_entities(entities)
    index.add_exact(questions)
    index.add_question_templates(questions)
    assert index.reason("As of now, who is the Prime Minister of India?") == "exact"
    assert index.reason("As of now, who is the CEO of Tesla?") == "template"
    assert index.reason("What is gravity?") == "template"
    assert index.reason("What is the role of mitochondria in animal cells?") is None
    assert index.reason("Who was the CEO of Apple in 2014?") == "template"
    assert index.reason("Which position did Ada Lovelace hold from Jan 1833 to Dec 1840?") == "template"
    assert index.reason("Which statement best explains why a sealed bottle collapses after cooling?") is None
    assert index.reason("Tell me about the Prime Minister of India please") == "entity"
    restored = LeakageIndex.from_dict(index.to_dict())
    assert restored.reason("What is gravity?") == "template"

    balanced = class_targets({"current": 3000, "historical": 40000, "timeless": 40000}, 50000, 75000)
    assert balanced["current"] == 3000
    assert balanced["historical"] == balanced["timeless"]
    assert sum(balanced.values()) >= 50000
    equal = class_targets({"current": 20000, "historical": 20000, "timeless": 20000}, 50000, 75000)
    assert equal == {"current": 20000, "historical": 20000, "timeless": 20000}
    capped = class_targets({"current": 30000, "historical": 30000, "timeless": 30000}, 50000, 75000)
    assert capped == {"current": 25000, "historical": 25000, "timeless": 25000}
    tiny = class_targets({"current": 100, "historical": 100, "timeless": 100}, 50000, 75000)
    assert sum(tiny.values()) == 300


def _row_text(row: Any) -> tuple[str, int, str, float]:
    if isinstance(row, dict):
        return (
            str(row["query"]),
            int(row["needs_retrieval"]),
            str(row["temporal_class"]),
            float(row.get("label_confidence", 1.0)),
        )
    return (
        str(row.query),
        int(row.needs_retrieval),
        str(row.temporal_class),
        float(getattr(row, "label_confidence", 1.0)),
    )


def load_split_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_large_router(artifacts_dir: str | Path, device: str | None = None):
    """Load a saved DeBERTa (or other) bundle without touching DistilBERT artifacts.

    The directory must contain analyzer.pt and analyzer_meta.json written by
    QueryAnalyzerService.save. Temperature and blend_confidence are read from
    the meta file when present. The default `load_query_analyzer` path is unchanged.
    """
    from .query_analyzer import QueryAnalyzerService

    artifacts_dir = Path(artifacts_dir)
    service = QueryAnalyzerService.from_pretrained_bundle(artifacts_dir, device=device)
    meta = json.loads((artifacts_dir / "analyzer_meta.json").read_text(encoding="utf-8"))
    if "temperature" in meta:
        service.temperature = float(meta["temperature"])
    if "blend_confidence" in meta:
        service.blend_confidence = bool(meta["blend_confidence"])
    return service


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_temperature(logits, labels) -> float:
    import torch

    best_t, best_nll = 1.0, float("inf")
    target = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    stacked = logits if isinstance(logits, torch.Tensor) else torch.stack(logits)
    for step in range(5, 31):
        temperature = step / 10
        log_probs = torch.log_softmax(stacked / temperature, dim=-1)
        nll = float(-log_probs[torch.arange(len(target)), target].mean())
        if nll < best_nll:
            best_t, best_nll = temperature, nll
    return float(best_t)


def evaluate_rows(
    model,
    tokenizer,
    rows: list[Any],
    device: str,
    max_length: int = 128,
    temperature: float = 1.0,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Necessity/temporal F1, confusion matrices, and ECE for a loaded encoder."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    from .query_analyzer import LABEL2ID

    class _EvalDS(Dataset):
        def __init__(self) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            text, needs, temporal, confidence = _row_text(self.rows[idx])
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "needs": torch.tensor(needs, dtype=torch.long),
                "temporal": torch.tensor(LABEL2ID[temporal], dtype=torch.long),
                "label_confidence": torch.tensor(confidence, dtype=torch.float),
            }

    if not rows:
        return {"n": 0, "val_macro_f1": float("nan")}
    loader = DataLoader(_EvalDS(), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    nec_gold: list[int] = []
    nec_pred: list[int] = []
    nec_conf: list[float] = []
    temp_gold: list[int] = []
    temp_pred: list[int] = []
    temp_conf: list[float] = []
    head_conf: list[float] = []
    temporal_logits = []
    temporal_labels = []
    temp = max(float(temperature), 1e-3)
    with torch.inference_mode():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            nec_probs = torch.softmax(out["necessity_logits"] / temp, dim=-1)
            temp_probs = torch.softmax(out["temporal_logits"] / temp, dim=-1)
            nec_hat = nec_probs.argmax(-1).cpu()
            temp_hat = temp_probs.argmax(-1).cpu()
            nec_gold.extend(batch["needs"].tolist())
            nec_pred.extend(nec_hat.tolist())
            nec_conf.extend(nec_probs.max(-1).values.cpu().tolist())
            temp_gold.extend(batch["temporal"].tolist())
            temp_pred.extend(temp_hat.tolist())
            temp_conf.extend(temp_probs.max(-1).values.cpu().tolist())
            head_conf.extend(out["confidence"].detach().cpu().tolist())
            temporal_logits.append(out["temporal_logits"].detach().cpu())
            temporal_labels.extend(batch["temporal"].tolist())
    report = score_predictions(nec_gold, nec_pred, nec_conf, temp_gold, temp_pred, temp_conf, head_conf)
    report["temperature"] = temp
    if temporal_logits:
        report["temporal_logits"] = torch.cat(temporal_logits, dim=0)
        report["temporal_labels"] = temporal_labels
    return report


def _atomic_torch_save(obj: Any, path: str | Path) -> None:
    """Write a PyTorch pickle atomically (`*.tmp` then os.replace)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _capture_rng_states() -> dict[str, Any]:
    import torch

    states: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": None,
        "cuda": None,
    }
    try:
        import numpy as np

        states["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            states["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return states


def _restore_rng_states(states: dict[str, Any] | None) -> None:
    if not states:
        return
    import torch

    if states.get("python") is not None:
        random.setstate(states["python"])
    if states.get("torch") is not None:
        torch.set_rng_state(states["torch"])
    if states.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(states["numpy"])
        except Exception:
            pass
    if states.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["cuda"])
        except Exception:
            pass


def _epoch_permutation(n: int, seed: int, epoch: int) -> list[int]:
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch) * 10007)
    return torch.randperm(n, generator=generator).tolist()


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def find_resume_checkpoint(
    resume_from: str | Path | None = None,
    output_dir: str | Path | None = None,
    extra_search: Iterable[str | Path] | None = None,
) -> Path | None:
    """Find the best `latest_checkpoint.pt` for a resumable Kaggle/Colab run.

    Preference: explicit `resume_from` or $RESUME_FROM, then `/kaggle/working`,
    then `/kaggle/input` (previous session output added as a dataset). Within
    the same bucket, the newest file wins.
    """
    candidates: list[Path] = []

    def _add(path: Path) -> None:
        if path.is_file() and path.name.endswith(".pt"):
            candidates.append(path)
        elif path.is_dir():
            for name in (
                path / "checkpoints" / "latest_checkpoint.pt",
                path / "latest_checkpoint.pt",
            ):
                if name.is_file():
                    candidates.append(name)

    for raw in (resume_from, os.environ.get("RESUME_FROM"), os.environ.get("KAGGLE_RESUME_FROM")):
        if raw:
            _add(Path(str(raw)))

    roots: list[Path] = []
    if output_dir is not None:
        roots.append(Path(output_dir))
    for raw in extra_search or []:
        roots.append(Path(raw))
    roots.extend(
        [
            Path("/kaggle/working/router_large_kaggle"),
            Path("/kaggle/working"),
            Path("/kaggle/input"),
        ]
    )
    seen_roots: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve() if root.exists() else root
        except Exception:
            resolved = root
        if resolved in seen_roots or not root.exists():
            continue
        seen_roots.add(resolved)
        if root.is_file():
            _add(root)
            continue
        _add(root)
        # One-level children cover Kaggle input datasets and output folders.
        try:
            children = list(root.iterdir())
        except Exception:
            children = []
        for child in children:
            _add(child)
            if child.is_dir():
                try:
                    for grandchild in child.iterdir():
                        if grandchild.is_dir() and grandchild.name in {"checkpoints", "router_large_kaggle"}:
                            _add(grandchild)
                except Exception:
                    pass
        # /kaggle/input can be a few levels deeper (dataset/version/checkpoints).
        if "kaggle/input" in str(root).replace("\\", "/").lower():
            try:
                for hit in root.rglob("latest_checkpoint.pt"):
                    _add(hit)
            except Exception:
                pass

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        return None

    def _rank(path: Path) -> tuple[int, float]:
        text = str(path).replace("\\", "/").lower()
        bucket = 0 if "/kaggle/working/" in text else (1 if "/kaggle/input/" in text else 2)
        try:
            mtime = -path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (bucket, mtime)

    unique.sort(key=_rank)
    return unique[0]


def _zip_output_dir(output_dir: Path, zip_path: Path) -> None:
    import shutil
    import zipfile

    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith(".tmp"):
                continue
            # Don't nest a previous output zip inside the new one.
            if path.resolve() == zip_path.resolve() or path.suffix == ".zip":
                continue
            zf.write(path, arcname=str(path.relative_to(output_dir)))
    os.replace(tmp, zip_path)
    print(f"Updated output zip {zip_path} ({zip_path.stat().st_size} bytes)")


def train_large_router(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    output_dir: str | Path | None = None,
    model_name: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    grad_accum: int = 1,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    patience: int = 2,
    confidence_loss_weight: float = 0.3,
    temporal_loss_weight: float = 1.75,
    use_compile: bool = True,
    use_bf16: bool | None = None,
    seed: int = 42,
    freeze_layers: int = 0,
    gradient_checkpointing: bool = False,
    fit_temperature: bool = True,
    device: str | None = None,
    resume_from: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every_steps: int = 500,
    output_zip_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune multitask heads, including the confidence head, on a saved bundle.

    The encoder class is MultitaskQueryAnalyzer, so analyzer.pt loads with the
    existing service. The confidence term is applied outside forward() and does
    not change DistilBERT training.

    Resumable checkpoints (`latest_checkpoint.pt`) are written atomically at
    the end of every epoch and every `checkpoint_every_steps` optimizer steps.
    A Kaggle 12-hour timeout can therefore resume from the exact step. Best
    weights are written to `best_model.pt` and a provisional `analyzer.pt` as
    soon as validation macro-F1 improves. Test rows are never read here.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset
    from transformers import AutoTokenizer

    from .hf_utils import from_pretrained_local_first
    from .query_analyzer import (
        LABEL2ID,
        TEMPORAL_LABELS as model_labels,
        MultitaskQueryAnalyzer,
        QueryAnalyzerService,
        _freeze_encoder_layers,
        _inverse_freq_class_weights,
        _select_training_precision,
        _training_preflight,
    )

    if tuple(model_labels) != tuple(TEMPORAL_LABELS):
        raise RuntimeError(f"Label schema drift: {model_labels} vs {TEMPORAL_LABELS}")
    if not train_rows or not val_rows:
        raise RuntimeError("train_large_router needs non-empty train and val rows")

    _set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = str(device).startswith("cuda")
    grad_accum = max(1, int(grad_accum))
    checkpoint_every_steps = max(0, int(checkpoint_every_steps))
    if use_cuda and torch.cuda.device_count() > 1:
        print(
            f"NOTE: {torch.cuda.device_count()} CUDA devices are visible; "
            f"training stays on a single device ({device}). DataParallel is not used."
        )

    output_dir = Path(output_dir) if output_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    elif output_dir is not None:
        checkpoint_dir = output_dir / "checkpoints"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_zip_path = Path(output_zip_path) if output_zip_path is not None else None

    tokenizer = from_pretrained_local_first(AutoTokenizer.from_pretrained, model_name)
    model = MultitaskQueryAnalyzer(
        model_name=model_name,
        label_smoothing=0.0,
        temporal_loss_weight=temporal_loss_weight,
    ).to(device)
    if gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        if use_compile:
            print("torch.compile disabled because gradient checkpointing is on")
            use_compile = False
    _freeze_encoder_layers(model.encoder, freeze_layers)
    model.float()

    nec_counts: dict[int, int] = Counter(int(row["needs_retrieval"]) for row in train_rows)
    temp_counts: dict[int, int] = Counter(LABEL2ID[row["temporal_class"]] for row in train_rows)
    nec_w = _inverse_freq_class_weights(dict(nec_counts), 2).to(device)
    temp_w = _inverse_freq_class_weights(dict(temp_counts), len(TEMPORAL_LABELS)).to(device)
    print(f"Class weights necessity={nec_w.detach().cpu().tolist()} temporal={temp_w.detach().cpu().tolist()}")

    class _DS(Dataset):
        def __init__(self, split_rows: list[dict[str, Any]]) -> None:
            self.rows = split_rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            text, needs, temporal, confidence = _row_text(self.rows[idx])
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "needs": torch.tensor(needs, dtype=torch.long),
                "temporal": torch.tensor(LABEL2ID[temporal], dtype=torch.long),
                "label_confidence": torch.tensor(confidence, dtype=torch.float),
            }

    train_ds = _DS(train_rows)

    def make_loader(perm: list[int], start_index: int = 0) -> DataLoader:
        remaining = perm[start_index:]
        if not remaining:
            raise RuntimeError("make_loader received an empty remainder; skip this epoch instead")
        return DataLoader(
            Subset(train_ds, remaining),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_cuda,
            num_workers=0,
        )

    def _load_pickle(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    amp_mode, amp_dtype = _select_training_precision(use_cuda)
    if use_bf16 is False:
        amp_mode, amp_dtype = "fp32", None
    elif use_bf16 is True and use_cuda and torch.cuda.is_bf16_supported():
        amp_mode, amp_dtype = "bf16-autocast", torch.bfloat16

    compiled = model
    compile_ok = False
    if use_compile and use_cuda:
        try:
            compiled = torch.compile(model)
            compile_ok = True
        except Exception as exc:
            print(f"torch.compile skipped: {type(exc).__name__}: {exc}")
            compiled = model

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-6,
    )
    batches_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
    total_updates = updates_per_epoch * max(1, epochs)
    warmup_steps = int(warmup_ratio * total_updates)
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    except Exception:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / max(warmup_steps, 1))
            * 0.5 * (1 + math.cos(math.pi * min(step, total_updates) / max(total_updates, 1))),
        )

    def confidence_loss(out, batch, dev: str) -> torch.Tensor:
        needs = batch["needs"].to(dev)
        temporal = batch["temporal"].to(dev)
        quality = batch["label_confidence"].to(dev)
        nec_ok = (out["necessity_logits"].argmax(-1) == needs).float()
        temp_ok = (out["temporal_logits"].argmax(-1) == temporal).float()
        target = quality * (0.5 * nec_ok + 0.5 * temp_ok)
        return nn.functional.binary_cross_entropy(out["confidence"], target.detach())

    _training_preflight(model, tokenizer, device, max_length)
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    if compile_ok:
        try:
            probe = next(iter(make_loader(_epoch_permutation(len(train_ds), seed, 0), 0)))
            with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
                probed = compiled(
                    input_ids=probe["input_ids"].to(device),
                    attention_mask=probe["attention_mask"].to(device),
                    needs=probe["needs"].to(device),
                    temporal=probe["temporal"].to(device),
                    nec_class_weight=nec_w,
                    temp_class_weight=temp_w,
                )
                probe_loss = probed["loss"] + confidence_loss_weight * confidence_loss(probed, probe, device)
                probe_loss.backward()
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            print(f"torch.compile failed on the first step ({type(exc).__name__}: {exc}); using eager mode")
            compiled = model
            compile_ok = False
            optimizer.zero_grad(set_to_none=True)

    config = {
        "model_name": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_epochs": epochs,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "patience": patience,
        "confidence_loss_weight": confidence_loss_weight,
        "temporal_loss_weight": temporal_loss_weight,
        "freeze_layers": freeze_layers,
        "seed": seed,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    stale = 0
    start_epoch = 1
    resume_steps_in_epoch = 0
    resume_perm: list[int] | None = None
    resume_loss_sum = 0.0
    resume_seen = 0
    global_step = 0
    resume_path = find_resume_checkpoint(resume_from, output_dir=output_dir)
    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        ckpt = _load_pickle(resume_path)
        ckpt_cfg = ckpt.get("config") or {}
        for key in ("model_name", "max_length", "batch_size", "grad_accum", "seed"):
            if key in ckpt_cfg and ckpt_cfg[key] != config[key]:
                raise RuntimeError(
                    f"Checkpoint {key}={ckpt_cfg[key]!r} does not match this run ({config[key]!r}). "
                    "Resume with the same training config."
                )
        _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_score = float(ckpt.get("best_val_macro_f1", -1.0))
        stale = int(ckpt.get("epochs_without_improvement", 0))
        history = list(ckpt.get("history") or [])
        global_step = int(ckpt.get("global_step", 0))
        _restore_rng_states(ckpt.get("rng_states"))
        if ckpt.get("epoch_complete"):
            start_epoch = int(ckpt["epoch"]) + 1
            resume_steps_in_epoch = 0
            resume_perm = None
        else:
            start_epoch = int(ckpt["epoch"])
            resume_steps_in_epoch = int(ckpt.get("steps_in_epoch", 0))
            resume_perm = list(ckpt["epoch_perm"]) if ckpt.get("epoch_perm") is not None else None
            resume_loss_sum = float(ckpt.get("train_loss_sum", 0.0))
            resume_seen = int(ckpt.get("train_seen", 0))
        best_file = None
        if output_dir is not None and (output_dir / "best_model.pt").is_file():
            best_file = output_dir / "best_model.pt"
        elif resume_path.parent.parent.joinpath("best_model.pt").is_file():
            best_file = resume_path.parent.parent / "best_model.pt"
        elif resume_path.parent.joinpath("best_model.pt").is_file():
            best_file = resume_path.parent / "best_model.pt"
        if best_file is not None:
            payload = _load_pickle(best_file)
            best_state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            print(f"Restored best_model from {best_file} (val_macro_f1={best_score:.4f})")
        print(
            f"Resume state: next_epoch={start_epoch} steps_in_epoch={resume_steps_in_epoch} "
            f"global_step={global_step} best_val_macro_f1={best_score:.4f} stale={stale}"
        )
        if start_epoch > epochs and (ckpt.get("epoch_complete") or resume_steps_in_epoch >= batches_per_epoch):
            print("Checkpoint already finished every requested epoch; skipping to final export.")

    def raw_state_dict() -> dict[str, Any]:
        return {key: value.detach().cpu().clone() for key, value in _unwrap_model(model).state_dict().items()}

    def save_best_bundle(val_metrics: dict[str, Any], epoch: int) -> None:
        if output_dir is None:
            return
        state = raw_state_dict()
        _atomic_torch_save({"state_dict": state, "val_macro_f1": best_score, "epoch": epoch}, output_dir / "best_model.pt")
        tmp_dir = output_dir / "_analyzer_tmp"
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
        service = QueryAnalyzerService(
            model=_unwrap_model(model),
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )
        service.temperature = 1.0
        service.blend_confidence = True
        service.save(
            tmp_dir,
            extra_meta={
                "temperature": 1.0,
                "temperature_fitted": False,
                "blend_confidence": True,
                "confidence_loss_weight": confidence_loss_weight,
                "best_val_macro_f1": best_score,
                "history": _json_ready(history),
                "precision": amp_mode,
                "compiled": compile_ok,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "hyperparams": config,
                "val_metrics": _json_ready({k: v for k, v in val_metrics.items() if k not in {"temporal_logits", "temporal_labels"}}),
                "loader": "src.router_large.load_large_router",
            },
        )
        for name in ("analyzer.pt", "analyzer_meta.json"):
            os.replace(tmp_dir / name, output_dir / name)
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        _atomic_write_text(
            output_dir / "best_val_metrics.json",
            json.dumps({"epoch": epoch, "best_val_macro_f1": best_score, "metrics": _json_ready({k: v for k, v in val_metrics.items() if k not in {"temporal_logits", "temporal_labels"}})}, indent=2),
        )
        print(f"Saved improved best bundle to {output_dir} (val_macro_f1={best_score:.4f}, epoch={epoch})")

    def save_latest_checkpoint(
        *,
        epoch: int,
        steps_in_epoch: int,
        epoch_complete: bool,
        epoch_perm: list[int] | None,
        train_loss_sum: float,
        train_seen: int,
    ) -> None:
        if checkpoint_dir is None:
            return
        payload = {
            "model_state_dict": raw_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "epoch_complete": bool(epoch_complete),
            "steps_in_epoch": int(steps_in_epoch),
            "global_step": int(global_step),
            "best_val_macro_f1": best_score,
            "epochs_without_improvement": int(stale),
            "rng_states": _capture_rng_states(),
            "config": config,
            "history": _json_ready(history),
            "amp_mode": amp_mode,
            "epoch_perm": list(epoch_perm) if epoch_perm is not None else None,
            "train_loss_sum": float(train_loss_sum),
            "train_seen": int(train_seen),
        }
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        _atomic_torch_save(payload, latest_path)
        meta = {
            "epoch": epoch,
            "epoch_complete": bool(epoch_complete),
            "steps_in_epoch": int(steps_in_epoch),
            "global_step": int(global_step),
            "best_val_macro_f1": best_score,
            "epochs_without_improvement": int(stale),
            "path": str(latest_path),
        }
        _atomic_write_text(checkpoint_dir / "checkpoint_meta.json", json.dumps(meta, indent=2))
        print(
            f"Wrote resumable checkpoint {latest_path} "
            f"(epoch={epoch}{' complete' if epoch_complete else ''} step={steps_in_epoch} global={global_step})"
        )
        if epoch_complete and output_zip_path is not None and output_dir is not None:
            try:
                _zip_output_dir(output_dir, output_zip_path)
            except Exception as exc:
                print(f"WARN could not refresh output zip: {type(exc).__name__}: {exc}")

    print(
        f"Training {model_name} on {device} batch={batch_size} accum={grad_accum} "
        f"lr={lr} precision={amp_mode} compile={compile_ok} epochs<={epochs} "
        f"checkpoint_every_steps={checkpoint_every_steps} train_rows={len(train_rows)}"
    )
    started = time.time()
    if start_epoch > epochs:
        print("No remaining epochs to run.")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if resume_perm is not None and epoch == start_epoch and resume_steps_in_epoch > 0:
            perm = resume_perm
            already = resume_steps_in_epoch
            total_loss = resume_loss_sum
            seen = resume_seen
            resume_perm = None
            resume_steps_in_epoch = 0
            resume_loss_sum = 0.0
            resume_seen = 0
            print(f"Continuing epoch {epoch} from batch {already + 1}/{batches_per_epoch}")
        else:
            perm = _epoch_permutation(len(train_ds), seed, epoch)
            already = 0
            total_loss = 0.0
            seen = 0
        start_index = already * batch_size
        if start_index >= len(perm):
            train_loader = []
        else:
            train_loader = make_loader(perm, start_index)
        optimizer.zero_grad(set_to_none=True)
        iterator = train_loader
        if _tqdm is not None and train_loader:
            iterator = _tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", initial=already, total=batches_per_epoch, leave=False)
        epoch_batch = already
        for batch in iterator:
            epoch_batch += 1
            autocast_on = amp_dtype is not None
            with torch.amp.autocast("cuda", enabled=autocast_on, dtype=amp_dtype or torch.float32):
                out = compiled(
                    input_ids=batch["input_ids"].to(device, non_blocking=use_cuda),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=use_cuda),
                    needs=batch["needs"].to(device, non_blocking=use_cuda),
                    temporal=batch["temporal"].to(device, non_blocking=use_cuda),
                    nec_class_weight=nec_w,
                    temp_class_weight=temp_w,
                )
                loss = out["loss"] + confidence_loss_weight * confidence_loss(out, batch, device)
                loss = loss / grad_accum
            if not torch.isfinite(loss).item():
                optimizer.zero_grad(set_to_none=True)
                print(f"WARN non-finite loss at epoch {epoch} step {epoch_batch}; skipping update")
                continue
            loss.backward()
            total_loss += float(loss.detach().item()) * grad_accum
            seen += 1
            if epoch_batch % grad_accum == 0 or epoch_batch == batches_per_epoch:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if (
                    checkpoint_dir is not None
                    and checkpoint_every_steps > 0
                    and global_step % checkpoint_every_steps == 0
                ):
                    elapsed = max(time.time() - started, 1e-6)
                    steps_done = max(global_step, 1)
                    # Remaining updates over the full schedule, not just this epoch.
                    remaining = max(total_updates - global_step, 0)
                    eta_s = remaining * (elapsed / steps_done)
                    print(
                        f"Throughput: {steps_done / elapsed:.3f} opt-steps/s "
                        f"elapsed={elapsed / 60:.1f}m remaining~{eta_s / 3600:.2f}h "
                        f"(global_step={global_step}/{total_updates})"
                    )
                    save_latest_checkpoint(
                        epoch=epoch,
                        steps_in_epoch=epoch_batch,
                        epoch_complete=False,
                        epoch_perm=perm,
                        train_loss_sum=total_loss,
                        train_seen=seen,
                    )
        metrics = evaluate_rows(model, tokenizer, val_rows, device, max_length, temperature=1.0, batch_size=batch_size)
        metrics.pop("temporal_logits", None)
        metrics.pop("temporal_labels", None)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(seen, 1)
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        metrics["global_step"] = global_step
        history.append({key: value for key, value in metrics.items() if key != "temporal_per_class"})
        score = float(metrics["val_macro_f1"])
        elapsed = time.time() - started
        print(
            f"Epoch {epoch}: loss={metrics['train_loss']:.4f} val_macro_f1={score:.4f} "
            f"nec_f1={metrics['necessity']['macro_f1']:.4f} temp_f1={metrics['temporal']['macro_f1']:.4f} "
            f"temp_ece={metrics['temporal_ece']:.4f} elapsed={elapsed / 60:.1f}m"
        )
        if score > best_score:
            best_score = score
            best_state = raw_state_dict()
            stale = 0
            save_best_bundle(metrics, epoch)
        else:
            stale += 1
        save_latest_checkpoint(
            epoch=epoch,
            steps_in_epoch=epoch_batch,
            epoch_complete=True,
            epoch_perm=perm,
            train_loss_sum=total_loss,
            train_seen=seen,
        )
        if patience > 0 and stale >= patience:
            print(f"Early stopping: no val macro-F1 improvement for {patience} epoch(s).")
            break

    if best_state is not None:
        _unwrap_model(model).load_state_dict(best_state)
        # Keep compiled wrapper, if any, in sync with the raw module.
        if compiled is not model:
            pass

    temperature = 1.0
    before = evaluate_rows(model, tokenizer, val_rows, device, max_length, temperature=1.0, batch_size=batch_size)
    logits = before.pop("temporal_logits", None)
    labels = before.pop("temporal_labels", None)
    if fit_temperature and logits is not None and labels is not None:
        temperature = _fit_temperature(logits, labels)
        print(f"Fitted temporal temperature {temperature:.1f} on val")
    after = evaluate_rows(model, tokenizer, val_rows, device, max_length, temperature=temperature, batch_size=batch_size)
    after.pop("temporal_logits", None)
    after.pop("temporal_labels", None)

    raw_model = _unwrap_model(model)
    result = {
        "model_name": model_name,
        "best_val_macro_f1": best_score,
        "history": history,
        "temperature": temperature,
        "precision": amp_mode,
        "compiled": compile_ok,
        "epochs_ran": history[-1]["epoch"] if history else 0,
        "global_step": global_step,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "val_before_temperature": {k: v for k, v in before.items() if k not in {"temporal_logits", "temporal_labels"}},
        "val_after_temperature": after,
        "class_counts": {"necessity": dict(nec_counts), "temporal": {model_labels[k]: v for k, v in temp_counts.items()}},
        "hyperparams": config,
    }
    if output_dir is not None:
        tmp_dir = output_dir / "_analyzer_tmp"
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
        service = QueryAnalyzerService(
            model=raw_model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )
        service.temperature = temperature
        service.blend_confidence = True
        service.save(
            tmp_dir,
            extra_meta={
                "temperature": temperature,
                "temperature_fitted": bool(fit_temperature),
                "blend_confidence": True,
                "confidence_loss_weight": confidence_loss_weight,
                "best_val_macro_f1": best_score,
                "history": _json_ready(history),
                "precision": amp_mode,
                "compiled": compile_ok,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "hyperparams": result["hyperparams"],
                "val_after_temperature": _json_ready(after),
                "loader": "src.router_large.load_large_router",
            },
        )
        for name in ("analyzer.pt", "analyzer_meta.json"):
            os.replace(tmp_dir / name, output_dir / name)
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
        if best_state is not None:
            _atomic_torch_save(
                {"state_dict": best_state, "val_macro_f1": best_score, "temperature": temperature},
                output_dir / "best_model.pt",
            )
        result["output_dir"] = str(output_dir)
        print(f"Saved analyzer bundle to {output_dir}")
        if output_zip_path is not None:
            try:
                _zip_output_dir(output_dir, output_zip_path)
            except Exception as exc:
                print(f"WARN could not write final output zip: {type(exc).__name__}: {exc}")
    return result




def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_ready(value.item())
        except Exception:
            return str(value)
    return value


def sweep_hyperparams(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    model_name: str = "microsoft/deberta-v3-base",
    n_trials: int = 6,
    use_optuna: bool = True,
    batch_choices: tuple[int, ...] = (16, 32),
    lr_choices: tuple[float, ...] = (5e-6, 1e-5, 2e-5),
    epoch_choices: tuple[int, ...] = (3, 5),
    grad_accum: int = 1,
    max_length: int = 128,
    patience: int = 1,
    seed: int = 42,
    gradient_checkpointing: bool = False,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """Small Optuna search, or a fixed 6-point grid if Optuna is unavailable.

    Each trial trains from the pretrained checkpoint. The caller retrains once
    with the winning hyperparameters. Nothing is written to the DistilBERT
    artifact directory.
    """
    grid = [
        {"lr": 1e-5, "batch_size": batch_choices[-1], "max_epochs": 4},
        {"lr": 2e-5, "batch_size": batch_choices[-1], "max_epochs": epoch_choices[0]},
        {"lr": 5e-6, "batch_size": batch_choices[0], "max_epochs": epoch_choices[-1]},
        {"lr": 1e-5, "batch_size": batch_choices[0], "max_epochs": 4},
        {"lr": lr_choices[-1], "batch_size": batch_choices[0], "max_epochs": epoch_choices[0]},
        {"lr": lr_choices[0], "batch_size": batch_choices[-1], "max_epochs": epoch_choices[-1]},
    ]

    def run_one(params: dict[str, Any]) -> dict[str, Any]:
        import torch
        try:
            trained = train_large_router(
                train_rows,
                val_rows,
                output_dir=None,
                model_name=model_name,
                epochs=int(params["max_epochs"]),
                batch_size=int(params["batch_size"]),
                lr=float(params["lr"]),
                max_length=max_length,
                grad_accum=grad_accum,
                patience=patience,
                fit_temperature=False,
                seed=seed,
                gradient_checkpointing=gradient_checkpointing,
                **train_kwargs,
            )
            record = {
                "lr": float(params["lr"]),
                "batch_size": int(params["batch_size"]),
                "max_epochs": int(params["max_epochs"]),
                "best_val_macro_f1": trained["best_val_macro_f1"],
                "epochs_ran": trained["epochs_ran"],
                "precision": trained["precision"],
            }
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            record = {
                "lr": float(params["lr"]),
                "batch_size": int(params["batch_size"]),
                "max_epochs": int(params["max_epochs"]),
                "error": str(exc),
                "best_val_macro_f1": None,
            }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Sweep trial", record)
        return record

    trials: list[dict[str, Any]] = []
    backend = "grid"
    if use_optuna:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: "optuna.Trial") -> float:
                params = {
                    "lr": trial.suggest_categorical("lr", list(lr_choices)),
                    "batch_size": trial.suggest_categorical("batch_size", list(batch_choices)),
                    "max_epochs": trial.suggest_categorical("max_epochs", list(epoch_choices)),
                }
                record = run_one(params)
                trials.append(record)
                if record.get("best_val_macro_f1") is None:
                    return -1.0
                return float(record["best_val_macro_f1"])

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))
            backend = "optuna"
        except Exception as exc:
            print(f"Optuna unavailable ({type(exc).__name__}: {exc}). Using the fixed grid.")
            trials = []
    if backend != "optuna":
        for params in grid[:n_trials]:
            trials.append(run_one(params))
    ranked = [trial for trial in trials if isinstance(trial.get("best_val_macro_f1"), float)]
    best = max(ranked, key=lambda trial: trial["best_val_macro_f1"]) if ranked else None
    return {"backend": backend, "trials": trials, "best": best}


def verify_saved_leakage(stats: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the forbidden index stored in stats.json and scan rows."""
    index = LeakageIndex.from_dict(stats.get("leakage") or {})
    return scan_leakage(rows, index)
