"""Build a >=50k leak-free router dataset from real QA sources.

Writes data/router_large/{train,val,test}.jsonl and stats.json.
Does not modify data/queries.jsonl, data/heldout, data/general, or artifacts/analyzer*.

Labels match src/query_analyzer.py: needs_retrieval and temporal_class
in {current, historical, timeless}. Synthetic rows are capped at 20%.
Train, val, and test are checked for exact, entity, and template overlap
with data/queries.jsonl test, data/heldout test, and the TimeQA 500 test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.data_schema import read_jsonl, write_jsonl
from src.router_large import (
    TEMPORAL_LABELS,
    LeakageIndex,
    assign_splits,
    canonicalize,
    extract_timeqa_subjects,
    label_question,
    scan_leakage,
    select_balanced_rows,
    self_test,
    wiki_title,
)

CACHE = Path.home() / ".cache" / "adaptive-temporal-rag" / "router_large"
SITUATED_URLS = {
    "train": "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/temp.train.jsonl",
    "validation": "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/temp.dev.jsonl",
    "test": "https://raw.githubusercontent.com/mikejqzhang/SituatedQA/master/data/qa_data/temp.test.jsonl",
}
TIMEQUESTIONS_URLS = (
    "https://raw.githubusercontent.com/zhenjia2017/EXAQT/master/data/TimeQuestions_train.json",
    "https://raw.githubusercontent.com/zhenjia2017/EXAQT/master/data/timequestions.json",
    "https://github.com/semantic-systems/KGQA-datasets/raw/master/timequestions/timequestions.json",
)
TEMPQUESTIONS_URLS = (
    "https://raw.githubusercontent.com/zhenjia2017/TEMPQUESTIONS/master/TempQuestions.json",
    "https://raw.githubusercontent.com/zhenjia2017/TEMPQUESTIONS/master/data/TempQuestions.json",
)
HF_QUESTION_IDS = (
    ("timequestions", ("AmazonScience/timequestions", "timequestions", "lmqg/qg_timequestions")),
    ("tempquestions", ("tempquestions", "AmazonScience/tempquestions")),
    ("freshqa", ("freshllms/FreshQA", "yentinglin/freshqa")),
    ("realtimeqa", ("realtimeqa/realtimeqa", "realtimeqa")),
    ("mintaka", ("AmazonScience/mintaka",)),
    ("tot", ("baharef/ToT",)),
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "value", "normalized_value", "aliases", "answer"):
            if key in value:
                found = _as_text(value[key])
                if found:
                    return found
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _as_text(item)
            if found:
                return found
        return ""
    return str(value).strip()


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-temporal-rag"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        dest.write_bytes(response.read())
    return dest


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_of(row: dict[str, Any]) -> str:
    for key in ("question", "Question", "query", "edited_question", "question_stem", "text"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    return ""


def _cap(rows: list[dict[str, Any]], limit: int, seed: int) -> tuple[list[dict[str, Any]], bool]:
    if limit <= 0 or len(rows) <= limit:
        return rows, False
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:limit], True


def make_labeled(
    question: str,
    source: str,
    *,
    gold: str = "",
    entity: str = "",
    group_id: str = "",
    date_type: str | None = None,
    is_synthetic: bool = False,
    reference_year: int = 2026,
    authored: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    text = " ".join((question or "").split())
    if len(text) < 12 or len(text) > 400 or len(text.split()) < 3:
        return None
    if sum(character.isalpha() for character in text) < 8:
        return None
    if authored is not None:
        temporal = authored.get("temporal_class")
        needs = int(authored.get("needs_retrieval", -1))
        if temporal not in TEMPORAL_LABELS or needs not in (0, 1):
            return None
        labeled = {
            "needs_retrieval": needs,
            "temporal_class": temporal,
            "time_hint": authored.get("time_hint"),
            "label_confidence": 0.95,
        }
    else:
        labeled = label_question(text, source, date_type=date_type, reference_year=reference_year)
        if labeled is None:
            return None
    gold_text = (gold or "")[:180]
    return {
        "query": text,
        "source": source,
        "gold_answer": gold_text,
        "gold_answers": [gold_text] if gold_text else [],
        "gold_doc_ids": [],
        "entity": entity or "",
        "group_id": group_id or entity or "",
        "is_synthetic": bool(is_synthetic),
        "is_paraphrase": False,
        "seed_query_id": None,
        "unanswerable": False,
        **labeled,
    }


def _record_failure(failed: list[dict[str, str]], source: str, exc: BaseException) -> None:
    message = f"{type(exc).__name__}: {exc}"
    failed.append({"source": source, "error": message})
    print(f"FAILED {source}: {message}", flush=True)


def load_timeqa(failed: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return train/validation question rows, official test questions, and test wiki titles."""
    pool: list[dict[str, Any]] = []
    test_questions: list[str] = []
    test_titles: list[str] = []
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq

        files = list_repo_files("hugosousa/TimeQA", repo_type="dataset", revision="refs/convert/parquet")
        parquet = [name for name in files if name.endswith(".parquet")]
        if not parquet:
            raise RuntimeError("no parquet files in hugosousa/TimeQA refs/convert/parquet")
        by_split: dict[str, list[str]] = defaultdict(list)
        for name in parquet:
            parts = name.split("/")
            split = parts[-2] if len(parts) >= 2 else "train"
            by_split[split].append(name)
        for split, names in sorted(by_split.items()):
            for name in names:
                path = hf_hub_download(
                    "hugosousa/TimeQA",
                    name,
                    repo_type="dataset",
                    revision="refs/convert/parquet",
                )
                table = pq.read_table(path, columns=["question", "idx", "targets"])
                data = table.to_pydict()
                questions = data.get("question") or []
                idxs = data.get("idx") or [""] * len(questions)
                targets = data.get("targets") or [None] * len(questions)
                for question, idx, target in zip(questions, idxs, targets):
                    title = wiki_title(str(idx or ""))
                    if split == "test":
                        test_questions.append(str(question or ""))
                        test_titles.append(title)
                        continue
                    row = make_labeled(
                        str(question or ""),
                        "timeqa",
                        gold=_as_text(target),
                        entity=title,
                        group_id=title,
                    )
                    if row:
                        pool.append(row)
        if not pool:
            raise RuntimeError("TimeQA parquet load produced no train/validation rows")
        print(f"loaded timeqa rows={len(pool)} official_test_questions={len(test_questions)}", flush=True)
        return pool, test_questions, test_titles
    except Exception as exc:
        _record_failure(failed, "timeqa", exc)
        traceback.print_exc()
        local = ROOT / "data" / "general" / "timeqa" / "queries.jsonl"
        if not local.exists():
            return [], [], []
        print("falling back to local TimeQA queries, excluding split=test", flush=True)
        for raw in read_jsonl(local):
            if raw.get("split") == "test":
                continue
            subject = extract_timeqa_subjects(str(raw.get("query") or ""))
            entity = subject[0] if subject else ""
            row = make_labeled(
                str(raw.get("query") or ""),
                "timeqa_local",
                gold=str(raw.get("gold_answer") or ""),
                entity=entity,
                group_id=entity,
                authored=raw,
            )
            if row:
                pool.append(row)
        print(f"loaded timeqa_local rows={len(pool)}", flush=True)
        return pool, [], []


def load_situatedqa(failed: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        cache = CACHE / "situatedqa"
        for split, url in SITUATED_URLS.items():
            path = _download(url, cache / f"temp.{split}.jsonl")
            for raw in _read_json_lines(path):
                question = str(raw.get("edited_question") or raw.get("question") or "").strip()
                group = str(raw.get("id") or "")
                row = make_labeled(
                    question,
                    "situatedqa",
                    gold=_as_text(raw.get("answer")),
                    entity="",
                    group_id=group or canonicalize(question),
                    date_type=str(raw.get("date_type") or ""),
                )
                if row:
                    rows.append(row)
        if len(rows) < 100:
            raise RuntimeError(f"SituatedQA download returned only {len(rows)} labeled rows")
        print(f"loaded situatedqa rows={len(rows)}", flush=True)
        return rows
    except Exception as exc:
        _record_failure(failed, "situatedqa", exc)
        local = ROOT / "data" / "general" / "situatedqa" / "queries.jsonl"
        if not local.exists():
            return []
        for raw in read_jsonl(local):
            row = make_labeled(
                str(raw.get("query") or ""),
                "situatedqa_local",
                gold=str(raw.get("gold_answer") or ""),
                group_id=str(raw.get("query_id") or ""),
                authored=raw,
            )
            if row:
                rows.append(row)
        print(f"loaded situatedqa_local rows={len(rows)}", flush=True)
        return rows


def _hf_dataset(path: str, config: str | None, split: str):
    from datasets import load_dataset

    if config:
        return load_dataset(path, config, split=split)
    return load_dataset(path, split=split)


def load_hf_questions(
    source: str,
    path: str,
    *,
    config: str | None = None,
    splits: tuple[str, ...] = ("train", "validation"),
    limit: int = 0,
    seed: int = 42,
    question_keys: tuple[str, ...] = ("question", "question_stem", "Question"),
    stream: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for split in splits:
        try:
            if stream and limit:
                from datasets import load_dataset
                if config:
                    dataset = load_dataset(path, config, split=split, streaming=True)
                else:
                    dataset = load_dataset(path, split=split, streaming=True)
            else:
                dataset = _hf_dataset(path, config, split)
        except Exception as exc:
            last_error = exc
            continue
        produced = 0
        for raw in dataset:
            question = ""
            for key in question_keys:
                if key in raw and isinstance(raw[key], str) and raw[key].strip():
                    question = raw[key].strip()
                    break
            if not question:
                question = _question_of(raw)
            gold = ""
            for key in ("answer", "answers", "choices"):
                if key in raw:
                    gold = _as_text(raw[key])
                    if gold:
                        break
            row = make_labeled(question, source, gold=gold, group_id=canonicalize(question))
            if row:
                rows.append(row)
            produced += 1
            if stream and limit and produced >= limit:
                break
    if not rows and last_error is not None:
        raise last_error
    capped, was_capped = _cap(rows, 0 if stream else limit, seed)
    print(
        f"loaded {source} via {path} config={config} rows={len(capped)} "
        f"capped={was_capped} streamed={bool(stream and limit)}",
        flush=True,
    )
    return capped


def load_optional_hf(source: str, ids: tuple[str, ...], limit: int, seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    for dataset_id in ids:
        try:
            return load_hf_questions(source, dataset_id, limit=limit, seed=seed, stream=True), errors
        except Exception as exc:
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
    return [], errors


def load_json_urls(source: str, urls: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    for url in urls:
        try:
            dest = CACHE / source / Path(url).name
            path = _download(url, dest, timeout=25)
            payload: Any
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".jsonl":
                raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    for key in ("data", "questions", "Questions", "examples"):
                        if isinstance(payload.get(key), list):
                            payload = payload[key]
                            break
                raw_rows = payload if isinstance(payload, list) else []
            rows = []
            for raw in raw_rows:
                if isinstance(raw, str):
                    question = raw
                    gold = ""
                elif isinstance(raw, dict):
                    question = _question_of(raw)
                    gold = _as_text(raw.get("answer") or raw.get("Answer") or raw.get("answers"))
                else:
                    continue
                row = make_labeled(question, source, gold=gold, group_id=canonicalize(question))
                if row:
                    rows.append(row)
            if rows:
                print(f"loaded {source} from {url} rows={len(rows)}", flush=True)
                return rows, errors
            errors.append(f"{url}: no question rows")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    return [], errors


def load_arc(failed: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for config in ("ARC-Easy", "ARC-Challenge"):
            for split in ("train", "validation", "test"):
                try:
                    part = load_hf_questions("arc", "allenai/ai2_arc", config=config, splits=(split,), question_keys=("question",))
                except Exception as exc:
                    print(f"arc {config} {split} skipped: {type(exc).__name__}: {exc}", flush=True)
                    continue
                rows.extend(part)
        if not rows:
            raise RuntimeError("ARC produced no rows")
        print(f"loaded arc rows={len(rows)}", flush=True)
        return rows
    except Exception as exc:
        _record_failure(failed, "arc", exc)
        return []


def load_synthetic(reference_year: int) -> list[dict[str, Any]]:
    del reference_year
    rows: list[dict[str, Any]] = []
    specs = (
        (ROOT / "data" / "queries.jsonl", "synthetic_queries"),
        (ROOT / "data" / "heldout" / "queries.jsonl", "synthetic_heldout"),
    )
    for path, source in specs:
        if not path.exists():
            continue
        for raw in read_jsonl(path):
            if raw.get("split") == "test" or raw.get("is_paraphrase"):
                continue
            row = make_labeled(
                str(raw.get("query") or ""),
                source,
                gold=str(raw.get("gold_answer") or ""),
                group_id=str(raw.get("seed_query_id") or raw.get("query_id") or ""),
                is_synthetic=True,
                authored=raw,
            )
            if row:
                rows.append(row)
    print(f"loaded synthetic non-test non-paraphrase rows={len(rows)}", flush=True)
    return rows


def build_forbidden_index() -> LeakageIndex:
    from src.benchmarks.heldout import entity_in_query, entity_names
    import build_dataset as dataset

    names = list(entity_names())
    names.extend(title for title, _topic, _text in dataset.TIMELESS_FACTS)
    index = LeakageIndex()
    specs = (
        (ROOT / "data" / "queries.jsonl", False),
        (ROOT / "data" / "heldout" / "queries.jsonl", False),
        (ROOT / "data" / "general" / "timeqa" / "queries.jsonl", True),
    )
    questions: list[str] = []
    entities: list[str] = []
    heldout_file = ROOT / "data" / "heldout" / "heldout_entities.txt"
    if heldout_file.exists():
        entities.extend(line.strip() for line in heldout_file.read_text(encoding="utf-8").splitlines() if line.strip())
    for path, timeqa in specs:
        if not path.exists():
            raise FileNotFoundError(path)
        for raw in read_jsonl(path):
            if raw.get("split") != "test":
                continue
            question = str(raw.get("query") or "")
            questions.append(question)
            found = entity_in_query(question, names)
            if found:
                entities.append(found)
            if timeqa:
                entities.extend(extract_timeqa_subjects(question))
    index.add_entities(entities)
    index.add_exact(questions)
    index.add_question_templates(questions)
    print(
        f"forbidden questions={len(index.exact)} entities={len(index.entities)} templates={len(index.templates)}",
        flush=True,
    )
    return index


def project_self_check(index: LeakageIndex) -> None:
    self_test()
    exact = index.reason("As of now, who is the Prime Minister of India?")
    if exact != "exact":
        # The surface form may differ; every stored test question must still leak.
        pass
    shortest = sorted((template.skeleton for template in index.templates), key=len)[:12]
    print(f"shortest forbidden templates: {shortest}", flush=True)
    leaks = 0
    for question in list(index.exact)[:50]:
        if index.reason(question) != "exact":
            raise AssertionError(f"forbidden question was not an exact leak: {question}")
        leaks += 1
    unrelated = index.reason("Which statement best explains why a sealed bottle collapses after cooling?")
    if unrelated is not None:
        raise AssertionError(f"unrelated question leaked as {unrelated}")
    print(f"self-check ok; sampled exact leaks={leaks}", flush=True)


def dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    source_rank = {
        "timeqa": 0,
        "situatedqa": 1,
        "timequestions": 2,
        "tempquestions": 3,
        "mintaka": 4,
        "freshqa": 5,
        "realtimeqa": 6,
        "tot": 7,
        "arc": 8,
        "sciq": 9,
        "openbookqa": 10,
        "qasc": 11,
        "commonsense_qa": 12,
        "squad_v2": 13,
        "nq_open": 14,
        "triviaqa": 15,
    }
    best: dict[str, dict[str, Any]] = {}
    dropped = 0
    for row in rows:
        key = canonicalize(row["query"])
        if not key:
            dropped += 1
            continue
        previous = best.get(key)
        rank = (
            1 if row.get("is_synthetic") else 0,
            -float(row.get("label_confidence", 0.0)),
            source_rank.get(row.get("source", ""), 40),
            len(row["query"]),
        )
        if previous is None:
            best[key] = row
            continue
        prev_rank = (
            1 if previous.get("is_synthetic") else 0,
            -float(previous.get("label_confidence", 0.0)),
            source_rank.get(previous.get("source", ""), 40),
            len(previous["query"]),
        )
        dropped += 1
        if rank < prev_rank:
            best[key] = row
    return list(best.values()), dropped


def assign_ids(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        digest = hashlib.md5(canonicalize(row["query"]).encode("utf-8")).hexdigest()[:12]
        row["query_id"] = f"rl_{digest}"


def counter_of(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def temporal_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in TEMPORAL_LABELS}
    for row in rows:
        label = row.get("temporal_class")
        if label in counts:
            counts[label] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=50000)
    parser.add_argument("--max-rows", type=int, default=75000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-year", type=int, default=2026)
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "router_large"))
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    index = build_forbidden_index()
    project_self_check(index)
    if args.self_check_only:
        print("self-check only; no download", flush=True)
        return

    failed: list[dict[str, str]] = []
    loaded_counts: dict[str, int] = {}
    pool: list[dict[str, Any]] = []

    def take(source: str, rows: list[dict[str, Any]]) -> None:
        loaded_counts[source] = len(rows)
        pool.extend(rows)

    timeqa_rows, official_test_questions, official_test_titles = load_timeqa(failed)
    take("timeqa" if timeqa_rows and timeqa_rows[0]["source"] == "timeqa" else "timeqa_local", timeqa_rows)
    local_exact = set(index.exact)
    titles_for_500 = [
        title for question, title in zip(official_test_questions, official_test_titles)
        if title and canonicalize(question) in local_exact
    ]
    if official_test_questions:
        added_exact = index.add_exact(official_test_questions)
        print(f"added TimeQA official test exact questions={added_exact}", flush=True)
    if titles_for_500:
        added_entities = index.add_entities(titles_for_500)
        print(f"added TimeQA 500 wiki titles={added_entities}", flush=True)

    take("situatedqa", load_situatedqa(failed))

    general_specs = (
        ("nq_open", "google-research-datasets/nq_open", None, ("train",), 0),
        ("triviaqa", "mandarjoshi/trivia_qa", "unfiltered.nocontext", ("train",), 50000),
        ("squad_v2", "rajpurkar/squad_v2", None, ("train",), 80000),
        ("commonsense_qa", "tau/commonsense_qa", None, ("train", "validation"), 0),
        ("openbookqa", "allenai/openbookqa", None, ("train", "validation", "test"), 0),
        ("sciq", "allenai/sciq", None, ("train", "validation", "test"), 0),
        ("qasc", "allenai/qasc", None, ("train", "validation"), 0),
    )
    for source, path, config, splits, limit in general_specs:
        try:
            rows = load_hf_questions(source, path, config=config, splits=splits, limit=limit, seed=args.seed)
            if source == "nq_open" and not rows:
                rows = load_hf_questions(source, "nq_open", splits=splits, limit=limit, seed=args.seed)
            take(source, rows)
        except Exception as exc:
            if source == "nq_open":
                try:
                    rows = load_hf_questions(source, "nq_open", splits=splits, limit=limit, seed=args.seed)
                    take(source, rows)
                    continue
                except Exception as second:
                    exc = second
            _record_failure(failed, source, exc)
            traceback.print_exc()
            take(source, [])

    take("arc", load_arc(failed))
    def take_fallback(source: str, urls: tuple[str, ...], hf_ids: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
        rows, url_errors = load_json_urls(source, urls)
        hf_errors: list[str] = []
        if not rows and hf_ids:
            rows, hf_errors = load_optional_hf(source, hf_ids, limit=limit, seed=args.seed)
        if not rows:
            _record_failure(failed, source, RuntimeError(" | ".join(url_errors + hf_errors) or "no rows"))
        return rows

    take("timequestions", take_fallback("timequestions", TIMEQUESTIONS_URLS, HF_QUESTION_IDS[0][1], 20000))
    take("tempquestions", take_fallback("tempquestions", TEMPQUESTIONS_URLS, HF_QUESTION_IDS[1][1], 5000))
    fresh_rows, fresh_errors = load_optional_hf("freshqa", HF_QUESTION_IDS[2][1], limit=5000, seed=args.seed)
    if not fresh_rows:
        _record_failure(failed, "freshqa", RuntimeError(" | ".join(fresh_errors) or "no rows"))
    take("freshqa", fresh_rows)
    realtime_rows, realtime_errors = load_optional_hf("realtimeqa", HF_QUESTION_IDS[3][1], limit=5000, seed=args.seed)
    if not realtime_rows:
        _record_failure(failed, "realtimeqa", RuntimeError(" | ".join(realtime_errors) or "no rows"))
    take("realtimeqa", realtime_rows)
    mintaka_rows, mintaka_errors = load_optional_hf("mintaka", HF_QUESTION_IDS[4][1], limit=20000, seed=args.seed)
    if not mintaka_rows:
        _record_failure(failed, "mintaka", RuntimeError(" | ".join(mintaka_errors) or "no rows"))
    take("mintaka", mintaka_rows)
    tot_rows, tot_errors = load_optional_hf("tot", HF_QUESTION_IDS[5][1], limit=8000, seed=args.seed)
    if not tot_rows:
        _record_failure(failed, "tot", RuntimeError(" | ".join(tot_errors) or "no rows"))
    for row in tot_rows:
        row["is_synthetic"] = True
        row["source"] = "tot"
    take("tot", tot_rows)

    try:
        from huggingface_hub import list_repo_files

        wiki_files = list_repo_files("seonghyeonye/TemporalWiki", repo_type="dataset")
        probe_files = [name for name in wiki_files if "probe" in name.lower() or "Probe" in name]
        full_wiki = [name for name in wiki_files if "Wikipedia_Full" in name or name.lower().endswith(".zip")]
        if not probe_files:
            raise RuntimeError(
                "TemporalWiki has no question split. Files are "
                + ", ".join(wiki_files[:12])
                + ". Probes are triples and wiki text is passages; they were not turned into template questions."
            )
        raise RuntimeError(
            "TemporalWiki probe files are not natural questions "
            f"({', '.join(probe_files[:8])}; other files include {', '.join(full_wiki[:4])}). "
            "They were not converted into templated questions."
        )
    except Exception as exc:
        _record_failure(failed, "temporalwiki", exc)
        take("temporalwiki", [])

    take("synthetic", load_synthetic(args.reference_year))

    labeled_by_source = counter_of(pool, "source")
    deduped, dedup_dropped = dedup(pool)
    leak_counts: Counter[str] = Counter()
    leak_templates: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for seen, row in enumerate(deduped, start=1):
        if seen % 20000 == 0:
            print(f"leak filter {seen}/{len(deduped)} kept={len(kept)}", flush=True)
        found = index.classify(row["query"], entity=row.get("entity") or None)
        if found is None:
            kept.append(row)
            continue
        leak_counts[found[0]] += 1
        if found[0] == "template":
            leak_templates[found[1]] += 1
    selected, balance_report = select_balanced_rows(
        kept,
        min_total=args.min_rows,
        max_total=args.max_rows,
        max_synthetic_fraction=0.20,
        seed=args.seed,
    )
    split_rows = assign_splits(selected, seed=args.seed)
    assign_ids(split_rows)
    residual = scan_leakage(split_rows, index)
    if residual["leaks"]:
        clean = []
        for row in split_rows:
            if index.reason(row["query"], entity=row.get("entity") or None) is None:
                clean.append(row)
        removed = len(split_rows) - len(clean)
        split_rows = clean
        residual = scan_leakage(split_rows, index)
        residual["second_pass_removed"] = removed
    if residual["leaks"]:
        print("ERROR residual leakage remains", residual, flush=True)

    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
    for row in split_rows:
        by_split[row["split"]].append(row)
    for name in by_split:
        by_split[name].sort(key=lambda row: row["query_id"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in by_split.items():
        write_jsonl(out / f"{name}.jsonl", rows)

    examples = []
    seen: Counter[tuple[str, str]] = Counter()
    for row in split_rows:
        key = (str(row["source"]), str(row["temporal_class"]))
        if seen[key] >= 2:
            continue
        seen[key] += 1
        examples.append(
            {
                "source": row["source"],
                "temporal_class": row["temporal_class"],
                "needs_retrieval": row["needs_retrieval"],
                "label_confidence": row["label_confidence"],
                "query": row["query"],
            }
        )
    synthetic_n = sum(1 for row in split_rows if row.get("is_synthetic"))
    total = len(split_rows)
    forbidden = index.to_dict()
    stats = {
        "reference_year": args.reference_year,
        "min_rows": args.min_rows,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "total": total,
        "by_split": {name: len(rows) for name, rows in by_split.items()},
        "by_source": counter_of(split_rows, "source"),
        "by_temporal_class": temporal_counts(split_rows),
        "by_needs_retrieval": counter_of(split_rows, "needs_retrieval"),
        "by_split_and_class": {
            name: temporal_counts(rows) for name, rows in by_split.items()
        },
        "by_split_and_needs": {
            name: counter_of(rows, "needs_retrieval") for name, rows in by_split.items()
        },
        "synthetic_rows": synthetic_n,
        "synthetic_fraction": (synthetic_n / total) if total else 0.0,
        "constraints": {
            "min_rows": total >= args.min_rows,
            "synthetic_cap": (synthetic_n / total) <= 0.20 + 1e-9 if total else False,
            "no_residual_leaks": residual["leaks"] == 0,
        },
        "funnel": {
            "loaded_by_source": loaded_counts,
            "labeled_by_source": labeled_by_source,
            "labeled_total": len(pool),
            "after_dedup": len(deduped),
            "dedup_dropped": dedup_dropped,
            "leak_dropped": dict(leak_counts),
            "top_dropped_templates": leak_templates.most_common(15),
            "after_leak_filter": len(kept),
            "after_leak_filter_by_class": temporal_counts(kept),
            "balance": balance_report,
        },
        "leakage": {
            "forbidden_question_count": len(forbidden["forbidden_questions"]),
            "forbidden_entity_count": len(forbidden["forbidden_entities"]),
            "forbidden_template_count": len(forbidden["forbidden_templates"]),
            "forbidden_questions": forbidden["forbidden_questions"],
            "forbidden_entities": forbidden["forbidden_entities"],
            "forbidden_templates": forbidden["forbidden_templates"],
            "timeqa_official_test_questions_added": len(official_test_questions),
            "timeqa_official_test_titles_seen": len(official_test_titles),
            "residual": residual,
        },
        "sources_failed": failed,
        "examples": examples,
        "labeling_policy": (
            "Explicit years map to historical when the latest year is before reference_year-1, "
            "else current. Explicit current cues, present-tense changing offices, SituatedQA "
            "original temporally dependent questions, and fresh/realtime sources map to current. "
            "TimeQA and other temporal sets default to historical. General QA without a time cue "
            "(NQ, TriviaQA, SQuAD v2, ARC, commonsense, OpenBookQA, SciQ, QASC) maps to timeless "
            "and needs_retrieval=0. Current and historical labels use needs_retrieval=1. "
            "Test of Time rows, when loaded, are marked synthetic and share the 20% cap."
        ),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print_balance(split_rows, synthetic_n)
    print(f"wrote {out}", flush=True)
    print("constraints", stats["constraints"], flush=True)
    print("failed sources", [item["source"] for item in failed], flush=True)
    if not stats["constraints"]["min_rows"] or not stats["constraints"]["synthetic_cap"] or not stats["constraints"]["no_residual_leaks"]:
        raise SystemExit(2)


def print_balance(rows: list[dict[str, Any]], synthetic_n: int) -> None:
    print(f"TOTAL {len(rows)}", flush=True)
    print(f"synthetic {synthetic_n} fraction {(synthetic_n / len(rows)) if rows else 0:.4f}", flush=True)
    print("by_source", counter_of(rows, "source"), flush=True)
    print("by_temporal_class", temporal_counts(rows), flush=True)
    print("by_needs_retrieval", counter_of(rows, "needs_retrieval"), flush=True)
    for split in ("train", "val", "test"):
        part = [row for row in rows if row["split"] == split]
        print(f"{split} {len(part)} class {temporal_counts(part)} needs {counter_of(part, 'needs_retrieval')}", flush=True)


if __name__ == "__main__":
    main()
