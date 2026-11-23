"""Prepare TimeQA, SituatedQA, and an entity-held-out synthetic split.

Writes under data/general and data/heldout. Does not modify data/corpus.jsonl or data/queries.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.heldout import build_heldout
from src.benchmarks.situatedqa import load_situatedqa_rows, situatedqa_queries
from src.benchmarks.timeqa import build_timeqa, load_timeqa_rows, sample_timeqa
from src.data_schema import write_jsonl


def _write_pair(directory: Path, docs, queries) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if docs is not None:
        write_jsonl(directory / "corpus.jsonl", docs)
    write_jsonl(directory / "queries.jsonl", queries)
    print(f"{directory}: docs={0 if docs is None else len(docs)} queries={len(queries)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeqa-test", type=int, default=500)
    parser.add_argument("--timeqa-train", type=int, default=400)
    parser.add_argument("--situated-train", type=int, default=1200)
    parser.add_argument("--situated-dev", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(ROOT / "data"))
    args = parser.parse_args()
    out = Path(args.out)

    print("Downloading TimeQA rows...")
    timeqa_dir = out / "general" / "timeqa"
    if (timeqa_dir / "queries.jsonl").exists() and (timeqa_dir / "corpus.jsonl").exists():
        print(f"Reusing {timeqa_dir}")
    else:
        test_rows = sample_timeqa(load_timeqa_rows("test", max(args.timeqa_test, 800)), args.timeqa_test, args.seed)
        train_rows = load_timeqa_rows("train", args.timeqa_train)
        test_docs, test_queries = build_timeqa(test_rows, "test")
        train_docs, train_queries = build_timeqa(train_rows, "train")
        by_id = {doc.doc_id: doc for doc in train_docs}
        by_id.update({doc.doc_id: doc for doc in test_docs})
        _write_pair(timeqa_dir, list(by_id.values()), train_queries + test_queries)

    print("Downloading SituatedQA temporal rows...")
    dev_rows = load_situatedqa_rows("dev", args.situated_dev)
    train_s = situatedqa_queries(load_situatedqa_rows("train", args.situated_train), "train")
    dev_s = situatedqa_queries(dev_rows, "val")
    _write_pair(out / "general" / "situatedqa", None, train_s + dev_s)

    print("Building entity-held-out synthetic split...")
    docs, queries, held = build_heldout(seed=args.seed)
    held_dir = out / "heldout"
    _write_pair(held_dir, docs, queries)
    (held_dir / "heldout_entities.txt").write_text("\n".join(sorted(held)) + "\n", encoding="utf-8")
    test_n = sum(1 for query in queries if query.split == "test")
    print(f"held-out entities={len(held)} test queries={test_n}")


if __name__ == "__main__":
    main()
