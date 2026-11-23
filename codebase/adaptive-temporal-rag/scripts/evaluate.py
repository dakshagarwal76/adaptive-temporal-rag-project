"""Evaluate B1/B2/B3/Adaptive on the held-out test split."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_loader import load_dotenv

load_dotenv(ROOT / ".env")

from src.data_schema import load_queries
from src.embed_index import EmbeddingIndex
from src.generate import Generator
from src.metrics import (
    abstention_correct,
    aggregate_rows,
    best_contains,
    best_exact_match,
    best_token_f1,
    gold_strings,
    precision_at_k,
    recall_at_k,
    temporal_correctness,
)
from src.pipelines import build_all_pipelines
from src.query_analyzer import load_query_analyzer


def _score_query(result, query, dataset: str) -> dict:
    ret_ids = [doc.doc_id for doc in result.retrieved_docs]
    golds = gold_strings(query)
    unanswerable = bool(getattr(query, "unanswerable", False)) or not golds
    if unanswerable:
        em = f1 = contains = float(abstention_correct(result.answer, True))
    else:
        em = best_exact_match(result.answer, golds)
        f1 = best_token_f1(result.answer, golds)
        contains = best_contains(result.answer, golds)
    row = {
        "system": result.system,
        "query_id": query.query_id,
        "query": query.query,
        "temporal_class": query.temporal_class,
        "needs_retrieval": query.needs_retrieval,
        "gold_answer": query.gold_answer,
        "pred_answer": result.answer,
        "em": em,
        "token_f1": f1,
        "contains": contains,
        "precision_at_k": precision_at_k(ret_ids, query.gold_doc_ids, k=5),
        "recall_at_k": recall_at_k(ret_ids, query.gold_doc_ids, k=5),
        "abstain_acc": abstention_correct(result.answer, unanswerable),
        "latency_s": result.latency_s,
        "did_retrieve": float(result.did_retrieve),
        "backend": result.backend,
        "policy": result.meta.get("policy") or result.meta.get("fixed_mode") or "",
    }
    if dataset != "timeqa":
        row["temporal_acc"] = temporal_correctness(result.retrieved_docs, query)
    if result.analyzer is not None:
        row["pred_needs"] = result.analyzer.needs_retrieval
        row["pred_temporal"] = result.analyzer.temporal_class
        row["pred_confidence"] = result.analyzer.confidence
        row["router_temporal_acc"] = float(result.analyzer.temporal_class == query.temporal_class)
        row["router_needs_acc"] = float(result.analyzer.needs_retrieval == query.needs_retrieval)
    return row


def _index_for(dataset: str, artifacts: Path) -> tuple[EmbeddingIndex, Path]:
    if dataset == "synthetic":
        index = EmbeddingIndex()
        index.load(artifacts)
        return index, artifacts
    if dataset == "heldout":
        corpus = ROOT / "data" / "heldout" / "corpus.jsonl"
        index_dir = artifacts / "heldout_index"
    elif dataset == "timeqa":
        corpus = ROOT / "data" / "general" / "timeqa" / "corpus.jsonl"
        index_dir = artifacts / "timeqa_index"
    else:
        raise SystemExit(f"Unknown dataset {dataset}")
    index = EmbeddingIndex()
    if (index_dir / "faiss.index").exists():
        index.load(index_dir)
        return index, index_dir
    from src.embed_index import build_index_from_corpus

    print(f"Building index for {dataset} from {corpus}")
    built = build_index_from_corpus(corpus, index_dir)
    return built, index_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=str, default="")
    p.add_argument("--artifacts", type=str, default=str(ROOT / "artifacts"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset", type=str, default="", choices=["", "synthetic", "heldout", "timeqa"])
    p.add_argument("--systems", type=str, default="", help="Comma-separated system names. Default: the four baselines.")
    p.add_argument("--include-learned", action="store_true")
    p.add_argument("--ablation", type=str, default="", choices=["", "ranker_off", "reader_off"])
    import os as _os

    _default_gen = _os.environ.get("GENERATOR_BACKEND", "extractive")
    if _default_gen not in ("extractive", "hf", "api"):
        _default_gen = "extractive"
    p.add_argument("--generator", type=str, default=_default_gen, choices=["extractive", "hf", "api"])
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    artifacts = Path(args.artifacts)
    dataset = args.dataset or "synthetic"
    if args.queries:
        queries_path = Path(args.queries)
    elif dataset == "heldout":
        queries_path = ROOT / "data" / "heldout" / "queries.jsonl"
    elif dataset == "timeqa":
        queries_path = ROOT / "data" / "general" / "timeqa" / "queries.jsonl"
    else:
        queries_path = ROOT / "data" / "queries.jsonl"

    out = Path(args.out) if args.out else (
        artifacts / "results.csv" if not args.dataset else artifacts / "general_eval" / dataset / "results.csv"
    )
    if args.ablation and args.dataset:
        out = out.with_name(f"results_{args.ablation}.csv")

    index, _index_dir = _index_for(dataset if dataset != "synthetic" or not args.queries else "synthetic", artifacts)
    if dataset == "synthetic":
        index = EmbeddingIndex()
        index.load(artifacts)

    analyzer = load_query_analyzer(artifacts)
    print("Loaded trained neural query analyzer")

    generator = Generator(backend=args.generator)
    pipes = build_all_pipelines(index=index, generator=generator, analyzer=analyzer)
    if args.include_learned:
        from src.learned_pipeline import load_learned_stack
        from src.model_config import DEFAULT_RERANKER_MODEL
        from src.rerank_ce import get_cross_encoder_reranker

        learned = load_learned_stack(
            artifacts,
            index,
            get_cross_encoder_reranker(DEFAULT_RERANKER_MODEL, enabled=True),
        )
        if args.ablation == "ranker_off":
            learned.use_learned_ranker = False
            learned.name = "adaptive_learned_ranker_off"
        elif args.ablation == "reader_off":
            learned.use_reader = False
            learned.name = "adaptive_learned_reader_off"
        pipes[learned.name] = learned

    if args.systems:
        wanted = {name.strip() for name in args.systems.split(",") if name.strip()}
        pipes = {name: pipe for name, pipe in pipes.items() if name in wanted}

    queries = [query for query in load_queries(queries_path) if query.split == args.split]
    if args.limit > 0:
        queries = queries[: args.limit]
    print(f"Evaluating {len(queries)} queries on dataset={dataset} split={args.split} systems={list(pipes)}")

    rows = []
    for query in queries:
        for pipe in pipes.values():
            result = pipe.run(query)
            rows.append(_score_query(result, query, dataset))

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = {}
    for system, group in df.groupby("system"):
        summary[system] = aggregate_rows(group.to_dict(orient="records"))
    summary_path = out.with_name(out.stem + "_summary.json") if args.dataset else out.with_name("results_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for system, metrics in summary.items():
        print(
            f"{system}: contains={metrics['contains']:.3f} "
            f"em={metrics['em']:.3f} "
            f"f1={metrics['token_f1']:.3f} "
            f"abstain={metrics.get('abstain_acc', float('nan')):.3f} "
            f"temp={metrics['temporal_acc']:.3f} "
            f"ret_rate={metrics['retrieval_rate']:.3f} "
            f"lat={metrics['latency_s']:.3f}s"
        )
    print("Wrote", out)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
