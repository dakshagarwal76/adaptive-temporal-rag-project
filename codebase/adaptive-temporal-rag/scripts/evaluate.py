"""Evaluate B1/B2/B3/Adaptive on a frozen Checkpoint B config."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_loader import load_dotenv

load_dotenv(ROOT / ".env")

from src.answer_projection import extract_factoid_span
from src.config import EvalConfig, load_eval_config
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
HEURISTIC_SYSTEMS = ("B1_llm_only", "B2_standard_rag", "B3_fixed_temporal", "adaptive")
ABLATIONS = (
    "",
    "router_off",
    "window_off",
    "sufficiency_off",
    "entity_cut_off",
    "sentence_selector_off",
    "ranker_off",
    "reader_off",
)


def _score_query(result, query, dataset: str, cfg: EvalConfig) -> dict:
    ret_ids = [doc.doc_id for doc in result.retrieved_docs]
    golds = gold_strings(query)
    unanswerable = bool(getattr(query, "unanswerable", False)) or not golds
    sentence = result.answer
    span = sentence
    if cfg.span_extract_timeqa and dataset == "timeqa" and not unanswerable:
        span = extract_factoid_span(query.query, sentence)
    if unanswerable:
        em = f1 = contains = float(abstention_correct(span, True))
    else:
        em = best_exact_match(span, golds)
        f1 = best_token_f1(span, golds)
        contains = best_contains(sentence, golds)
    row = {
        "system": result.system,
        "query_id": query.query_id,
        "query": query.query,
        "temporal_class": query.temporal_class,
        "needs_retrieval": query.needs_retrieval,
        "gold_answer": query.gold_answer,
        "pred_answer": span,
        "pred_sentence": sentence,
        "em": em,
        "token_f1": f1,
        "contains": contains,
        "precision_at_k": precision_at_k(ret_ids, query.gold_doc_ids, k=5),
        "recall_at_k": recall_at_k(ret_ids, query.gold_doc_ids, k=5),
        "abstain_acc": abstention_correct(span, unanswerable),
        "latency_s": result.latency_s,
        "did_retrieve": float(result.did_retrieve),
        "backend": result.backend,
        "policy": result.meta.get("policy") or result.meta.get("fixed_mode") or "",
        "timed_out": float(bool((result.meta.get("query_budget") or {}).get("timed_out"))),
        "capped": float(bool((result.meta.get("query_budget") or {}).get("capped"))),
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


def _default_out(artifacts: Path, dataset: str, ablation: str, final: bool) -> Path:
    if final:
        if ablation:
            return artifacts / "general_eval" / "final" / "ablations" / dataset / f"results_{ablation}.csv"
        return artifacts / "general_eval" / "final" / dataset / "results.csv"
    if ablation and dataset:
        return artifacts / "general_eval" / dataset / f"results_{ablation}.csv"
    if dataset:
        return artifacts / "general_eval" / dataset / "results.csv"
    return artifacts / "results.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=str, default="")
    p.add_argument("--artifacts", type=str, default=str(ROOT / "artifacts"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset", type=str, default="", choices=["", "synthetic", "heldout", "timeqa"])
    p.add_argument("--systems", type=str, default="", help="Comma-separated system names. Default: the four baselines.")
    p.add_argument("--include-learned", action="store_true", help="Experimental learned path (not in the frozen table).")
    p.add_argument("--ablation", type=str, default="", choices=ABLATIONS)
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "final.yaml"))
    p.add_argument("--final", action="store_true", help="Write under artifacts/general_eval/final/")
    _default_gen = os.environ.get("GENERATOR_BACKEND", "extractive")
    if _default_gen not in ("extractive", "hf", "api"):
        _default_gen = "extractive"
    p.add_argument("--generator", type=str, default="", choices=["", "extractive", "hf", "api"])
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    artifacts = Path(args.artifacts)
    dataset = args.dataset or "synthetic"
    cfg = load_eval_config(args.config)
    if args.ablation in ("router_off", "window_off", "sufficiency_off", "entity_cut_off", "sentence_selector_off"):
        cfg = cfg.apply_ablation(args.ablation)
    if args.generator:
        cfg.generator = args.generator

    if args.queries:
        queries_path = Path(args.queries)
    elif dataset == "heldout":
        queries_path = ROOT / "data" / "heldout" / "queries.jsonl"
    elif dataset == "timeqa":
        queries_path = ROOT / "data" / "general" / "timeqa" / "queries.jsonl"
    else:
        queries_path = ROOT / "data" / "queries.jsonl"

    out = Path(args.out) if args.out else _default_out(artifacts, args.dataset or dataset, args.ablation, args.final)

    index, _index_dir = _index_for(dataset, artifacts)
    analyzer = load_query_analyzer(artifacts)
    print("Loaded trained neural query analyzer")
    print("Config", cfg.name, cfg.as_dict())

    generator = Generator(backend=cfg.generator, use_sentence_selector=cfg.enable_sentence_selector)
    pipes = build_all_pipelines(index=index, generator=generator, analyzer=analyzer, config=cfg)
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

    if args.ablation in ("router_off", "window_off", "sufficiency_off", "entity_cut_off", "sentence_selector_off"):
        pipes = {name: pipe for name, pipe in pipes.items() if name == "adaptive"}
        if "adaptive" in pipes:
            pipes["adaptive"].name = f"adaptive_{args.ablation}"  # type: ignore[misc]

    if args.systems:
        wanted = {name.strip() for name in args.systems.split(",") if name.strip()}
        pipes = {name: pipe for name, pipe in pipes.items() if name in wanted}
    elif not args.include_learned and not args.ablation:
        pipes = {name: pipe for name, pipe in pipes.items() if name in HEURISTIC_SYSTEMS}

    queries = [query for query in load_queries(queries_path) if query.split == args.split]
    if args.limit > 0:
        queries = queries[: args.limit]
    print(f"Evaluating {len(queries)} queries on dataset={dataset} split={args.split} systems={list(pipes)}")

    rows = []
    for i, query in enumerate(queries, start=1):
        if i == 1 or i % 25 == 0 or i == len(queries):
            print(f"  query {i}/{len(queries)}", flush=True)
        for pipe in pipes.values():
            result = pipe.run(query)
            row = _score_query(result, query, dataset, cfg)
            if args.ablation and row["system"] == "adaptive":
                row["system"] = f"adaptive_{args.ablation}"
            rows.append(row)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = {}
    for system, group in df.groupby("system"):
        summary[system] = aggregate_rows(group.to_dict(orient="records"))
    summary_path = out.with_name(out.stem + "_summary.json") if (args.dataset or args.final or args.ablation) else out.with_name("results_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for system, metrics in summary.items():
        temp = metrics.get("temporal_acc", float("nan"))
        temp_s = f"{temp:.3f}" if temp == temp else "nan"
        print(
            f"{system}: contains={metrics['contains']:.3f} "
            f"em={metrics['em']:.3f} "
            f"f1={metrics['token_f1']:.3f} "
            f"p@5={metrics.get('precision_at_k', float('nan')):.3f} "
            f"r@5={metrics.get('recall_at_k', float('nan')):.3f} "
            f"temp={temp_s} "
            f"ret_rate={metrics['retrieval_rate']:.3f} "
            f"lat_mean={metrics['latency_s']:.3f}s "
            f"lat_med={metrics.get('latency_s_median', float('nan')):.3f}s"
        )
    print("Wrote", out)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
