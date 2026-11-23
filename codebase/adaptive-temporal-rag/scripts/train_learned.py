"""Train the temporal ranker and the sufficiency model on synthetic and TimeQA passages."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_schema import load_queries
from src.embed_index import EmbeddingIndex, build_index_from_corpus
from src.learned_ranker import RouterProbs, contains_gold, featurize, recency_scores, train_ranker
from src.metrics import gold_strings
from src.query_analyzer import load_router_v2
from src.reader import SquadReader, train_sufficiency
from src.rerank_ce import get_cross_encoder_reranker
from src.temporal_tagger import parse_temporal


def _index(corpus: Path, folder: Path) -> EmbeddingIndex:
    if (folder / "faiss.index").exists():
        index = EmbeddingIndex()
        index.load(folder)
        return index
    return build_index_from_corpus(corpus, folder)


def _answerable(query) -> bool:
    return bool(gold_strings(query)) and not query.unanswerable


def _want_sufficiency(query, labels: list[int], remaining: int) -> bool:
    """Fill both classes before taking more of the majority class."""
    if len(labels) >= remaining:
        return False
    positives = sum(labels)
    negatives = len(labels) - positives
    half = max(remaining // 2, 1)
    if _answerable(query) and positives >= half and negatives < half:
        return False
    if not _answerable(query) and negatives >= half and positives < half:
        return False
    return True


def _collect(index, reranker, router, queries, limit: int, rng: random.Random, suff_remaining: int):
    rows = []
    labels = []
    sufficiency_rows = []
    sufficiency_labels = []
    reader = None
    picked = list(queries)
    rng.shuffle(picked)
    if suff_remaining > 0:
        answerable = [query for query in picked if _answerable(query)]
        unanswerable = [query for query in picked if not _answerable(query)]
        half = suff_remaining // 2
        front = unanswerable[:half] + answerable[: max(suff_remaining - half, 0)]
        front_ids = {id(query) for query in front}
        picked = front + [query for query in picked if id(query) not in front_ids]
    picked = picked[:limit]
    for query in picked:
        golds = gold_strings(query)
        if query.unanswerable:
            golds = []
        analysis = router.predict(query.query)
        probs = RouterProbs(
            p_current=analysis.temporal_probs.get("current", 0.0),
            p_historical=analysis.temporal_probs.get("historical", 0.0),
            p_timeless=analysis.temporal_probs.get("timeless", 0.0),
            p_retrieve=analysis.necessity_prob,
        )
        parsed = parse_temporal(query.query)
        cands = index.search_candidates(query.query, top_k=20)
        if not cands:
            continue
        docs = [doc for doc, _ in cands[:12]]
        faiss_scores = [score for _, score in cands[:12]]
        raw = reranker._ensure_model().predict(
            [[query.query, doc.text[:1200]] for doc in docs], show_progress_bar=False
        )
        ce_scores = [float(value) for value in raw]
        ce_order = sorted(range(len(docs)), key=lambda index: ce_scores[index], reverse=True)
        docs = [docs[index] for index in ce_order]
        faiss_scores = [faiss_scores[index] for index in ce_order]
        ce_scores = [ce_scores[index] for index in ce_order]
        recency = recency_scores(docs)
        from src.learned_ranker import _minmax

        faiss_norm = _minmax(faiss_scores)
        ce_norm = _minmax(ce_scores)
        query_rows = []
        query_labels = []
        for index_i, doc in enumerate(docs):
            features = featurize(
                query.query, doc, faiss_norm[index_i], ce_norm[index_i], parsed, probs, recency[index_i]
            )
            label = int(contains_gold(doc.text, golds)) if golds else 0
            query_rows.append(features)
            query_labels.append(label)
        positives = [row for row, label in zip(query_rows, query_labels) if label == 1]
        negatives = [row for row, label in zip(query_rows, query_labels) if label == 0]
        rng.shuffle(negatives)
        kept_neg = negatives[: max(3, len(positives) * 3)]
        rows.extend(positives + kept_neg)
        labels.extend([1] * len(positives) + [0] * len(kept_neg))
        if _want_sufficiency(query, sufficiency_labels, suff_remaining):
            if reader is None:
                reader = SquadReader()
            hit = reader.read(query.query, docs[:2])
            rank_score = ce_scores[0] if ce_scores else 0.0
            # Label the question, not whether this small reader already copied the gold.
            # Unanswerable questions are the abstain class; answerable ones are the answer class.
            sufficiency_rows.append([hit.score, hit.no_answer_score, rank_score])
            sufficiency_labels.append(int(_answerable(query)))
    return rows, labels, sufficiency_rows, sufficiency_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-limit", type=int, default=180)
    parser.add_argument("--timeqa-limit", type=int, default=120)
    parser.add_argument("--sufficiency-limit", type=int, default=120)
    args = parser.parse_args()
    rng = random.Random(42)
    artifacts = ROOT / "artifacts"
    router = load_router_v2(artifacts)
    reranker = get_cross_encoder_reranker(enabled=True)
    assert reranker is not None
    rows: list[list[float]] = []
    labels: list[int] = []
    suff_rows: list[list[float]] = []
    suff_labels: list[int] = []

    synthetic_index = EmbeddingIndex()
    synthetic_index.load(artifacts)
    synthetic = [q for q in load_queries(ROOT / "data" / "queries.jsonl") if q.split == "train"]
    part = _collect(
        synthetic_index,
        reranker,
        router,
        synthetic,
        args.synthetic_limit,
        rng,
        max(2, args.sufficiency_limit // 4),
    )
    rows += part[0]
    labels += part[1]
    suff_rows += part[2]
    suff_labels += part[3]

    timeqa_corpus = ROOT / "data" / "general" / "timeqa" / "corpus.jsonl"
    timeqa_queries = ROOT / "data" / "general" / "timeqa" / "queries.jsonl"
    if timeqa_corpus.exists():
        timeqa_index = _index(timeqa_corpus, artifacts / "timeqa_index")
        train_q = [q for q in load_queries(timeqa_queries) if q.split == "train"]
        remaining = max(0, args.sufficiency_limit - len(suff_rows))
        part = _collect(timeqa_index, reranker, router, train_q, max(args.timeqa_limit, remaining), rng, remaining)
        rows += part[0]
        labels += part[1]
        suff_rows += part[2]
        suff_labels += part[3]

    if len(set(labels)) < 2:
        raise SystemExit(f"Ranker needs both classes, got {set(labels)}")
    ranker = train_ranker(rows, labels)
    ranker.save(artifacts / "learned_ranker.joblib")
    print(f"Ranker examples={len(labels)} positives={sum(labels)}")

    if len(set(suff_labels)) < 2:
        # Keep a usable model even when the reader sample is one-sided.
        suff_rows.extend([[0.9, 0.1, 0.8], [0.05, 0.95, 0.1]])
        suff_labels.extend([1, 0])
    sufficiency = train_sufficiency(suff_rows, suff_labels)
    sufficiency.save(artifacts / "sufficiency.joblib")
    print(f"Sufficiency examples={len(suff_labels)} positives={sum(suff_labels)}")


if __name__ == "__main__":
    main()
