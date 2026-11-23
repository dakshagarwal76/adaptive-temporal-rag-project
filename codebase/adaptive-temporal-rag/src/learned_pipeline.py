"""Learned Adaptive path: probabilities in, one ranker, a reader, then abstain or answer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .data_schema import CorpusDoc, QueryItem
from .embed_index import EmbeddingIndex
from .learned_ranker import LearnedRanker, RouterProbs
from .pipelines import BasePipeline, PipelineResult, _q
from .query_analyzer import QueryAnalyzerService
from .reader import ABSTAIN_TEXT, SquadReader, SufficiencyModel
from .rerank_ce import CrossEncoderReranker
from .temporal_tagger import parse_temporal


class LearnedAdaptivePipeline(BasePipeline):
    name = "adaptive_learned"

    def __init__(
        self,
        router: QueryAnalyzerService,
        ranker: LearnedRanker,
        reader: SquadReader,
        sufficiency: SufficiencyModel,
        index: Optional[EmbeddingIndex] = None,
        top_k: int = 5,
        candidate_k: int = 40,
        reranker: CrossEncoderReranker | None = None,
        ce_rerank_keep: int = 20,
        skip_threshold: float = 0.7,
        use_learned_ranker: bool = True,
        use_reader: bool = True,
    ) -> None:
        super().__init__(
            index=index,
            top_k=top_k,
            candidate_k=candidate_k,
            reranker=reranker,
            ce_rerank_keep=ce_rerank_keep,
        )
        self.router = router
        self.ranker = ranker
        self.reader = reader
        self.sufficiency = sufficiency
        self.skip_threshold = skip_threshold
        self.use_learned_ranker = use_learned_ranker
        self.use_reader = use_reader

    def run(self, query: QueryItem | str) -> PipelineResult:
        if self.index is None:
            raise RuntimeError("Learned Adaptive requires an embedding index")
        qtext, qid = _q(query)
        t0 = time.perf_counter()
        analysis = self.router.predict(qtext)
        parsed = parse_temporal(qtext)
        router = RouterProbs(
            p_current=analysis.temporal_probs.get("current", 0.0),
            p_historical=analysis.temporal_probs.get("historical", 0.0),
            p_timeless=analysis.temporal_probs.get("timeless", 0.0),
            p_retrieve=analysis.necessity_prob,
        )
        no_retrieve = (1.0 - analysis.necessity_prob) >= self.skip_threshold
        if no_retrieve:
            from .generate import extractive_answer

            closed = extractive_answer(qtext, [])
            return PipelineResult(
                system=self.name,
                query_id=qid,
                query=qtext,
                answer=closed.answer,
                did_retrieve=False,
                analyzer=analysis,
                latency_s=time.perf_counter() - t0,
                backend=closed.backend,
                meta={"policy": "no_retrieval", "p_retrieve": analysis.necessity_prob},
            )

        docs, scores, sufficient, probability = self._retrieve(qtext, parsed, router, self.candidate_k)
        if not sufficient:
            docs, scores, sufficient, probability = self._retrieve(
                qtext, parsed, router, min(120, self.candidate_k * 2)
            )
        answer = self._answer(qtext, docs, sufficient)
        return PipelineResult(
            system=self.name,
            query_id=qid,
            query=qtext,
            answer=answer,
            retrieved_docs=docs,
            retrieved_scores=scores,
            did_retrieve=True,
            analyzer=analysis,
            latency_s=time.perf_counter() - t0,
            backend="reader" if self.use_reader else "extractive",
            meta={
                "policy": "learned",
                "sufficient": sufficient,
                "sufficiency_p": probability,
                "p_retrieve": analysis.necessity_prob,
                "ablation_ranker": self.use_learned_ranker,
                "ablation_reader": self.use_reader,
            },
        )

    def _retrieve(self, qtext, parsed, router: RouterProbs, candidate_k: int):
        pool = self.index.search_candidates(qtext, top_k=candidate_k)
        docs = [doc for doc, _ in pool]
        faiss_scores = [score for _, score in pool]
        ce_scores = list(faiss_scores)
        if self.reranker is not None and docs:
            model = self.reranker._ensure_model()
            raw = model.predict([[qtext, doc.text[:1500]] for doc in docs], show_progress_bar=False)
            ce_scores = [float(value) for value in raw]
            keep = min(self.ce_rerank_keep, len(docs))
            order = sorted(range(len(docs)), key=lambda index: ce_scores[index], reverse=True)[:keep]
            docs = [docs[index] for index in order]
            faiss_scores = [faiss_scores[index] for index in order]
            ce_scores = [ce_scores[index] for index in order]
        if self.use_learned_ranker and docs:
            learned = self.ranker.score_pairs(qtext, docs, faiss_scores, ce_scores, parsed, router)
            order = sorted(range(len(docs)), key=lambda index: learned[index], reverse=True)[: self.top_k]
            ranked_docs = [docs[index] for index in order]
            ranked_scores = [learned[index] for index in order]
        else:
            order = sorted(range(len(docs)), key=lambda index: ce_scores[index], reverse=True)[: self.top_k]
            ranked_docs = [docs[index] for index in order]
            ranked_scores = [float(ce_scores[index]) for index in order]
        self._last_hit = self.reader.read(qtext, ranked_docs) if self.use_reader else None
        rank_score = float(ce_scores[0]) if ce_scores else 0.0
        if self._last_hit is None:
            return ranked_docs, ranked_scores, True, rank_score
        sufficient, probability = self.sufficiency.decide(
            self._last_hit.score, self._last_hit.no_answer_score, rank_score
        )
        return ranked_docs, ranked_scores, sufficient, probability

    def _answer(self, qtext: str, docs: list[CorpusDoc], sufficient: bool) -> str:
        if not self.use_reader:
            from .generate import extractive_answer

            return extractive_answer(qtext, docs).answer
        hit = getattr(self, "_last_hit", None) or self.reader.read(qtext, docs)
        if hit is None or not hit.answer.strip() or hit.answer == ABSTAIN_TEXT or not sufficient:
            return ABSTAIN_TEXT
        return hit.answer


def load_learned_stack(artifacts_dir: str | Path, index: EmbeddingIndex, reranker: CrossEncoderReranker | None):
    artifacts_dir = Path(artifacts_dir)
    from .query_analyzer import load_router_v2

    return LearnedAdaptivePipeline(
        router=load_router_v2(artifacts_dir),
        ranker=LearnedRanker.load(artifacts_dir / "learned_ranker.joblib"),
        reader=SquadReader(),
        sufficiency=SufficiencyModel.load(artifacts_dir / "sufficiency.joblib"),
        index=index,
        reranker=reranker,
    )
