"""B1 LLM-only, B2 Standard RAG, B3 Fixed Temporal RAG, Adaptive Temporal RAG."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from .adaptive_policy import resolve_adaptive_routing
from .answer_projection import project_answer
from .config import EvalConfig
from .data_schema import CorpusDoc, QueryItem
from .embed_index import EmbeddingIndex
from .evidence_sufficiency import (
    MAX_REPAIR_ROUNDS,
    QUERY_TIMEOUT_S,
    QueryBudget,
    RepairAction,
    SufficiencyResult,
    assess_evidence_sufficiency,
)
from .generate import Generator, is_trivial_query
from .query_analyzer import AnalyzerOutput
from .rerank_ce import CrossEncoderReranker, get_cross_encoder_reranker
from .temporal_rank import (
    adaptive_filter_by_window,
    extract_year_hint,
    filter_by_year_window,
    filter_subject,
    infer_adaptive_window,
    lexical_overlap_boost,
    rerank,
)

SystemName = Literal[
    "B1_llm_only",
    "B2_standard_rag",
    "B3_fixed_temporal",
    "adaptive",
    "adaptive_learned",
]


class AnalyzerLike(Protocol):
    conf_threshold: float

    def predict(self, query: str) -> AnalyzerOutput: ...


@dataclass
class PipelineResult:
    system: SystemName
    query_id: str
    query: str
    answer: str
    retrieved_docs: list[CorpusDoc] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    did_retrieve: bool = False
    analyzer: Optional[AnalyzerOutput] = None
    latency_s: float = 0.0
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class BasePipeline:
    name: SystemName = "B1_llm_only"

    def __init__(
        self,
        index: Optional[EmbeddingIndex] = None,
        generator: Optional[Generator] = None,
        top_k: int = 5,
        candidate_k: int = 50,
        alpha: float = 0.6,
        reranker: CrossEncoderReranker | None = None,
        ce_rerank_keep: int = 40,
        use_entity_cut: bool = True,
        use_answer_projection: bool = True,
        use_sentence_selector: bool = True,
    ) -> None:
        self.index = index
        self.generator = generator or Generator(
            backend="extractive", use_sentence_selector=use_sentence_selector
        )
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.alpha = alpha
        self.reranker = reranker
        self.ce_rerank_keep = ce_rerank_keep
        self.use_entity_cut = use_entity_cut
        self.use_answer_projection = use_answer_projection
        self.use_sentence_selector = use_sentence_selector

    def _subject_pool(
        self, query: str, candidates: list[tuple[CorpusDoc, float]]
    ) -> list[tuple[CorpusDoc, float]]:
        if not self.use_entity_cut:
            return candidates
        return filter_subject(candidates, query)

    def _read_answer(self, query: str, docs: list[CorpusDoc] | None) -> tuple[str, str]:
        gen = self.generator.generate(query, docs=docs)
        answer = project_answer(query, gen.answer) if self.use_answer_projection else gen.answer
        return answer, gen.backend

    def _rerank_faiss_candidates(
        self, query: str, candidates: list[tuple[CorpusDoc, float]], keep: int
    ) -> list[tuple[CorpusDoc, float]]:
        if self.reranker is None or not candidates:
            return candidates[:keep]
        return self.reranker.rerank(query, candidates, keep=keep)

    def run(self, query: QueryItem | str) -> PipelineResult:
        raise NotImplementedError


class LLMOnlyPipeline(BasePipeline):
    name: SystemName = "B1_llm_only"

    def run(self, query: QueryItem | str) -> PipelineResult:
        qtext, qid = _q(query)
        t0 = time.perf_counter()
        answer, backend = self._read_answer(qtext, None)
        return PipelineResult(
            system=self.name,
            query_id=qid,
            query=qtext,
            answer=answer,
            did_retrieve=False,
            latency_s=time.perf_counter() - t0,
            backend=backend,
        )


class StandardRAGPipeline(BasePipeline):
    name: SystemName = "B2_standard_rag"

    def run(self, query: QueryItem | str) -> PipelineResult:
        if self.index is None:
            raise RuntimeError("StandardRAG requires an embedding index")
        qtext, qid = _q(query)
        t0 = time.perf_counter()
        pool = self.index.search_candidates(qtext, top_k=max(self.candidate_k, self.top_k))
        hits = self._rerank_faiss_candidates(qtext, pool, keep=min(self.ce_rerank_keep, len(pool)))
        hits = self._subject_pool(qtext, hits)[: self.top_k]
        docs = [d for d, _ in hits]
        scores = [s for _, s in hits]
        answer, backend = self._read_answer(qtext, docs)
        return PipelineResult(
            system=self.name,
            query_id=qid,
            query=qtext,
            answer=answer,
            retrieved_docs=docs,
            retrieved_scores=scores,
            did_retrieve=True,
            latency_s=time.perf_counter() - t0,
            backend=backend,
        )


class FixedTemporalRAGPipeline(BasePipeline):
    name: SystemName = "B3_fixed_temporal"

    def run(self, query: QueryItem | str) -> PipelineResult:
        if self.index is None:
            raise RuntimeError("FixedTemporalRAG requires an embedding index")
        qtext, qid = _q(query)
        time_hint = query.time_hint if isinstance(query, QueryItem) else None
        t0 = time.perf_counter()
        cands = self.index.search_candidates(qtext, top_k=self.candidate_k)
        cands = self._rerank_faiss_candidates(qtext, cands, keep=min(self.ce_rerank_keep, len(cands)))
        cands = self._subject_pool(qtext, cands)

        # Fixed policy: if a year is present treat as historical window; else recency boost
        year = extract_year_hint(time_hint, qtext)
        if year is not None:
            cands = filter_by_year_window(cands, year, window_years=2)
            ranked = rerank(cands, mode="historical", query=qtext, time_hint=time_hint or str(year), alpha=self.alpha, top_k=self.top_k)
            mode = "historical"
        else:
            ranked = rerank(cands, mode="current", query=qtext, alpha=self.alpha, top_k=self.top_k)
            mode = "current"

        docs = [d for d, _, _, _ in ranked]
        scores = [f for _, f, _, _ in ranked]
        answer, backend = self._read_answer(qtext, docs)
        return PipelineResult(
            system=self.name,
            query_id=qid,
            query=qtext,
            answer=answer,
            retrieved_docs=docs,
            retrieved_scores=scores,
            did_retrieve=True,
            latency_s=time.perf_counter() - t0,
            backend=backend,
            meta={"fixed_mode": mode},
        )


class AdaptiveTemporalRAGPipeline(BasePipeline):
    """Adaptive retrieval + adaptive windows + evidence sufficiency feedback loop."""

    name: SystemName = "adaptive"

    def __init__(
        self,
        analyzer: AnalyzerLike,
        index: Optional[EmbeddingIndex] = None,
        generator: Optional[Generator] = None,
        top_k: int = 5,
        candidate_k: int = 50,
        alpha: float = 0.6,
        conf_threshold: float = 0.55,
        max_sufficiency_rounds: int = 2,
        sufficiency_threshold: float = 0.55,
        enable_sufficiency_loop: bool = True,
        enable_adaptive_window: bool = True,
        always_retrieve: bool = False,
        query_timeout_s: float = QUERY_TIMEOUT_S,
        reranker: CrossEncoderReranker | None = None,
        ce_rerank_keep: int = 40,
        use_entity_cut: bool = True,
        use_answer_projection: bool = True,
        use_sentence_selector: bool = True,
    ) -> None:
        super().__init__(
            index=index,
            generator=generator,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
            reranker=reranker,
            ce_rerank_keep=ce_rerank_keep,
            use_entity_cut=use_entity_cut,
            use_answer_projection=use_answer_projection,
            use_sentence_selector=use_sentence_selector,
        )
        self.analyzer = analyzer
        self.conf_threshold = conf_threshold
        self.max_sufficiency_rounds = max(0, min(max_sufficiency_rounds, MAX_REPAIR_ROUNDS - 1))
        self.sufficiency_threshold = sufficiency_threshold
        self.enable_sufficiency_loop = enable_sufficiency_loop
        self.enable_adaptive_window = enable_adaptive_window
        self.always_retrieve = always_retrieve
        self.query_timeout_s = query_timeout_s

    def run(
        self,
        query: QueryItem | str,
        inference_trace: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        qtext, qid = _q(query)
        time_hint = query.time_hint if isinstance(query, QueryItem) else None
        t0 = time.perf_counter()
        analysis = self.analyzer.predict(qtext)
        if inference_trace is not None:
            inference_trace.append(
                {
                    "stage": "query_analyzer",
                    "needs_retrieval": analysis.needs_retrieval,
                    "temporal_class": analysis.temporal_class,
                    "confidence": round(analysis.confidence, 4),
                    "necessity_prob": round(analysis.necessity_prob, 4),
                    "temporal_probs": {k: round(v, 4) for k, v in analysis.temporal_probs.items()},
                }
            )

        routing = resolve_adaptive_routing(analysis, qtext, self.conf_threshold)
        if self.always_retrieve and routing.skip_retrieval:
            routing = resolve_adaptive_routing(analysis, qtext, 0.0)
            if routing.skip_retrieval:
                from .adaptive_policy import AdaptiveRoutingDecision

                mode = analysis.temporal_class if analysis.temporal_class in ("current", "historical") else "semantic"
                routing = AdaptiveRoutingDecision(False, "router_off_always_retrieve", mode)
        if inference_trace is not None:
            inference_trace.append(
                {
                    "stage": "policy_router",
                    "skip_retrieval": routing.skip_retrieval,
                    "policy": routing.policy,
                    "retrieve_mode": routing.retrieve_mode or None,
                    "conf_threshold": self.conf_threshold,
                    "always_retrieve": self.always_retrieve,
                }
            )
        if routing.skip_retrieval:
            if inference_trace is not None:
                inference_trace.append(
                    {"stage": "generator_no_docs", "reason": "trivial_and_no_retrieval"}
                )
            answer, backend = self._read_answer(qtext, None)
            return PipelineResult(
                system=self.name,
                query_id=qid,
                query=qtext,
                answer=answer,
                did_retrieve=False,
                analyzer=analysis,
                latency_s=time.perf_counter() - t0,
                backend=backend,
                meta={"policy": "no_retrieval", "sufficiency_loop": []},
            )

        if self.index is None:
            raise RuntimeError("Adaptive pipeline requires an embedding index when retrieving")

        policy = routing.policy
        retrieve_mode = routing.retrieve_mode

        # ---- evidence sufficiency feedback loop ----
        cand_k = self.candidate_k
        top_k = self.top_k
        alpha = self.alpha
        force_semantic = retrieve_mode == "semantic"
        force_recency = False
        window_boost = 0.0
        window_meta: dict[str, Any] | None = None
        loop_log: list[dict[str, Any]] = []
        docs: list[CorpusDoc] = []
        scores: list[float] = []
        sem_scores: list[float] = []
        last_suff: SufficiencyResult | None = None
        used_mode = retrieve_mode
        historical_docs: list[CorpusDoc] | None = None
        historical_scores: list[float] | None = None
        historical_sem: list[float] | None = None
        historical_suff: SufficiencyResult | None = None
        historical_window: dict[str, Any] | None = None
        cand_cache: dict[int, list[tuple[CorpusDoc, float]]] = {}
        budget = QueryBudget(
            timeout_s=self.query_timeout_s,
            max_rounds=MAX_REPAIR_ROUNDS if self.enable_sufficiency_loop else 1,
        )

        while budget.allow_round():
            round_i = budget.rounds - 1
            effective_mode = (
                "semantic"
                if force_semantic
                else ("current" if force_recency and retrieve_mode == "current" else retrieve_mode)
            )
            if inference_trace is not None:
                inference_trace.append(
                    {
                        "stage": "retrieve_rank_start",
                        "round": round_i,
                        "retrieve_mode": effective_mode,
                        "candidate_k": cand_k,
                        "top_k": top_k,
                        "alpha": 0.7 if force_semantic else (0.45 if force_recency else alpha),
                        "window_boost": window_boost,
                    }
                )
            ranked, round_window, used_mode, entity_ok = self._retrieve_rank(
                qtext=qtext,
                time_hint=time_hint,
                retrieve_mode="semantic" if force_semantic else ("current" if force_recency and retrieve_mode == "current" else retrieve_mode),
                candidate_k=cand_k,
                top_k=top_k,
                alpha=0.7 if force_semantic else (0.45 if force_recency else alpha),
                window_boost=window_boost,
                cand_cache=cand_cache,
            )
            docs = [d for d, _, _, _ in ranked]
            scores = [final for _, final, _, _ in ranked]
            sem_scores = [sem for _, _, sem, _ in ranked]
            if round_window is not None:
                window_meta = round_window
            if inference_trace is not None:
                inference_trace.append(
                    {
                        "stage": "retrieve_rank_done",
                        "round": round_i,
                        "used_mode": used_mode,
                        "n_ranked": len(docs),
                        "top_doc_ids": [d.doc_id for d in docs[:3]],
                        "top_scores": [round(s, 4) for s in scores[:3]],
                        "adaptive_window": round_window,
                        "entity_matched": entity_ok,
                    }
                )

            suff = assess_evidence_sufficiency(
                query=qtext,
                docs=docs,
                scores=sem_scores,
                temporal_class=used_mode if used_mode != "semantic" else analysis.temporal_class,
                time_hint=time_hint,
                min_score=self.sufficiency_threshold,
                entity_matched=entity_ok,
            )
            last_suff = suff
            if used_mode == "historical" and round_window is not None:
                if historical_suff is None or suff.score >= historical_suff.score:
                    historical_docs = list(docs)
                    historical_scores = list(scores)
                    historical_sem = list(sem_scores)
                    historical_suff = suff
                    historical_window = round_window
            loop_log.append(
                {
                    "round": round_i,
                    "mode": used_mode,
                    "candidate_k": cand_k,
                    "top_k": top_k,
                    "sufficiency": suff.as_meta(),
                    "repairs_applied": [],
                }
            )
            if inference_trace is not None:
                inference_trace.append(
                    {
                        "stage": "evidence_sufficiency",
                        "round": round_i,
                        **suff.as_meta(),
                    }
                )

            if suff.sufficient or not self.enable_sufficiency_loop or budget.expired():
                break

            applied = self._apply_repairs(
                repairs=suff.suggested_repairs,
                cand_k_ref={"v": cand_k},
                top_k_ref={"v": top_k},
                window_boost_ref={"v": window_boost},
                force_semantic_ref={"v": force_semantic},
                force_recency_ref={"v": force_recency},
            )
            cand_k = applied["candidate_k"]
            top_k = applied["top_k"]
            window_boost = applied["window_boost"]
            force_semantic = applied["force_semantic"]
            force_recency = applied["force_recency"]
            loop_log[-1]["repairs_applied"] = applied["applied"]
            if inference_trace is not None:
                inference_trace.append(
                    {"stage": "sufficiency_repairs", "round": round_i, **applied}
                )

            if not applied["applied"]:
                break

        if (
            historical_suff is not None
            and historical_docs is not None
            and historical_scores is not None
            and historical_sem is not None
            and used_mode != "historical"
            and (last_suff is None or last_suff.score <= historical_suff.score)
        ):
            docs = historical_docs
            scores = historical_scores
            sem_scores = historical_sem
            last_suff = historical_suff
            window_meta = historical_window
            used_mode = "historical"

        if inference_trace is not None:
            inference_trace.append(
                {
                    "stage": "generator",
                    "n_context_docs": len(docs),
                    "backend": getattr(self.generator, "backend_name", "extractive"),
                }
            )
        answer, backend = self._read_answer(qtext, docs)

        # One post-answer sufficiency check: if answer is weak, try a final expand once
        if self.enable_sufficiency_loop and docs and budget.allow_round() and not budget.expired():
            post = assess_evidence_sufficiency(
                query=qtext,
                docs=docs,
                scores=sem_scores,
                temporal_class=analysis.temporal_class,
                time_hint=time_hint,
                answer=answer,
                min_score=self.sufficiency_threshold,
                entity_matched=True,
            )
            if not post.sufficient and "weak_generated_answer" in post.reasons:
                if inference_trace is not None:
                    inference_trace.append(
                        {"stage": "post_answer_retry", "sufficiency": post.as_meta()}
                    )
                loop_log.append({"round": "post_answer", "sufficiency": post.as_meta(), "repairs_applied": ["expand_candidates", "relax_to_semantic"]})
                ranked2, window2, _, entity2 = self._retrieve_rank(
                    qtext=qtext,
                    time_hint=time_hint,
                    retrieve_mode="semantic",
                    candidate_k=min(200, max(cand_k * 2, self.candidate_k * 2)),
                    top_k=min(12, top_k + 3),
                    alpha=0.75,
                    window_boost=window_boost + 1.0,
                    cand_cache=cand_cache,
                )
                docs2 = [d for d, _, _, _ in ranked2]
                scores2 = [final for _, final, _, _ in ranked2]
                sem2 = [sem for _, _, sem, _ in ranked2]
                answer2, backend2 = self._read_answer(qtext, docs2)
                post2 = assess_evidence_sufficiency(
                    query=qtext,
                    docs=docs2,
                    scores=sem2,
                    temporal_class=analysis.temporal_class,
                    time_hint=time_hint,
                    answer=answer2,
                    min_score=self.sufficiency_threshold,
                    entity_matched=entity2,
                )
                if post2.score > post.score:
                    docs, scores, sem_scores, answer, backend, last_suff = docs2, scores2, sem2, answer2, backend2, post2
                    policy = f"{policy}+sufficiency_retry"
                    window_meta = window2

        budget.log_stop(qtext)
        meta: dict[str, Any] = {
            "policy": policy,
            "sufficiency_loop": loop_log,
            "query_budget": budget.as_meta(),
        }
        if last_suff is not None:
            meta["evidence_sufficiency"] = last_suff.as_meta()
        if window_meta is not None:
            meta["adaptive_window"] = window_meta

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
            backend=backend,
            meta=meta,
        )

    def _retrieve_rank(
        self,
        qtext: str,
        time_hint: Optional[str],
        retrieve_mode: str,
        candidate_k: int,
        top_k: int,
        alpha: float,
        window_boost: float,
        cand_cache: dict[int, list[tuple[CorpusDoc, float]]] | None = None,
    ) -> tuple[list[tuple[CorpusDoc, float, float, float]], dict[str, Any] | None, str, bool]:
        assert self.index is not None
        if cand_cache is not None and candidate_k in cand_cache:
            cands = list(cand_cache[candidate_k])
        else:
            cands = self.index.search_candidates(qtext, top_k=candidate_k)
            cands = self._rerank_faiss_candidates(
                qtext, cands, keep=min(self.ce_rerank_keep, len(cands))
            )
            if cand_cache is not None:
                cand_cache[candidate_k] = list(cands)
        window_meta: dict[str, Any] | None = None
        used_mode = retrieve_mode

        if retrieve_mode == "historical":
            if self.enable_adaptive_window:
                window = infer_adaptive_window(qtext, time_hint=time_hint, min_docs=max(3, top_k // 2))
                if window_boost > 0:
                    window.initial_years = min(window.max_years, window.initial_years + window_boost)
                    window.max_years = min(12.0, window.max_years + window_boost)
                filtered, window = adaptive_filter_by_window(
                    cands, window, query=qtext, top_k=top_k, use_entity_cut=self.use_entity_cut
                )
                ranked = rerank(
                    filtered,
                    mode="historical",
                    query=qtext,
                    time_hint=time_hint,
                    alpha=alpha,
                    top_k=top_k,
                    window_years=window.final_years or window.initial_years,
                    prefer_time=True,
                )
                window_meta = window.as_meta()
            else:
                year = extract_year_hint(time_hint, qtext)
                pooled = self._subject_pool(qtext, cands)
                filtered = filter_by_year_window(pooled, year, window_years=2)
                ranked = rerank(
                    filtered,
                    mode="historical",
                    query=qtext,
                    time_hint=time_hint or (str(year) if year else None),
                    alpha=alpha,
                    top_k=top_k,
                    prefer_time=True,
                )
                window_meta = {"reason": "fixed_pm2", "window_years": 2, "target_year": year}
        elif retrieve_mode == "current":
            ranked = rerank(
                self._subject_pool(qtext, cands), mode="current", query=qtext, alpha=alpha, top_k=top_k
            )
        else:
            ranked = rerank(
                self._subject_pool(qtext, cands), mode="semantic", query=qtext, alpha=alpha, top_k=top_k
            )
            used_mode = "semantic"
        return ranked, window_meta, used_mode, _top_matches_best_entity(qtext, ranked, cands)

    @staticmethod
    def _apply_repairs(
        repairs: list[RepairAction],
        cand_k_ref: dict[str, int],
        top_k_ref: dict[str, int],
        window_boost_ref: dict[str, float],
        force_semantic_ref: dict[str, bool],
        force_recency_ref: dict[str, bool],
    ) -> dict[str, Any]:
        applied: list[str] = []
        cand_k = cand_k_ref["v"]
        top_k = top_k_ref["v"]
        window_boost = window_boost_ref["v"]
        force_semantic = force_semantic_ref["v"]
        force_recency = force_recency_ref["v"]

        for action in repairs:
            if action == "none":
                continue
            if action == "expand_candidates":
                new_k = min(200, max(cand_k + 30, int(cand_k * 1.75)))
                if new_k > cand_k:
                    cand_k = new_k
                    applied.append(action)
            elif action == "raise_topk":
                new_top = min(12, top_k + 3)
                if new_top > top_k:
                    top_k = new_top
                    applied.append(action)
            elif action == "widen_window":
                window_boost += 1.25
                applied.append(action)
            elif action == "relax_to_semantic":
                if not force_semantic:
                    force_semantic = True
                    applied.append(action)
            elif action == "boost_recency":
                if not force_recency:
                    force_recency = True
                    applied.append(action)

        return {
            "applied": applied,
            "candidate_k": cand_k,
            "top_k": top_k,
            "window_boost": window_boost,
            "force_semantic": force_semantic,
            "force_recency": force_recency,
        }


def build_all_pipelines(
    index: EmbeddingIndex,
    generator: Optional[Generator] = None,
    analyzer: AnalyzerLike | None = None,
    top_k: int = 5,
    alpha: float = 0.6,
    conf_threshold: float = 0.55,
    candidate_k: int = 80,
    enable_ce_rerank: bool = True,
    reranker_model: str | None = None,
    config: EvalConfig | None = None,
) -> dict[str, BasePipeline]:
    from .model_config import DEFAULT_RERANKER_MODEL

    cfg = config or EvalConfig()
    generator = generator or Generator(
        backend=cfg.generator, use_sentence_selector=cfg.enable_sentence_selector
    )
    if hasattr(generator, "use_sentence_selector"):
        generator.use_sentence_selector = cfg.enable_sentence_selector
    reranker = get_cross_encoder_reranker(
        model_name=reranker_model or DEFAULT_RERANKER_MODEL,
        enabled=enable_ce_rerank and cfg.enable_ce_rerank,
    )
    shared = dict(
        index=index,
        generator=generator,
        top_k=cfg.top_k if config else top_k,
        candidate_k=cfg.candidate_k if config else candidate_k,
        alpha=cfg.alpha if config else alpha,
        reranker=reranker,
        ce_rerank_keep=cfg.ce_rerank_keep,
        use_entity_cut=cfg.enable_entity_cut,
        use_answer_projection=cfg.enable_answer_projection,
        use_sentence_selector=cfg.enable_sentence_selector,
    )
    return {
        "B1_llm_only": LLMOnlyPipeline(
            generator=generator,
            use_answer_projection=cfg.enable_answer_projection,
            use_sentence_selector=cfg.enable_sentence_selector,
        ),
        "B2_standard_rag": StandardRAGPipeline(**shared),
        "B3_fixed_temporal": FixedTemporalRAGPipeline(**shared),
        "adaptive": AdaptiveTemporalRAGPipeline(
            analyzer=analyzer,
            **shared,
            conf_threshold=cfg.conf_threshold if config else conf_threshold,
            enable_sufficiency_loop=cfg.enable_sufficiency_loop,
            enable_adaptive_window=cfg.enable_adaptive_window,
            always_retrieve=cfg.always_retrieve,
            max_sufficiency_rounds=cfg.max_sufficiency_rounds,
            sufficiency_threshold=cfg.sufficiency_threshold,
            query_timeout_s=cfg.query_timeout_s,
        ),
    }


def _top_matches_best_entity(
    query: str,
    ranked: list[tuple[CorpusDoc, float, float, float]],
    pool: list[tuple[CorpusDoc, float]],
) -> bool:
    """False when a better entity match exists in the candidate pool than the top hit."""
    if not ranked or not query or not pool:
        return True
    best = max(lexical_overlap_boost(query, doc, ignore_years=True, entity_only=True) for doc, _ in pool)
    if best <= 0:
        return True
    top = lexical_overlap_boost(query, ranked[0][0], ignore_years=True, entity_only=True)
    return top >= best - 0.01


def _q(query: QueryItem | str) -> tuple[str, str]:
    if isinstance(query, QueryItem):
        return query.query, query.query_id
    return query, "adhoc"
