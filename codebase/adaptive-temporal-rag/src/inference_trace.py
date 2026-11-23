"""Step-by-step inference architecture trace for Colab / reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .embed_index import EmbeddingIndex
from .generate import Generator
from .pipelines import (
    AdaptiveTemporalRAGPipeline,
    FixedTemporalRAGPipeline,
    LLMOnlyPipeline,
    PipelineResult,
    StandardRAGPipeline,
)
from .query_analyzer import AnalyzerLike, load_query_analyzer


def trace_adaptive_inference(
    query: str,
    pipeline: AdaptiveTemporalRAGPipeline,
    trace: Optional[list[dict[str, Any]]] = None,
) -> tuple[PipelineResult, list[dict[str, Any]]]:
    """Run Adaptive pipeline with full step log (same code path as production)."""
    log: list[dict[str, Any]] = trace if trace is not None else []
    log.append({"stage": "input", "query": query})
    result = pipeline.run(query, inference_trace=log)
    log.append(
        {
            "stage": "output",
            "answer_preview": (result.answer or "")[:500],
            "did_retrieve": result.did_retrieve,
            "backend": result.backend,
            "n_docs": len(result.retrieved_docs),
            "policy": result.meta.get("policy"),
        }
    )
    return result, log


def trace_all_systems(
    query: str,
    index: EmbeddingIndex,
    generator: Generator,
    analyzer: AnalyzerLike,
    top_k: int = 8,
    candidate_k: int = 80,
    alpha: float = 0.55,
    conf_threshold: float = 0.55,
) -> dict[str, Any]:
    """Run B1–B4 on one query; return paths + results for architecture comparison."""
    out: dict[str, Any] = {"query": query, "systems": {}}

    b1 = LLMOnlyPipeline(generator=generator)
    r1 = b1.run(query)
    out["systems"]["B1_llm_only"] = {
        "path": ["query", "generator (no FAISS)", "answer"],
        "did_retrieve": r1.did_retrieve,
        "answer_preview": (r1.answer or "")[:280],
        "latency_s": round(r1.latency_s, 4),
    }

    b2 = StandardRAGPipeline(index=index, generator=generator, top_k=top_k)
    r2 = b2.run(query)
    out["systems"]["B2_standard_rag"] = {
        "path": ["query", "BGE+FAISS", "cross-encoder rerank", "generator", "answer"],
        "did_retrieve": r2.did_retrieve,
        "n_docs": len(r2.retrieved_docs),
        "answer_preview": (r2.answer or "")[:280],
        "latency_s": round(r2.latency_s, 4),
    }

    b3 = FixedTemporalRAGPipeline(index=index, generator=generator, top_k=top_k, alpha=alpha)
    r3 = b3.run(query)
    out["systems"]["B3_fixed_temporal"] = {
        "path": [
            "query",
            "BGE+FAISS",
            "cross-encoder rerank",
            "fixed temporal window / recency",
            "generator",
            "answer",
        ],
        "did_retrieve": r3.did_retrieve,
        "n_docs": len(r3.retrieved_docs),
        "meta": r3.meta,
        "answer_preview": (r3.answer or "")[:280],
        "latency_s": round(r3.latency_s, 4),
    }

    adaptive = AdaptiveTemporalRAGPipeline(
        analyzer=analyzer,
        index=index,
        generator=generator,
        top_k=top_k,
        candidate_k=candidate_k,
        alpha=alpha,
        conf_threshold=conf_threshold,
    )
    trace: list[dict[str, Any]] = []
    r4, trace = trace_adaptive_inference(query, adaptive, trace=trace)
    out["systems"]["adaptive"] = {
        "path": [
            "query",
            "trained analyzer",
            "policy router",
            "BGE+FAISS",
            "cross-encoder rerank",
            "temporal rerank (current / historical window / semantic)",
            "evidence sufficiency loop",
            "generator",
            "optional post-answer retry",
            "answer",
        ],
        "did_retrieve": r4.did_retrieve,
        "n_docs": len(r4.retrieved_docs),
        "policy": r4.meta.get("policy"),
        "adaptive_window": r4.meta.get("adaptive_window"),
        "evidence_sufficiency": r4.meta.get("evidence_sufficiency"),
        "sufficiency_loop": r4.meta.get("sufficiency_loop"),
        "answer_preview": (r4.answer or "")[:280],
        "latency_s": round(r4.latency_s, 4),
        "trace": trace,
    }
    return out


def format_trace_text(trace: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, step in enumerate(trace, start=1):
        stage = step.get("stage", "?")
        lines.append(f"--- Step {i}: {stage} ---")
        for k, v in step.items():
            if k == "stage":
                continue
            if isinstance(v, (dict, list)):
                lines.append(f"  {k}: {json.dumps(v, ensure_ascii=False)[:2000]}")
            else:
                lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def print_architecture_walkthrough(walkthrough: dict[str, Any]) -> None:
    print("=" * 72)
    print("QUERY:", walkthrough["query"])
    print("=" * 72)
    for name, info in walkthrough["systems"].items():
        print(f"\n### {name}")
        print("Path:", " → ".join(info["path"]))
        print(f"retrieve={info.get('did_retrieve')} latency={info.get('latency_s')}s")
        if info.get("policy"):
            print("policy:", info["policy"])
        if info.get("n_docs") is not None:
            print("n_docs:", info["n_docs"])
        print("answer:", info.get("answer_preview", ""))
        if name == "adaptive" and info.get("trace"):
            print("\nAdaptive step log:")
            print(format_trace_text(info["trace"]))


def load_walkthrough_from_artifacts(
    query: str,
    artifacts_dir: str | Path,
    generator_backend: str = "extractive",
    **kwargs: Any,
) -> dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    index = EmbeddingIndex()
    index.load(artifacts_dir)
    analyzer = load_query_analyzer(artifacts_dir)
    generator = Generator(backend=generator_backend)
    return trace_all_systems(query, index, generator, analyzer, **kwargs)


def save_walkthrough_json(walkthrough: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(walkthrough, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
