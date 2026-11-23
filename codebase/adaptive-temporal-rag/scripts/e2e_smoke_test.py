"""Thorough end-to-end smoke test for Adaptive Temporal RAG.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env_loader import load_dotenv

load_dotenv(ROOT / ".env")

PASS = 0
FAIL = 0
WARN = 0
FAILURES: list[str] = []


def ok(name: str, detail: str = "") -> None:
    global PASS
    PASS += 1
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def fail(name: str, detail: str) -> None:
    global FAIL
    FAIL += 1
    FAILURES.append(f"{name}: {detail}")
    print(f"  [FAIL] {name} — {detail}")


def warn(name: str, detail: str) -> None:
    global WARN
    WARN += 1
    print(f"  [WARN] {name} — {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    artifacts = ROOT / "artifacts"
    data = ROOT / "data"

    # ------------------------------------------------------------------
    section("1. Project files exist")
    required = [
        data / "corpus.jsonl",
        data / "queries.jsonl",
        artifacts / "faiss.index",
        artifacts / "embeddings.npy",
        artifacts / "analyzer.pt",
        artifacts / "analyzer_meta.json",
        ROOT / "app" / "streamlit_app.py",
        ROOT / "requirements.txt",
    ]
    for p in required:
        if p.exists():
            ok(f"exists:{p.relative_to(ROOT)}")
        else:
            fail(f"exists:{p.relative_to(ROOT)}", "missing")

    # ------------------------------------------------------------------
    section("2. Dataset integrity")
    from src.data_schema import load_corpus, load_queries

    docs = load_corpus(data / "corpus.jsonl")
    queries = load_queries(data / "queries.jsonl")
    if 600 <= len(docs) <= 1000:
        ok("corpus_size", str(len(docs)))
    else:
        fail("corpus_size", f"expected 600-1000, got {len(docs)}")

    if 800 <= len(queries) <= 4000:
        ok("queries_size", str(len(queries)))
    else:
        fail("queries_size", f"expected 800-4000, got {len(queries)}")

    splits = {s: sum(1 for q in queries if q.split == s) for s in ("train", "val", "test")}
    if all(splits[s] > 0 for s in splits):
        ok("splits_present", str(splits))
    else:
        fail("splits_present", str(splits))

    bad_docs = [d for d in docs if not d.doc_id or not d.text or not d.timestamp]
    if not bad_docs:
        ok("corpus_fields")
    else:
        fail("corpus_fields", f"{len(bad_docs)} invalid docs")

    bad_q = [
        q
        for q in queries
        if q.needs_retrieval not in (0, 1)
        or q.temporal_class not in ("current", "historical", "timeless")
        or not q.query
    ]
    if not bad_q:
        ok("query_fields")
    else:
        fail("query_fields", f"{len(bad_q)} invalid queries")

    # ------------------------------------------------------------------
    section("3. Index load + retrieval")
    from src.embed_index import EmbeddingIndex

    try:
        index = EmbeddingIndex()
        index.load(artifacts)
        ok("index_load", f"{len(index.docs)} docs")
    except Exception as exc:
        fail("index_load", repr(exc))
        print(traceback.format_exc())
        return 1

    if len(index.docs) != len(docs):
        # snapshot may match generated corpus; warn if mismatch
        warn("index_doc_count", f"index={len(index.docs)} corpus={len(docs)}")
    else:
        ok("index_doc_count_matches_corpus")

    hits = index.search("Who is the current Prime Minister of India?", top_k=5)
    if hits and "India" in hits[0][0].text:
        ok("retrieval_india_pm", hits[0][0].text[:80])
    else:
        fail("retrieval_india_pm", hits[0][0].text[:120] if hits else "no hits")

    hits2 = index.search("Define photosynthesis", top_k=5)
    if hits2 and "Photosynthesis" in hits2[0][0].text:
        ok("retrieval_photosynthesis", hits2[0][0].text[:80])
    else:
        fail("retrieval_photosynthesis", hits2[0][0].text[:120] if hits2 else "no hits")

    # ------------------------------------------------------------------
    section("4. Analyzer predictions")
    from src.query_analyzer import QueryAnalyzerService

    try:
        analyzer = QueryAnalyzerService.from_pretrained_bundle(artifacts)
        ok("analyzer_load")
    except Exception as exc:
        fail("analyzer_load", repr(exc))
        return 1

    cases = [
        ("Who is the current Prime Minister of India?", "current", 1),
        ("Who was the President of the United States in 2018?", "historical", 1),
        ("Define photosynthesis.", "timeless", None),  # needs may be 0 or 1
        ("What is 2 + 2?", "timeless", 0),
    ]
    for q, expect_class, expect_needs in cases:
        pred = analyzer.predict(q)
        if pred.temporal_class == expect_class:
            ok(f"analyzer_class:{expect_class}", q)
        else:
            fail(f"analyzer_class:{expect_class}", f"got {pred.temporal_class} for {q}")
        if expect_needs is not None:
            if pred.needs_retrieval == expect_needs:
                ok(f"analyzer_needs:{expect_needs}", q)
            else:
                fail(f"analyzer_needs:{expect_needs}", f"got {pred.needs_retrieval} for {q}")
        if not (0.0 <= pred.confidence <= 1.0):
            fail("analyzer_confidence_range", str(pred.confidence))

    # ------------------------------------------------------------------
    section("5. Temporal rerank sanity")
    from src.temporal_rank import rerank

    cands = index.search_candidates("Who is the current Prime Minister of India?", top_k=30)
    ranked = rerank(cands, mode="current", query="Who is the current Prime Minister of India?", top_k=5)
    if ranked and "India" in ranked[0][0].text and "Modi" in ranked[0][0].text:
        ok("rerank_current_india", ranked[0][0].text[:90])
    else:
        fail("rerank_current_india", ranked[0][0].text[:120] if ranked else "empty")

    cands_h = index.search_candidates("Who was the President of the United States in 2018?", top_k=30)
    ranked_h = rerank(
        cands_h,
        mode="historical",
        query="Who was the President of the United States in 2018?",
        time_hint="2018",
        top_k=5,
    )
    top_h = ranked_h[0][0] if ranked_h else None
    if top_h and "Trump" in top_h.text and "2018" in (top_h.timestamp + top_h.text):
        ok("rerank_historical_2018", top_h.text[:90])
    else:
        fail("rerank_historical_2018", (top_h.text[:120] if top_h else "empty"))

    # ------------------------------------------------------------------
    section("6. All pipelines (extractive)")
    from src.generate import Generator, extractive_answer, is_trivial_query
    from src.pipelines import build_all_pipelines

    gen = Generator(backend="extractive")
    pipes = build_all_pipelines(index=index, generator=gen, analyzer=analyzer)

    expectations = [
        (
            "adaptive",
            "Define photosynthesis.",
            lambda r: "process" in r.answer.lower() and "plant" in r.answer.lower() and "fallback after error" not in r.answer.lower(),
            True,
        ),
        (
            "adaptive",
            "What is 2 + 2?",
            lambda r: r.answer.strip().startswith("4") and r.did_retrieve is False,
            False,
        ),
        (
            "adaptive",
            "Who is the current Prime Minister of India?",
            lambda r: "Modi" in r.answer and "fallback after error" not in r.answer,
            True,
        ),
        (
            "adaptive",
            "Who was the President of the United States in 2018?",
            lambda r: "Trump" in r.answer,
            True,
        ),
        (
            "B2_standard_rag",
            "Define photosynthesis.",
            lambda r: "Photosynthesis" in r.answer or "process" in r.answer.lower(),
            True,
        ),
        (
            "B3_fixed_temporal",
            "Who was the President of the United States in 2018?",
            lambda r: "Trump" in r.answer,
            True,
        ),
        (
            "B1_llm_only",
            "What is 2 + 2?",
            lambda r: r.answer.strip().startswith("4") and r.did_retrieve is False,
            False,
        ),
    ]

    for system, q, pred, expect_ret in expectations:
        try:
            r = pipes[system].run(q)
            if pred(r):
                ok(f"{system}:{q[:40]}", r.answer[:100])
            else:
                fail(f"{system}:{q[:40]}", f"answer={r.answer!r} ret={r.did_retrieve}")
            if expect_ret and not r.did_retrieve:
                fail(f"{system}:expected_retrieve", q)
            if "fallback after error" in r.answer.lower():
                fail(f"{system}:dirty_fallback", r.answer)
        except Exception as exc:
            fail(f"{system}:{q[:40]}", repr(exc))
            print(traceback.format_exc())

    # ------------------------------------------------------------------
    section("7. Extractive edge cases")
    from src.data_schema import CorpusDoc

    photo_doc = CorpusDoc(
        doc_id="t1",
        text="Photosynthesis is the process by which green plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
        timestamp="2022-01-15",
        title="Photosynthesis",
        entity="Photosynthesis",
    )
    ans = extractive_answer("Define photosynthesis.", [photo_doc])
    if ans.answer.startswith("Photosynthesis is the process") and "fallback" not in ans.answer:
        ok("extractive_definition")
    else:
        fail("extractive_definition", ans.answer)

    # Must NOT return only the word Photosynthesis
    if ans.answer.strip() == "Photosynthesis":
        fail("extractive_not_name_only", ans.answer)

    ans2 = extractive_answer("What is 2 + 2?", None)
    if ans2.answer == "4":
        ok("extractive_trivial")
    else:
        fail("extractive_trivial", ans2.answer)

    if is_trivial_query("What is 2 + 2?") and not is_trivial_query("Define photosynthesis."):
        ok("trivial_detector")
    else:
        fail("trivial_detector", "logic mismatch")

    # ------------------------------------------------------------------
    section("8. API failover routing (Groq -> OpenRouter -> OpenCode)")
    import os
    from src.generate import (
        FailoverAPIGenerator,
        Generator as Gen2,
        ProviderConfig,
        _is_failover_error,
        groq_healthcheck,
        load_provider_chain,
    )

    if _is_failover_error(RuntimeError("HTTP 429 rate limit exceeded")) and _is_failover_error(
        RuntimeError("provider overloaded 503")
    ):
        ok("failover_error_detector")
    else:
        fail("failover_error_detector", "429/503 not classified as failover")

    chain = load_provider_chain()
    present = [p.name for p in chain]
    if present:
        ok("provider_chain_loaded", " -> ".join(present))
    else:
        warn("provider_chain_loaded", "no API keys — extractive only")

    for name in ("groq", "openrouter", "opencode"):
        if name in present:
            ok(f"provider_key_present:{name}")
        else:
            warn(f"provider_key_present:{name}", "not set in .env / environment")

    # Simulated routing: dead provider first, then a working one (or extractive)
    fake_dead = ProviderConfig(
        name="fake_dead",
        api_key="invalid",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
    )
    simulated = list(chain)
    # Always put a guaranteed-failing provider first to exercise skip logic
    sim_chain = [fake_dead] + simulated
    failover = FailoverAPIGenerator(sim_chain)
    routed = failover.generate("Define photosynthesis.", docs=[photo_doc])
    attempts = getattr(failover, "attempts", [])
    if attempts and attempts[0] == "fake_dead":
        ok("failover_tries_first_provider", str(attempts))
    else:
        fail("failover_tries_first_provider", str(attempts))

    if routed.backend.startswith("api:") and "process" in routed.answer.lower():
        ok("failover_reached_live_provider", routed.backend)
    elif routed.backend.startswith("extractive") and "process" in routed.answer.lower():
        # Expected when only invalid keys / network blocks all providers
        warn("failover_reached_live_provider", f"fell back to extractive after {attempts}: {routed.error[:120]}")
        ok("failover_extractive_safety_net", routed.answer[:80])
    else:
        fail("failover_routing_result", f"backend={routed.backend} answer={routed.answer[:120]}")

    # All-invalid chain must still return a clean extractive answer
    all_bad = FailoverAPIGenerator(
        [
            ProviderConfig("bad_a", "x", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
            ProviderConfig("bad_b", "y", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ]
    )
    bad_out = all_bad.generate("Define photosynthesis.", docs=[photo_doc])
    if "fallback after error" in bad_out.answer.lower() or "RuntimeError" in bad_out.answer:
        fail("api_fallback_clean", bad_out.answer)
    elif "process" in bad_out.answer.lower():
        ok("api_fallback_clean", bad_out.answer[:90])
    else:
        fail("api_fallback_clean", bad_out.answer)

    # Live healthcheck through the real chain (no keys printed)
    if chain:
        good, msg = groq_healthcheck()
        safe = str(msg)[:100].encode("ascii", "replace").decode()
        if good:
            ok("api_chain_live", safe)
            live = Gen2(backend="api")
            live_ans = live.generate("Define photosynthesis.", docs=[photo_doc])
            clean = live_ans.answer.encode("ascii", "replace").decode()
            if "process" in live_ans.answer.lower() and "fallback after error" not in live_ans.answer.lower():
                ok("api_grounded_photosynthesis", f"{live_ans.backend} | {clean[:80]}")
            else:
                fail("api_grounded_photosynthesis", clean[:160])
        else:
            warn("api_chain_live", f"unavailable (extractive fallback): {safe}")
    else:
        warn("api_chain_live", "no keys configured")

    # ------------------------------------------------------------------
    section("9. Eval smoke on 12 test queries")
    from src.metrics import contains_answer, temporal_correctness, token_f1
    from src.pipelines import AdaptiveTemporalRAGPipeline

    test_qs = [q for q in queries if q.split == "test"][:12]
    pipe = AdaptiveTemporalRAGPipeline(analyzer=analyzer, index=index, generator=gen)
    rows = []
    for q in test_qs:
        r = pipe.run(q)
        rows.append(
            {
                "contains": contains_answer(r.answer, q.gold_answer),
                "temporal": temporal_correctness(r.retrieved_docs, q),
                "f1": token_f1(r.answer, q.gold_answer),
                "dirty": "fallback after error" in r.answer.lower(),
            }
        )
    if any(row["dirty"] for row in rows):
        fail("eval_dirty_answers", "found fallback after error text")
    else:
        ok("eval_dirty_answers_clean")
    mean_temp = sum(row["temporal"] for row in rows) / len(rows)
    if mean_temp >= 0.5:
        ok("eval_temporal_reasonable", f"{mean_temp:.2f}")
    else:
        fail("eval_temporal_reasonable", f"{mean_temp:.2f}")

    # ------------------------------------------------------------------
    section("10. Streamlit app import + syntax")
    import ast
    import importlib.util

    app_path = ROOT / "app" / "streamlit_app.py"
    try:
        ast.parse(app_path.read_text(encoding="utf-8"))
        ok("streamlit_syntax")
    except SyntaxError as exc:
        fail("streamlit_syntax", str(exc))

    # Import modules used by app without launching server
    try:
        from app import streamlit_app as _app  # type: ignore

        ok("streamlit_module_path_note", "package import may fail; checking load_stack via exec path")
    except Exception:
        # expected if app is not a package; load by path
        pass

    try:
        spec = importlib.util.spec_from_file_location("streamlit_app_e2e", app_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        # Don't execute main(); just compile load
        # Executing module runs load_dotenv and imports — fine
        spec.loader.exec_module(mod)
        stack = mod.load_stack(str(artifacts), "extractive")
        if len(stack) == 4:
            ok("streamlit_load_stack")
        else:
            fail("streamlit_load_stack", f"unexpected return {len(stack)}")
        # clear streamlit cache if present
        if hasattr(mod.load_stack, "clear"):
            mod.load_stack.clear()
    except Exception as exc:
        fail("streamlit_load_stack", repr(exc))
        print(traceback.format_exc())

    # ------------------------------------------------------------------
    section("11. Scripts CLI dry checks")
    for script in [
        "build_dataset.py",
        "build_index.py",
        "train_analyzer.py",
        "evaluate.py",
        "plot_results.py",
        "e2e_smoke_test.py",
    ]:
        p = ROOT / "scripts" / script
        if p.exists():
            try:
                ast.parse(p.read_text(encoding="utf-8"))
                ok(f"script_syntax:{script}")
            except SyntaxError as exc:
                fail(f"script_syntax:{script}", str(exc))
        else:
            fail(f"script_syntax:{script}", "missing")

    # ------------------------------------------------------------------
    section("12. Adaptive window + evidence sufficiency feedback loop")
    from src.evidence_sufficiency import assess_evidence_sufficiency
    from src.temporal_rank import infer_adaptive_window

    w_year = infer_adaptive_window("Who was PM in 2018?")
    if w_year.reason == "year" and w_year.initial_years == 1.0:
        ok("adaptive_window_year", str(w_year.as_meta()))
    else:
        fail("adaptive_window_year", str(w_year.as_meta()))

    w_date = infer_adaptive_window("As of 2022-01-20, who was the Chief Justice of India?")
    if w_date.reason == "exact_date":
        ok("adaptive_window_exact_date", str(w_date.as_meta()))
    else:
        fail("adaptive_window_exact_date", str(w_date.as_meta()))

    empty_suff = assess_evidence_sufficiency("Who was president in 2018?", [], temporal_class="historical")
    if (not empty_suff.sufficient) and "widen_window" in empty_suff.suggested_repairs:
        ok("sufficiency_flags_empty_evidence", str(empty_suff.suggested_repairs))
    else:
        fail("sufficiency_flags_empty_evidence", str(empty_suff.as_meta()))

    # Force a weak first pass so the loop must repair (tiny candidate pool)
    from src.pipelines import AdaptiveTemporalRAGPipeline

    loop_pipe = AdaptiveTemporalRAGPipeline(
        analyzer=analyzer,
        index=index,
        generator=gen,
        top_k=3,
        candidate_k=5,  # intentionally tiny → often insufficient → expand
        enable_sufficiency_loop=True,
        max_sufficiency_rounds=2,
        sufficiency_threshold=0.75,  # stricter so loop is likely to fire
    )
    hard_q = "Who was the Chancellor of Germany in 2015?"
    loop_res = loop_pipe.run(hard_q)
    slog = loop_res.meta.get("sufficiency_loop") or []
    if not slog:
        fail("sufficiency_loop_present", "missing sufficiency_loop meta")
    else:
        ok("sufficiency_loop_present", f"{len(slog)} round record(s)")

    repairs_seen = []
    for row in slog:
        repairs_seen.extend(row.get("repairs_applied") or [])
    # Either evidence was already good (1 round, no repairs) OR repairs fired
    if len(slog) >= 2 or repairs_seen or (loop_res.meta.get("evidence_sufficiency") or {}).get("sufficient"):
        ok(
            "sufficiency_loop_behavior",
            f"rounds={len(slog)} repairs={repairs_seen} final={loop_res.meta.get('evidence_sufficiency')}",
        )
    else:
        fail("sufficiency_loop_behavior", str(loop_res.meta)[:240])

    # Good query should usually be sufficient without dirty answers
    good = AdaptiveTemporalRAGPipeline(
        analyzer=analyzer, index=index, generator=gen, enable_sufficiency_loop=True
    ).run("Who was the President of the United States in 2018?")
    es = good.meta.get("evidence_sufficiency") or {}
    if good.meta.get("adaptive_window") and "Trump" in good.answer:
        ok("sufficiency_with_adaptive_window", f"window={good.meta.get('adaptive_window')} suff={es}")
    else:
        fail("sufficiency_with_adaptive_window", f"answer={good.answer!r} meta_keys={list(good.meta)}")

    # ------------------------------------------------------------------
    section("13. Colab E2E notebook sanity")
    nb_path = ROOT / "notebooks" / "00_colab_e2e.ipynb"
    if not nb_path.exists():
        fail("colab_e2e_exists", "missing notebooks/00_colab_e2e.ipynb")
    else:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        src_all = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        ok("colab_e2e_exists", f"{len(nb.get('cells', []))} cells")
        for needle, label in [
            ("TARGET_QUERIES", "has_train_knobs"),
            ("LLM_PROVIDER_ORDER", "has_provider_order"),
            ("OPENROUTER_API_KEY", "has_openrouter"),
            ("OPENCODE_API_KEY", "has_opencode"),
            ("evidence_sufficiency", "has_sufficiency_ui"),
            ("adaptive_window", "has_adaptive_window_ui"),
            ("train_analyzer", "has_training"),
            ("plot_results", "has_eval_plots"),
            ("files.download", "has_artifact_download"),
        ]:
            if needle in src_all:
                ok(f"colab_e2e:{label}")
            else:
                fail(f"colab_e2e:{label}", f"missing `{needle}` in notebook")

    # ------------------------------------------------------------------
    section("SUMMARY")
    print(f"PASS={PASS} FAIL={FAIL} WARN={WARN}")
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print(" -", f)
    out = {
        "pass": PASS,
        "fail": FAIL,
        "warn": WARN,
        "failures": FAILURES,
    }
    (artifacts / "e2e_smoke_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {artifacts / 'e2e_smoke_report.json'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
