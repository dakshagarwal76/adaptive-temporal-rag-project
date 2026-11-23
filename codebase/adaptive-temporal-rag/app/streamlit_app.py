"""Streamlit demo for Adaptive Temporal RAG (Colab or local)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comparative_analysis import (  # noqa: E402
    DISPLAY_METRICS,
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    build_analysis_bundle,
    load_results,
)
from src.embed_index import EmbeddingIndex
from src.env_loader import load_dotenv
from src.generate import Generator
from src.hf_utils import prefer_local_hf_cache
from src.config import load_eval_config
from src.pipelines import build_all_pipelines
from src.query_analyzer import load_query_analyzer

load_dotenv(ROOT / ".env")
prefer_local_hf_cache()

SYSTEM_COLORS = {
    "B1_llm_only": "#8d99ae",
    "B2_standard_rag": "#2a9d8f",
    "B3_fixed_temporal": "#e9c46a",
    "adaptive": "#e76f51",
}


@st.cache_resource
def load_stack(artifacts_dir: str, backend: str):
    artifacts = Path(artifacts_dir)
    index = EmbeddingIndex()
    index.load(artifacts)
    analyzer = load_query_analyzer(artifacts)
    meta_path = artifacts / "analyzer_meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        analyzer_name = meta.get("model_name", "trained query analyzer")
    else:
        analyzer_name = "trained query analyzer"
    generator = Generator(backend=backend)
    return index, analyzer, analyzer_name, generator


@st.cache_data
def load_eval_bundle(results_path: str):
    path = Path(results_path)
    if not path.exists():
        return None
    df = load_results(path)
    return df, build_analysis_bundle(df)


def _style_fig(fig):
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_live_demo(artifacts: str) -> None:
    cfg = load_eval_config(ROOT / "configs" / "final.yaml")
    default_backend = cfg.generator if cfg.generator in ("api", "extractive", "hf") else "extractive"
    backend = st.sidebar.selectbox(
        "Answer generator",
        ["extractive", "api", "hf"],
        index=["extractive", "api", "hf"].index(default_backend),
        help="Frozen default is configs/final.yaml. extractive = offline from docs.",
    )
    system = st.sidebar.selectbox(
        "System",
        [
            "adaptive",
            "adaptive_fast",
            "B1_llm_only",
            "B2_standard_rag",
            "B3_fixed_temporal",
            "tempralm_decay",
        ],
    )
    st.sidebar.caption(
        f"Frozen config {cfg.name}: top_k={cfg.top_k}, alpha={cfg.alpha}, "
        f"conf={cfg.conf_threshold}, sufficiency={cfg.enable_sufficiency_loop}, "
        f"window={cfg.enable_adaptive_window}, entity cut={cfg.enable_entity_cut}, "
        f"sentence selector={cfg.enable_sentence_selector}, projection={cfg.enable_answer_projection}."
    )

    try:
        index, analyzer, analyzer_name, generator = load_stack(artifacts, backend)
        generator.use_sentence_selector = cfg.enable_sentence_selector
        st.sidebar.success(f"Index ready ({len(index.docs)} docs)")
        st.sidebar.info(f"Analyzer: {analyzer_name}")
        if backend == "api":
            from src.generate import load_provider_chain

            chain = load_provider_chain()
            if chain:
                st.sidebar.info("API failover: " + " → ".join(p.name for p in chain))
                st.sidebar.caption("On rate-limit/outage the next provider is tried; then extractive.")
            else:
                st.sidebar.warning("No GROQ / OPENROUTER / OPENCODE keys — extractive fallback")
        else:
            st.sidebar.success("Using offline extractive answers")
    except Exception as exc:
        st.error(f"Failed to load artifacts from `{artifacts}`: {exc}")
        st.info("Run `python scripts/build_index.py` first (and optionally train the analyzer).")
        return

    query = st.text_input(
        "Ask a time-sensitive question",
        "Define photosynthesis.",
    )
    run = st.button("Run", type="primary")

    if not run or not query.strip():
        st.markdown(
            """
            **Try:**
            - Who is the current Prime Minister of India?
            - Who was the President of the United States in 2018?
            - Define photosynthesis.
            - What is 2 + 2?
            """
        )
        return

    live_cfg = cfg
    if system == "adaptive_fast":
        from dataclasses import replace

        live_cfg = replace(cfg, enable_sufficiency_loop=False)
    pipes = build_all_pipelines(index=index, generator=generator, analyzer=analyzer, config=live_cfg)
    pipe_name = "adaptive" if system == "adaptive_fast" else system
    pipe = pipes[pipe_name]

    result = pipe.run(query)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Answer")
        st.write(result.answer)
        st.caption(f"Latency: {result.latency_s:.3f}s · Backend: {result.backend}")
        if result.meta.get("generator_error"):
            st.warning("All API providers failed or hit limits — used offline extractive answer.")
            with st.expander("API error details"):
                st.code(result.meta["generator_error"])
    with c2:
        st.subheader("Analyzer")
        if result.analyzer is None and system == "adaptive":
            st.write("No analyzer output")
        elif result.analyzer is not None:
            payload = {
                "needs_retrieval": result.analyzer.needs_retrieval,
                "temporal_class": result.analyzer.temporal_class,
                "confidence": round(result.analyzer.confidence, 3),
                "necessity_prob": round(result.analyzer.necessity_prob, 3),
                "temporal_probs": {k: round(v, 3) for k, v in result.analyzer.temporal_probs.items()},
                "policy": result.meta.get("policy"),
            }
            if result.meta.get("adaptive_window"):
                payload["adaptive_window"] = result.meta["adaptive_window"]
            if result.meta.get("evidence_sufficiency"):
                payload["evidence_sufficiency"] = result.meta["evidence_sufficiency"]
            st.json(payload)
            if result.meta.get("sufficiency_loop"):
                with st.expander("Evidence sufficiency feedback loop"):
                    st.json(result.meta["sufficiency_loop"])
        else:
            st.write("Analyzer used only by Adaptive system.")
            if result.meta:
                st.json({k: v for k, v in result.meta.items() if k != "generator_error"})

    st.subheader("Retrieved evidence")
    if not result.retrieved_docs:
        st.write("_No retrieval_")
    else:
        for i, (doc, score) in enumerate(zip(result.retrieved_docs, result.retrieved_scores), start=1):
            with st.expander(f"[{i}] {doc.timestamp} · score={score:.3f} · {doc.title or doc.entity}"):
                st.write(doc.text)
                st.caption(f"doc_id={doc.doc_id} · entity={doc.entity}")


def render_comparative_analysis(artifacts: str) -> None:
    artifacts_dir = Path(artifacts)
    final_md = artifacts_dir / "final_results.md"
    frozen_csv = artifacts_dir / "general_eval" / "final" / "synthetic" / "results.csv"
    results_path = frozen_csv if frozen_csv.exists() else artifacts_dir / "results.csv"
    charts_path = artifacts_dir / "results_charts.png"
    report_path = final_md if final_md.exists() else artifacts_dir / "comparative_analysis.md"

    st.subheader("How Adaptive compares to existing systems")
    st.caption(
        "Frozen Checkpoint B (`configs/final.yaml`). Numbers come from `artifacts/final_results.md`. "
        "B1/B2/B3/Adaptive share the reader, answer projection, and entity/surname cut."
    )
    if final_md.exists():
        st.markdown(final_md.read_text(encoding="utf-8"))
        st.markdown("---")

    loaded = load_eval_bundle(str(results_path))
    if loaded is None:
        st.warning("No evaluation results found yet.")
        st.code(
            "python scripts/evaluate.py --split test --generator extractive\n"
            "python scripts/plot_results.py"
        )
        return

    df, bundle = loaded
    overall = bundle["overall"].copy()
    overall["System"] = overall["system"].map(SYSTEM_LABELS)

    # KPI strip vs strongest practical baseline (B2) and always-on temporal (B3)
    o = overall.set_index("system")
    if "adaptive" in o.index:
        a = o.loc["adaptive"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Adaptive Contains", f"{a['contains']:.1%}")
        if "B2_standard_rag" in o.index:
            k2.metric(
                "vs Standard RAG",
                f"{a['contains'] - o.loc['B2_standard_rag', 'contains']:+.1%}",
                help="Absolute gain in answer-contains rate",
            )
        if "B3_fixed_temporal" in o.index:
            k3.metric(
                "vs Fixed Temporal",
                f"{a['contains'] - o.loc['B3_fixed_temporal', 'contains']:+.1%}",
                help="Absolute gap vs always-on temporal RAG",
            )
        k4.metric("Retrieval rate", f"{a['retrieval_rate']:.0%}", help="Fraction of queries that trigger retrieval")

    for h in bundle["highlights"]:
        st.markdown(f"- {h}")

    st.markdown("---")
    st.markdown("#### Overall scoreboard")
    show_cols = [
        "System",
        "n",
        "contains",
        "em",
        "token_f1",
        "temporal_acc",
        "precision_at_k",
        "recall_at_k",
        "retrieval_rate",
        "latency_s",
    ]
    show = overall[[c for c in show_cols if c in overall.columns]].copy()
    for c in show.columns:
        if c in ("System", "n"):
            continue
        if c == "latency_s":
            show[c] = show[c].map(lambda x: round(float(x), 3))
        else:
            show[c] = show[c].map(lambda x: round(float(x), 3))
    show = show.rename(
        columns={
            "n": "N",
            "contains": "Contains",
            "em": "EM",
            "token_f1": "Token F1",
            "temporal_acc": "Temp Acc",
            "precision_at_k": "P@5",
            "recall_at_k": "R@5",
            "retrieval_rate": "Ret Rate",
            "latency_s": "Latency (s)",
        }
    )
    st.dataframe(show, width="stretch", hide_index=True)

    # Grouped bar — overall metrics
    melt_metrics = ["contains", "em", "token_f1", "temporal_acc", "precision_at_k", "recall_at_k"]
    long = overall.melt(
        id_vars=["system", "System"],
        value_vars=[m for m in melt_metrics if m in overall.columns],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(DISPLAY_METRICS)
    fig = px.bar(
        long,
        x="metric",
        y="score",
        color="System",
        barmode="group",
        color_discrete_map={SYSTEM_LABELS[k]: v for k, v in SYSTEM_COLORS.items() if k in SYSTEM_LABELS},
        range_y=[0, 1.05],
        title="Overall metrics by system",
    )
    st.plotly_chart(_style_fig(fig), width="stretch")

    c1, c2 = st.columns(2)
    by_class = bundle["by_class"].copy()
    by_class["System"] = by_class["label"]

    with c1:
        fig_c = px.bar(
            by_class,
            x="temporal_class",
            y="contains",
            color="System",
            barmode="group",
            range_y=[0, 1.05],
            title="Contains by temporal class",
            category_orders={"temporal_class": ["current", "historical", "timeless"]},
            color_discrete_map={SYSTEM_LABELS[k]: v for k, v in SYSTEM_COLORS.items()},
        )
        st.plotly_chart(_style_fig(fig_c), width="stretch")
    with c2:
        fig_t = px.bar(
            by_class,
            x="temporal_class",
            y="temporal_acc",
            color="System",
            barmode="group",
            range_y=[0, 1.05],
            title="Temporal accuracy by class",
            category_orders={"temporal_class": ["current", "historical", "timeless"]},
            color_discrete_map={SYSTEM_LABELS[k]: v for k, v in SYSTEM_COLORS.items()},
        )
        st.plotly_chart(_style_fig(fig_t), width="stretch")

    # Efficiency scatter
    fig_e = go.Figure()
    for sys in SYSTEM_ORDER:
        if sys not in set(overall["system"]):
            continue
        row = overall[overall["system"] == sys].iloc[0]
        fig_e.add_trace(
            go.Scatter(
                x=[row["retrieval_rate"]],
                y=[row["contains"]],
                mode="markers+text",
                marker=dict(size=18, color=SYSTEM_COLORS.get(sys, "#457b9d")),
                text=[SYSTEM_LABELS.get(sys, sys)],
                textposition="top center",
                name=SYSTEM_LABELS.get(sys, sys),
            )
        )
    fig_e.update_layout(
        title="Efficiency frontier: answer quality vs retrieval cost",
        xaxis_title="Retrieval rate (lower is cheaper)",
        yaxis_title="Contains (higher is better)",
        xaxis=dict(range=[-0.05, 1.1]),
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(_style_fig(fig_e), width="stretch")

    # Deltas table
    deltas = bundle["deltas"]
    if not deltas.empty:
        st.markdown("#### Adaptive − baseline (absolute deltas)")
        dshow = deltas[
            ["baseline_label", "delta_contains", "delta_em", "delta_token_f1", "delta_temporal_acc", "delta_retrieval_rate"]
        ].copy()
        dshow.columns = ["Baseline", "Δ Contains", "Δ EM", "Δ Token F1", "Δ Temp Acc", "Δ Ret Rate"]
        for c in dshow.columns:
            if c.startswith("Δ"):
                dshow[c] = dshow[c].map(lambda x: f"{x:+.3f}")
        st.dataframe(dshow, width="stretch", hide_index=True)

    if charts_path.exists():
        st.markdown("#### Static chart export")
        st.image(str(charts_path), width="stretch")

    if report_path.exists() and report_path != final_md:
        with st.expander("Full comparative report (markdown)"):
            st.markdown(report_path.read_text(encoding="utf-8"))

    st.info(
        "Quote `artifacts/final_results.md` for the paper and the report. "
        "Do not reuse older 16-question slices or Checkpoint A files."
    )


def main() -> None:
    st.set_page_config(page_title="Adaptive Temporal RAG", layout="wide")
    st.title("Adaptive Temporal RAG")
    st.caption("When and what to retrieve for time-sensitive question answering")

    artifacts = st.sidebar.text_input("Artifacts directory", str(ROOT / "artifacts"))
    page = st.sidebar.radio("Page", ["Live demo", "Comparative analysis"], index=0)

    if page == "Live demo":
        render_live_demo(artifacts)
    else:
        render_comparative_analysis(artifacts)


if __name__ == "__main__":
    main()
