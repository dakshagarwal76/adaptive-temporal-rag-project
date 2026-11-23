"""Comparative analysis helpers for B1 / B2 / B3 / Adaptive evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEM_ORDER = ["B1_llm_only", "B2_standard_rag", "B3_fixed_temporal", "adaptive"]

SYSTEM_LABELS = {
    "B1_llm_only": "B1 LLM-only",
    "B2_standard_rag": "B2 Standard RAG",
    "B3_fixed_temporal": "B3 Fixed Temporal",
    "adaptive": "Adaptive (ours)",
}

METRIC_COLS = [
    "contains",
    "em",
    "token_f1",
    "temporal_acc",
    "precision_at_k",
    "recall_at_k",
    "retrieval_rate",
    "latency_s",
]

DISPLAY_METRICS = {
    "contains": "Contains (answer quality)",
    "em": "Exact Match",
    "token_f1": "Token F1",
    "temporal_acc": "Temporal Accuracy",
    "precision_at_k": "Precision@5",
    "recall_at_k": "Recall@5",
    "retrieval_rate": "Retrieval Rate",
    "latency_s": "Latency (s)",
}


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "did_retrieve" in df.columns and "retrieval_rate" not in df.columns:
        df = df.copy()
        df["retrieval_rate"] = df["did_retrieve"]
    return df


def _ordered_systems(systems: list[str]) -> list[str]:
    seen = set(systems)
    ordered = [s for s in SYSTEM_ORDER if s in seen]
    ordered.extend(sorted(s for s in systems if s not in SYSTEM_ORDER))
    return ordered


def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per system with mean metrics."""
    work = df.copy()
    if "retrieval_rate" not in work.columns and "did_retrieve" in work.columns:
        work["retrieval_rate"] = work["did_retrieve"]

    agg = {m: "mean" for m in METRIC_COLS if m in work.columns}
    agg["query_id"] = "count"
    table = work.groupby("system", as_index=False).agg(agg).rename(columns={"query_id": "n"})
    table["label"] = table["system"].map(SYSTEM_LABELS).fillna(table["system"])
    table = table.set_index("system").loc[_ordered_systems(table["system"].tolist())].reset_index()
    return table


def by_temporal_class(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ["contains", "em", "token_f1", "temporal_acc", "precision_at_k", "recall_at_k"] if m in df.columns]
    table = (
        df.groupby(["system", "temporal_class"], as_index=False)[metrics]
        .mean()
    )
    table["label"] = table["system"].map(SYSTEM_LABELS).fillna(table["system"])
    sys_order = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    class_order = {"current": 0, "historical": 1, "timeless": 2}
    table["_s"] = table["system"].map(sys_order).fillna(99)
    table["_c"] = table["temporal_class"].map(class_order).fillna(99)
    table = table.sort_values(["_s", "_c"]).drop(columns=["_s", "_c"]).reset_index(drop=True)
    return table


def by_retrieval_need(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "retrieval_rate" not in work.columns and "did_retrieve" in work.columns:
        work["retrieval_rate"] = work["did_retrieve"]
    metrics = [m for m in ["contains", "em", "token_f1", "temporal_acc", "retrieval_rate"] if m in work.columns]
    table = work.groupby(["system", "needs_retrieval"], as_index=False)[metrics].mean()
    table["label"] = table["system"].map(SYSTEM_LABELS).fillna(table["system"])
    table["needs_retrieval_label"] = table["needs_retrieval"].map({0: "No retrieval needed", 1: "Needs retrieval"})
    return table


def deltas_vs_baselines(overall: pd.DataFrame, proposed: str = "adaptive") -> pd.DataFrame:
    """Absolute deltas of proposed system vs each baseline on key metrics."""
    if proposed not in set(overall["system"]):
        return pd.DataFrame()
    prop = overall.set_index("system").loc[proposed]
    rows = []
    for baseline in _ordered_systems([s for s in overall["system"] if s != proposed]):
        base = overall.set_index("system").loc[baseline]
        row: dict[str, Any] = {
            "baseline": baseline,
            "baseline_label": SYSTEM_LABELS.get(baseline, baseline),
            "proposed": proposed,
        }
        for m in METRIC_COLS:
            if m in overall.columns:
                row[f"delta_{m}"] = float(prop[m] - base[m])
                row[f"baseline_{m}"] = float(base[m])
                row[f"proposed_{m}"] = float(prop[m])
        rows.append(row)
    return pd.DataFrame(rows)


def win_rate_vs(df: pd.DataFrame, proposed: str = "adaptive", baseline: str = "B2_standard_rag", metric: str = "contains") -> dict[str, float]:
    """Per-query win / tie / loss rate of proposed vs one baseline."""
    a = df[df["system"] == proposed][["query_id", metric]].rename(columns={metric: "prop"})
    b = df[df["system"] == baseline][["query_id", metric]].rename(columns={metric: "base"})
    merged = a.merge(b, on="query_id")
    if merged.empty:
        return {"n": 0.0, "win": 0.0, "tie": 0.0, "loss": 0.0}
    win = (merged["prop"] > merged["base"]).mean()
    tie = (merged["prop"] == merged["base"]).mean()
    loss = (merged["prop"] < merged["base"]).mean()
    return {"n": float(len(merged)), "win": float(win), "tie": float(tie), "loss": float(loss)}


def head_to_head_matrix(df: pd.DataFrame, metric: str = "contains") -> pd.DataFrame:
    """Pairwise win rates (row beats column) on a metric."""
    systems = _ordered_systems(df["system"].unique().tolist())
    data = []
    for row_sys in systems:
        row = {"system": row_sys, "label": SYSTEM_LABELS.get(row_sys, row_sys)}
        for col_sys in systems:
            if row_sys == col_sys:
                row[col_sys] = float("nan")
            else:
                stats = win_rate_vs(df, proposed=row_sys, baseline=col_sys, metric=metric)
                row[col_sys] = stats["win"]
        data.append(row)
    return pd.DataFrame(data)


def build_analysis_bundle(df: pd.DataFrame) -> dict[str, Any]:
    overall = overall_table(df)
    by_class = by_temporal_class(df)
    by_need = by_retrieval_need(df)
    deltas = deltas_vs_baselines(overall)
    hh_contains = head_to_head_matrix(df, "contains")
    hh_temporal = head_to_head_matrix(df, "temporal_acc")

    highlights = []
    if "adaptive" in set(overall["system"]):
        o = overall.set_index("system")
        adapt = o.loc["adaptive"]
        if "B2_standard_rag" in o.index:
            b2 = o.loc["B2_standard_rag"]
            highlights.append(
                f"Adaptive improves answer contains by {(adapt['contains'] - b2['contains']):+.1%} "
                f"and temporal accuracy by {(adapt['temporal_acc'] - b2['temporal_acc']):+.1%} vs Standard RAG."
            )
            highlights.append(
                f"Adaptive retrieves on {adapt['retrieval_rate']:.0%} of queries vs 100% for Standard/Fixed RAG "
                f"(saves ~{(1 - adapt['retrieval_rate']):.0%} retrieval calls)."
            )
        if "B3_fixed_temporal" in o.index:
            b3 = o.loc["B3_fixed_temporal"]
            highlights.append(
                f"vs Fixed Temporal: temporal accuracy gap is {(adapt['temporal_acc'] - b3['temporal_acc']):+.1%}; "
                f"contains gap is {(adapt['contains'] - b3['contains']):+.1%} "
                f"(often from skipping retrieval on timeless items under extractive eval)."
            )
        wr = win_rate_vs(df, "adaptive", "B2_standard_rag", "contains")
        highlights.append(
            f"Per-query vs Standard RAG (contains): win {wr['win']:.0%} / tie {wr['tie']:.0%} / loss {wr['loss']:.0%}."
        )

    return {
        "overall": overall,
        "by_class": by_class,
        "by_need": by_need,
        "deltas": deltas,
        "head_to_head_contains": hh_contains,
        "head_to_head_temporal": hh_temporal,
        "highlights": highlights,
        "n_queries": int(df["query_id"].nunique()) if "query_id" in df.columns else len(df) // max(df["system"].nunique(), 1),
    }


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no tabulate dep)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows])


def format_overall_markdown(overall: pd.DataFrame) -> str:
    cols = ["label", "n", "contains", "em", "token_f1", "temporal_acc", "precision_at_k", "recall_at_k", "retrieval_rate", "latency_s"]
    cols = [c for c in cols if c in overall.columns]
    show = overall[cols].copy()
    for c in show.columns:
        if c in ("label",):
            continue
        if c == "n":
            show[c] = show[c].astype(int)
        elif c == "latency_s":
            show[c] = show[c].map(lambda x: f"{x:.3f}")
        else:
            show[c] = show[c].map(lambda x: f"{x:.3f}")
    headers = {
        "label": "System",
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
    show = show.rename(columns=headers)
    return _md_table(show)


def write_comparison_markdown(bundle: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    lines = [
        "# Comparative Analysis: Adaptive Temporal RAG vs Baselines",
        "",
        f"Evaluated on **{bundle['n_queries']}** held-out test queries (same split for all systems).",
        "",
        "## Systems compared",
        "",
        "| ID | System | Behavior |",
        "|----|--------|----------|",
        "| B1 | LLM-only | Never retrieves |",
        "| B2 | Standard RAG | Always retrieves (semantic only) |",
        "| B3 | Fixed Temporal RAG | Always retrieves with fixed time policy |",
        "| Ours | Adaptive Temporal RAG | Conditional retrieval + class-specific temporal policy |",
        "",
        "## Overall results",
        "",
        format_overall_markdown(bundle["overall"]),
        "",
        "## Key takeaways",
        "",
    ]
    for h in bundle["highlights"]:
        lines.append(f"- {h}")
    lines.extend(["", "## Results by temporal class (Contains)", ""])

    by_class = bundle["by_class"]
    pivot = by_class.pivot(index="label", columns="temporal_class", values="contains")
    for col in ["current", "historical", "timeless"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot = pivot[["current", "historical", "timeless"]].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    lines.append(_md_table(pivot.reset_index().rename(columns={"label": "System"})))
    lines.extend(["", "## Results by temporal class (Temporal accuracy)", ""])
    pivot_t = by_class.pivot(index="label", columns="temporal_class", values="temporal_acc")
    for col in ["current", "historical", "timeless"]:
        if col not in pivot_t.columns:
            pivot_t[col] = float("nan")
    pivot_t = pivot_t[["current", "historical", "timeless"]].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    lines.append(_md_table(pivot_t.reset_index().rename(columns={"label": "System"})))

    deltas = bundle["deltas"]
    if not deltas.empty:
        lines.extend(["", "## Deltas: Adaptive − baseline", ""])
        dshow = deltas[
            ["baseline_label", "delta_contains", "delta_temporal_acc", "delta_retrieval_rate", "delta_token_f1"]
        ].copy()
        for c in dshow.columns:
            if c.startswith("delta_"):
                dshow[c] = dshow[c].map(lambda x: f"{x:+.3f}")
        dshow = dshow.rename(
            columns={
                "baseline_label": "Baseline",
                "delta_contains": "Δ Contains",
                "delta_temporal_acc": "Δ Temp Acc",
                "delta_retrieval_rate": "Δ Ret Rate",
                "delta_token_f1": "Δ Token F1",
            }
        )
        lines.append(_md_table(dshow))

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- **B1** is weak on time-sensitive facts without a parametric LLM answerer (extractive stub returns little).",
            "- **B2** often ranks stale passages highly for “current” questions → lower temporal accuracy.",
            "- **B3** is a strong always-on temporal upper bound but retrieves on every query.",
            "- **Adaptive** matches B3 on *current* / *historical* answer quality while skipping many timeless/no-need queries; "
            "under extractive eval, skipped retrieval lowers timeless *contains* (re-run with `hf`/`api` generator for fairer QA scores).",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_comparison_json(bundle: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)

    def _df(d: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(d.to_json(orient="records"))

    payload = {
        "n_queries": bundle["n_queries"],
        "highlights": bundle["highlights"],
        "overall": _df(bundle["overall"]),
        "by_class": _df(bundle["by_class"]),
        "by_need": _df(bundle["by_need"]),
        "deltas": _df(bundle["deltas"]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
