"""Plot comparative evaluation charts from results.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comparative_analysis import (  # noqa: E402
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    build_analysis_bundle,
    load_results,
    write_comparison_json,
    write_comparison_markdown,
)

COLORS = {
    "B1_llm_only": "#8d99ae",
    "B2_standard_rag": "#2a9d8f",
    "B3_fixed_temporal": "#e9c46a",
    "adaptive": "#e76f51",
}


def _systems(df: pd.DataFrame) -> list[str]:
    present = set(df["system"].unique())
    return [s for s in SYSTEM_ORDER if s in present]


def plot_comparative(df: pd.DataFrame, out: Path) -> None:
    systems = _systems(df)
    colors = [COLORS.get(s, "#457b9d") for s in systems]
    overall = (
        df.groupby("system")
        .agg(
            contains=("contains", "mean"),
            em=("em", "mean"),
            token_f1=("token_f1", "mean"),
            temporal_acc=("temporal_acc", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            retrieval_rate=("did_retrieve", "mean"),
        )
        .reindex(systems)
    )

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Panel 1: grouped answer + temporal metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["contains", "em", "token_f1", "temporal_acc"]
    metric_names = ["Contains", "EM", "Token F1", "Temp Acc"]
    x = np.arange(len(metrics))
    width = 0.18
    for i, sys in enumerate(systems):
        vals = [overall.loc[sys, m] for m in metrics]
        ax1.bar(x + (i - 1.5) * width, vals, width, label=SYSTEM_LABELS.get(sys, sys), color=COLORS.get(sys))
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Overall quality metrics")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: contains by temporal class
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ["current", "historical", "timeless"]
    by_class = df.groupby(["system", "temporal_class"])["contains"].mean()
    x = np.arange(len(classes))
    for i, sys in enumerate(systems):
        vals = [float(by_class.get((sys, c), np.nan)) for c in classes]
        ax2.bar(x + (i - 1.5) * width, vals, width, label=SYSTEM_LABELS.get(sys, sys), color=COLORS.get(sys))
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.title() for c in classes])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Contains")
    ax2.set_title("Answer quality by temporal class")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    # Panel 3: temporal accuracy by class
    ax3 = fig.add_subplot(gs[1, 0])
    by_temp = df.groupby(["system", "temporal_class"])["temporal_acc"].mean()
    for i, sys in enumerate(systems):
        vals = [float(by_temp.get((sys, c), np.nan)) for c in classes]
        ax3.bar(x + (i - 1.5) * width, vals, width, label=SYSTEM_LABELS.get(sys, sys), color=COLORS.get(sys))
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.title() for c in classes])
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Temporal accuracy")
    ax3.set_title("Temporal accuracy by class")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(axis="y", alpha=0.25)

    # Panel 4: efficiency — retrieval rate vs contains
    ax4 = fig.add_subplot(gs[1, 1])
    for sys, color in zip(systems, colors):
        ax4.scatter(
            overall.loc[sys, "retrieval_rate"],
            overall.loc[sys, "contains"],
            s=160,
            color=color,
            zorder=3,
            label=SYSTEM_LABELS.get(sys, sys),
        )
        ax4.annotate(
            SYSTEM_LABELS.get(sys, sys),
            (overall.loc[sys, "retrieval_rate"], overall.loc[sys, "contains"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    ax4.set_xlim(-0.05, 1.1)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("Retrieval rate (lower = fewer calls)")
    ax4.set_ylabel("Contains (higher = better answers)")
    ax4.set_title("Efficiency frontier: quality vs retrieval cost")
    ax4.grid(alpha=0.25)

    fig.suptitle("Comparative Analysis — Adaptive Temporal RAG vs Baselines", fontsize=13, y=0.98)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default=str(ROOT / "artifacts" / "results.csv"))
    p.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "results_charts.png"))
    p.add_argument("--report", type=str, default=str(ROOT / "artifacts" / "comparative_analysis.md"))
    p.add_argument("--json", type=str, default=str(ROOT / "artifacts" / "comparative_analysis.json"))
    args = p.parse_args()

    df = load_results(args.results)
    plot_comparative(df, Path(args.out))

    bundle = build_analysis_bundle(df)
    md_path = write_comparison_markdown(bundle, args.report)
    json_path = write_comparison_json(bundle, args.json)
    print("Saved", md_path)
    print("Saved", json_path)

    print("\n=== Comparative highlights ===")
    for h in bundle["highlights"]:
        print("-", h)


if __name__ == "__main__":
    main()
