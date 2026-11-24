"""Rescore the 18 November 2026 full-test predictions after answer projection.

Reads the saved per-query CSVs. Does not retrieve, embed, or call a model.
B2 and B3 (and B1) predictions are passed through src.answer_projection.project_answer.
Adaptive predictions are left as saved, because that run already projected them.
Exact match, token F1, and containment are recomputed with src.metrics on the
same gold strings and unanswerable rule as scripts/evaluate.py, without the
later TimeQA span extractor.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.answer_projection import project_answer
from src.data_schema import load_queries
from src.metrics import (
    abstention_correct,
    best_contains,
    best_exact_match,
    best_token_f1,
    gold_strings,
)

DATASETS = {
    "synthetic": {
        "csv": ROOT / "artifacts" / "general_eval" / "synthetic_full_test" / "results.csv",
        "summary": ROOT / "artifacts" / "general_eval" / "synthetic_full_test" / "results_summary.json",
        "queries": ROOT / "data" / "queries.jsonl",
    },
    "heldout": {
        "csv": ROOT / "artifacts" / "general_eval" / "heldout_full_test" / "results.csv",
        "summary": ROOT / "artifacts" / "general_eval" / "heldout_full_test" / "results_summary.json",
        "queries": ROOT / "data" / "heldout" / "queries.jsonl",
    },
    "timeqa": {
        "csv": ROOT / "artifacts" / "general_eval" / "timeqa_full_test" / "results.csv",
        "summary": ROOT / "artifacts" / "general_eval" / "timeqa_full_test" / "results_summary.json",
        "queries": ROOT / "data" / "general" / "timeqa" / "queries.jsonl",
    },
}

PROJECT_SYSTEMS = ("B1_llm_only", "B2_standard_rag", "B3_fixed_temporal")


def _score(pred: str, query) -> dict[str, float]:
    golds = gold_strings(query)
    unanswerable = bool(query.unanswerable) or not golds
    if unanswerable:
        bit = float(abstention_correct(pred, True))
        return {"em": bit, "token_f1": bit, "contains": bit, "unanswerable": 1.0}
    return {
        "em": best_exact_match(pred, golds),
        "token_f1": best_token_f1(pred, golds),
        "contains": best_contains(pred, golds),
        "unanswerable": 0.0,
    }


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    ordered = sorted(vals)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    report: dict = {
        "method": (
            "Post-hoc project_answer on saved 18 November 2026 predictions. "
            "No retrieval rerun. Adaptive rows are unchanged."
        ),
        "projection": "src.answer_projection.project_answer",
        "metrics": "src.metrics best_exact_match, best_token_f1, best_contains, abstention_correct",
        "datasets": {},
    }

    for name, paths in DATASETS.items():
        frame = pd.read_csv(paths["csv"])
        queries = {q.query_id: q for q in load_queries(paths["queries"])}
        missing = sorted({qid for qid in frame["query_id"] if qid not in queries})
        if missing:
            raise SystemExit(f"{name}: {len(missing)} query ids missing from {paths['queries']}")

        mismatches = []
        rows_out = []
        for record in frame.to_dict(orient="records"):
            query = queries[record["query_id"]]
            pred = "" if pd.isna(record["pred_answer"]) else str(record["pred_answer"])
            original = _score(pred, query)
            for key in ("em", "token_f1", "contains"):
                saved = float(record[key])
                if not _close(original[key], saved, tol=1e-6):
                    mismatches.append(
                        {
                            "system": record["system"],
                            "query_id": record["query_id"],
                            "metric": key,
                            "saved": saved,
                            "recomputed": original[key],
                        }
                    )
            system = record["system"]
            if system in PROJECT_SYSTEMS:
                projected = project_answer(query.query, pred)
            else:
                projected = pred
            rescored = _score(projected, query)
            rows_out.append(
                {
                    "system": system,
                    "query_id": record["query_id"],
                    "changed": int(projected != pred),
                    "em_before": original["em"],
                    "f1_before": original["token_f1"],
                    "contains_before": original["contains"],
                    "em_after": rescored["em"],
                    "f1_after": rescored["token_f1"],
                    "contains_after": rescored["contains"],
                    "latency_s": float(record["latency_s"]),
                    "did_retrieve": float(record["did_retrieve"]),
                }
            )

        if mismatches:
            sample = mismatches[:8]
            raise SystemExit(
                f"{name}: recomputed metrics disagree with the CSV on {len(mismatches)} cells. Sample: {sample}"
            )

        by_system: dict[str, dict] = {}
        grouped: dict[str, list[dict]] = {}
        for row in rows_out:
            grouped.setdefault(row["system"], []).append(row)
        for system, group in grouped.items():
            by_system[system] = {
                "n": len(group),
                "changed_predictions": int(sum(r["changed"] for r in group)),
                "em_before": _mean([r["em_before"] for r in group]),
                "token_f1_before": _mean([r["f1_before"] for r in group]),
                "contains_before": _mean([r["contains_before"] for r in group]),
                "em_after": _mean([r["em_after"] for r in group]),
                "token_f1_after": _mean([r["f1_after"] for r in group]),
                "contains_after": _mean([r["contains_after"] for r in group]),
                "em_count_before": int(sum(r["em_before"] for r in group)),
                "em_count_after": int(sum(r["em_after"] for r in group)),
                "contains_count_before": int(sum(r["contains_before"] for r in group)),
                "contains_count_after": int(sum(r["contains_after"] for r in group)),
                "latency_s_mean": _mean([r["latency_s"] for r in group]),
                "latency_s_median": _median([r["latency_s"] for r in group]),
                "retrieval_rate": _mean([r["did_retrieve"] for r in group]),
            }

        # Questions where Adaptive matches and B3 does not, before and after.
        wide: dict[str, dict[str, dict]] = {}
        for row in rows_out:
            wide.setdefault(row["query_id"], {})[row["system"]] = row
        explained = {"n_adaptive_only_em": 0, "still_only_after": 0, "b3_gains_em": 0}
        for qid, systems in wide.items():
            if "adaptive" not in systems or "B3_fixed_temporal" not in systems:
                continue
            ada = systems["adaptive"]
            b3 = systems["B3_fixed_temporal"]
            if ada["em_before"] == 1.0 and b3["em_before"] == 0.0:
                explained["n_adaptive_only_em"] += 1
                if b3["em_after"] == 0.0:
                    explained["still_only_after"] += 1
                if b3["em_after"] == 1.0:
                    explained["b3_gains_em"] += 1

        tails = {}
        for system, group in grouped.items():
            lats = [r["latency_s"] for r in group]
            tails[system] = {
                "max_latency_s": max(lats) if lats else None,
                "n_over_10s": int(sum(x > 10 for x in lats)),
                "n_over_30s": int(sum(x > 30 for x in lats)),
            }
        outlier = None
        if grouped.get("adaptive"):
            worst = max(grouped["adaptive"], key=lambda r: r["latency_s"])
            others = [r["latency_s"] for r in grouped["adaptive"] if r["query_id"] != worst["query_id"]]
            csv_row = frame[(frame["system"] == "adaptive") & (frame["query_id"] == worst["query_id"])].iloc[0]
            outlier = {
                "query_id": worst["query_id"],
                "query": "" if pd.isna(csv_row["query"]) else str(csv_row["query"]),
                "latency_s": worst["latency_s"],
                "em": worst["em_before"],
                "contains": worst["contains_before"],
                "mean_of_other_queries": _mean(others),
                "n_other": len(others),
            }

        report["datasets"][name] = {
            "csv": str(paths["csv"].relative_to(ROOT)).replace("\\", "/"),
            "queries": str(paths["queries"].relative_to(ROOT)).replace("\\", "/"),
            "n_rows": int(len(frame)),
            "systems": by_system,
            "adaptive_only_em_vs_b3": explained,
            "latency_tails": tails,
            "adaptive_latency_outlier": outlier,
        }
        print(name)
        for system, stats in by_system.items():
            print(
                f"  {system}: n={stats['n']} changed={stats['changed_predictions']} "
                f"EM {stats['em_before']:.4f}->{stats['em_after']:.4f} "
                f"({stats['em_count_before']}->{stats['em_count_after']}) "
                f"F1 {stats['token_f1_before']:.4f}->{stats['token_f1_after']:.4f} "
                f"Cont {stats['contains_before']:.4f}->{stats['contains_after']:.4f} "
                f"({stats['contains_count_before']}->{stats['contains_count_after']}) "
                f"med {stats['latency_s_median']:.4f}"
            )
        print(f"  adaptive-only EM vs B3: {explained}")
        if outlier:
            print(
                f"  outlier {outlier['query_id']} {outlier['latency_s']:.3f}s "
                f"others_mean={outlier['mean_of_other_queries']:.4f}"
            )

    dest = out_dir / "projection_fair_comparison.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
