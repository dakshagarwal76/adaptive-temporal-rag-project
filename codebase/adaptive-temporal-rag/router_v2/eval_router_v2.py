"""Evaluate the 100k DeBERTa router inside the frozen Checkpoint B Adaptive pipeline.

Pure learned routing: retrieval is skipped when the router predicts needs_retrieval == 0.
The trivia whitelist in src.adaptive_policy is not used. The swap is process-local
(src.pipelines.resolve_adaptive_routing); no src file is edited.

Full runs write per-query tables under artifacts/general_eval/learned_router_100k/
and statistics at artifacts/learned_router_100k_significance.json and
artifacts/learned_router_100k_eval.json. --limit N writes the tables under
out-dir/_smoke/ and the same two statistics filenames under that smoke directory,
so a sample does not replace a full-run statistics file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Frozen embedder, reranker, and the router all stay on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ATRAG_REPO", HERE.parent))
RESULTS_DIR = Path(os.environ.get("ROUTER_V2_RESULTS", HERE / "results"))
TAG = os.environ.get("ROUTER_TAG", "router_v2")
HYBRID = os.environ.get("ROUTER_HYBRID", "0") == "1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("synthetic", "heldout", "timeqa")
SYSTEMS = ("adaptive_learned100k", "adaptive_fast_learned100k")
ROUTING_BASELINE = {
    "adaptive_learned100k": "adaptive",
    "adaptive_fast_learned100k": "adaptive_sufficiency_off",
}
SIGNIFICANCE_BASELINES = (
    "adaptive",
    "adaptive_sufficiency_off",
    "B2_standard_rag",
    "B3_fixed_temporal",
    "tempralm_decay",
)
METRICS = ("em", "contains")
FINAL = ROOT / "artifacts" / "general_eval" / "final"
CONFIG_PATH = ROOT / "configs" / "final.yaml"
ARTIFACTS = ROOT / "artifacts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_module("evaluate", ROOT / "scripts" / "evaluate.py")
significance = _load_module("compute_significance", ROOT / "scripts" / "compute_significance.py")

from src.adaptive_policy import AdaptiveRoutingDecision  # noqa: E402
import src.pipelines  # noqa: E402
from src.config import EvalConfig, load_eval_config  # noqa: E402
from src.data_schema import load_queries  # noqa: E402
from src.generate import Generator  # noqa: E402
from src.metrics import aggregate_rows  # noqa: E402
from src.pipelines import build_all_pipelines  # noqa: E402
from src.router_large import load_large_router  # noqa: E402


def resolve_pure_learned_routing(analysis, query_text, conf_threshold):
    """Copy of resolve_adaptive_routing. Skip condition does not call is_trivial_query."""
    if analysis.needs_retrieval == 0:
        return AdaptiveRoutingDecision(True, "no_retrieval", "")

    mode = analysis.temporal_class
    soft = analysis.needs_retrieval == 0
    policy = "soft_retrieve_timeless" if soft else mode
    if soft or analysis.confidence < conf_threshold or mode == "timeless":
        if not soft and analysis.confidence < conf_threshold:
            policy = "low_conf_semantic"
        elif mode == "timeless" and not soft:
            policy = "timeless_semantic"
        retrieve_mode = "semantic"
    elif mode == "current":
        retrieve_mode = "current"
    elif mode == "historical":
        retrieve_mode = "historical"
        policy = "historical_adaptive_window"
    else:
        retrieve_mode = "semantic"
    return AdaptiveRoutingDecision(False, policy, retrieve_mode)


def install_pure_learned_routing() -> None:
    # pipelines.py bound the name into its own module global. Run looks that name up.
    src.pipelines.resolve_adaptive_routing = resolve_pure_learned_routing


def set_seeds() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    return target


def queries_path(dataset: str) -> Path:
    if dataset == "synthetic":
        return ROOT / "data" / "queries.jsonl"
    if dataset == "heldout":
        return ROOT / "data" / "heldout" / "queries.jsonl"
    if dataset == "timeqa":
        return ROOT / "data" / "general" / "timeqa" / "queries.jsonl"
    raise SystemExit(f"Unknown dataset {dataset}")


def load_test_queries(dataset: str, limit: int):
    queries = [query for query in load_queries(queries_path(dataset)) if query.split == "test"]
    if limit > 0:
        queries = queries[:limit]
    return queries


def config_for_system(base: EvalConfig, system_name: str) -> EvalConfig:
    if system_name == "adaptive_learned100k":
        return base
    if system_name == "adaptive_fast_learned100k":
        return base.apply_ablation("sufficiency_off")
    raise SystemExit(f"Unknown system {system_name}")


def results_csv(run_dir: Path, dataset: str, system_name: str) -> Path:
    return run_dir / dataset / f"results_{system_name}.csv"


def summary_json(run_dir: Path, dataset: str, system_name: str) -> Path:
    return run_dir / dataset / f"results_{system_name}_summary.json"


def results_complete(path: Path, expected: int) -> bool:
    if expected <= 0 or not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["query_id"])
    except Exception:
        return False
    return len(frame) == expected


def _norm_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _norm_flag(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def frozen_results_path(dataset: str, baseline: str) -> Path | None:
    if baseline == "adaptive_sufficiency_off":
        return FINAL / "ablations" / dataset / "results_sufficiency_off.csv"
    if baseline == "tempralm_decay":
        return FINAL / dataset / "b4_tempralm_decay" / "results.csv"
    if baseline in {"adaptive", "B2_standard_rag", "B3_fixed_temporal"}:
        return FINAL / dataset / "results.csv"
    return None


def load_frozen_frame(dataset: str, baseline: str) -> pd.DataFrame | None:
    path = frozen_results_path(dataset, baseline)
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["query_id"] = frame["query_id"].astype(str)
    if "system" in frame.columns:
        frame = frame[frame["system"].astype(str) == baseline]
    return frame


def attach_frozen(rows: list[dict], frozen: pd.DataFrame | None) -> None:
    indexed = None
    if frozen is not None and not frozen.empty:
        indexed = frozen.drop_duplicates("query_id", keep="first").set_index("query_id")
    for row in rows:
        qid = str(row["query_id"])
        if indexed is None or qid not in indexed.index:
            row["frozen_policy"] = ""
            row["frozen_did_retrieve"] = float("nan")
            row["frozen_pred_needs"] = float("nan")
            row["frozen_pred_temporal"] = ""
            row["routing_differs"] = 1
            continue
        source = indexed.loc[qid]
        frozen_policy = _norm_text(source.get("policy"))
        frozen_retrieve = _norm_flag(source.get("did_retrieve"))
        frozen_needs = _norm_flag(source.get("pred_needs"))
        frozen_temporal = _norm_text(source.get("pred_temporal"))
        row["frozen_policy"] = frozen_policy
        row["frozen_did_retrieve"] = float("nan") if frozen_retrieve is None else float(frozen_retrieve)
        row["frozen_pred_needs"] = float("nan") if frozen_needs is None else float(frozen_needs)
        row["frozen_pred_temporal"] = frozen_temporal
        differs = (
            _norm_text(row.get("policy")) != frozen_policy
            or _norm_flag(row.get("did_retrieve")) != frozen_retrieve
            or _norm_text(row.get("pred_temporal")) != frozen_temporal
        )
        row["routing_differs"] = int(differs)


def _fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.3f}"


def update_timings(path: Path, dataset: str, system_name: str, seconds: float) -> None:
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    block = payload.get(dataset)
    if not isinstance(block, dict):
        block = {}
    block[system_name] = seconds
    payload[dataset] = block
    write_json(path, payload)


def run_system(pipe, queries, dataset: str, system_name: str, cfg: EvalConfig, frozen: pd.DataFrame | None) -> tuple[list[dict], float]:
    rows: list[dict] = []
    total = len(queries)
    started = time.perf_counter()
    for index, query in enumerate(queries, start=1):
        if index % 25 == 0 or index == total:
            print(f"  {dataset} {system_name} query {index}/{total}", flush=True)
        try:
            result = pipe.run(query)
        except Exception:
            logging.exception(
                "pipe.run failed dataset=%s system=%s query_id=%s",
                dataset,
                system_name,
                getattr(query, "query_id", ""),
            )
            raise
        row = evaluate._score_query(result, query, dataset, cfg)
        row["router_skip"] = int(_norm_text(row.get("policy")) == "no_retrieval")
        if result.analyzer is not None:
            row["necessity_prob"] = float(result.analyzer.necessity_prob)
            row["temporal_probs_json"] = json.dumps(result.analyzer.temporal_probs)
        else:
            row["necessity_prob"] = float("nan")
            row["temporal_probs_json"] = "{}"
        rows.append(row)
    elapsed = time.perf_counter() - started
    attach_frozen(rows, frozen)
    return rows, elapsed


def build_adaptive(index, analyzer, cfg: EvalConfig, system_name: str):
    generator = Generator(backend=cfg.generator, use_sentence_selector=cfg.enable_sentence_selector)
    pipes = build_all_pipelines(index=index, generator=generator, analyzer=analyzer, config=cfg)
    pipe = pipes["adaptive"]
    pipe.name = system_name
    return pipe


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.map(_norm_text).value_counts()
    return {str(key): int(value) for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def _column_mean(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def routing_agreement(frame: pd.DataFrame) -> dict:
    learned_policy = frame["policy"].map(_norm_text) if "policy" in frame.columns else pd.Series([""] * len(frame))
    frozen_policy = frame["frozen_policy"].map(_norm_text) if "frozen_policy" in frame.columns else pd.Series([""] * len(frame))
    learned_retrieve = frame["did_retrieve"].map(_norm_flag) if "did_retrieve" in frame.columns else pd.Series([None] * len(frame))
    frozen_retrieve = frame["frozen_did_retrieve"].map(_norm_flag) if "frozen_did_retrieve" in frame.columns else pd.Series([None] * len(frame))
    learned_temporal = frame["pred_temporal"].map(_norm_text) if "pred_temporal" in frame.columns else pd.Series([""] * len(frame))
    frozen_temporal = frame["frozen_pred_temporal"].map(_norm_text) if "frozen_pred_temporal" in frame.columns else pd.Series([""] * len(frame))
    if "routing_differs" in frame.columns:
        n_routing = int(pd.to_numeric(frame["routing_differs"], errors="coerce").fillna(0).sum())
    else:
        n_routing = 0
    return {
        "n_routing_differs": n_routing,
        "n_did_retrieve_differs": int(sum(left != right for left, right in zip(learned_retrieve, frozen_retrieve))),
        "n_pred_temporal_differs": int(sum(left != right for left, right in zip(learned_temporal, frozen_temporal))),
        "n_skip_learned": int((learned_policy == "no_retrieval").sum()),
        "n_skip_frozen": int((frozen_policy == "no_retrieval").sum()),
        "learned_policy_counts": _value_counts(learned_policy),
        "frozen_policy_counts": _value_counts(frozen_policy),
    }


def load_timings(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def metadata(router_dir: Path, limit: int) -> dict:
    meta = json.loads((router_dir / "analyzer_meta.json").read_text(encoding="utf-8"))
    import transformers

    return {
        "router_bundle": str(router_dir.resolve()),
        "analyzer_pt_sha256": sha256_file(router_dir / "analyzer.pt"),
        "model_name": meta.get("model_name"),
        "temperature": meta.get("temperature"),
        "config_path": str(CONFIG_PATH.resolve()),
        "device": "cpu",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": int(limit),
    }


def _binary(frame: pd.DataFrame, metric: str) -> pd.Series:
    copied = frame.copy()
    copied["query_id"] = copied["query_id"].astype(str)
    return (copied.set_index("query_id")[metric].astype(float) >= 0.5).astype(int)


def compare_frames(frame_a: pd.DataFrame, frame_b: pd.DataFrame, rng: np.random.Generator) -> dict:
    compared = {}
    for metric in METRICS:
        left = _binary(frame_a, metric)
        right = _binary(frame_b, metric)
        joined = pd.concat([left.rename("a"), right.rename("b")], axis=1, join="inner").dropna()
        compared[metric] = significance._paired(joined["a"].to_numpy(), joined["b"].to_numpy(), rng)
    return compared


def statistics_paths(run_dir: Path, limit: int) -> tuple[Path, Path]:
    if limit > 0:
        return (
            run_dir / f"{TAG}_significance.json",
            run_dir / f"{TAG}_eval.json",
        )
    return (
        RESULTS_DIR / f"{TAG}_significance.json",
        RESULTS_DIR / f"{TAG}_eval.json",
    )


def run_statistics(datasets: list[str], systems: list[str], run_dir: Path, router_dir: Path, limit: int) -> list[Path]:
    timings = load_timings(run_dir / "timings.json")
    significance_payload: dict = {}
    eval_payload: dict = {"metadata": metadata(router_dir, limit), "datasets": {}}
    for dataset in datasets:
        rng = np.random.default_rng(0)
        learned: dict[str, pd.DataFrame] = {}
        dataset_block: dict = {}
        for system_name in systems:
            path = results_csv(run_dir, dataset, system_name)
            if not path.exists():
                print(f"Missing learned results {path}", flush=True)
                continue
            frame = pd.read_csv(path)
            frame["query_id"] = frame["query_id"].astype(str)
            learned[system_name] = frame
            record = {
                "n": int(len(frame)),
                "summary": aggregate_rows(frame.to_dict(orient="records")),
                "wall_clock_s": (timings.get(dataset) or {}).get(system_name),
                "routing_agreement": routing_agreement(frame),
                "router_temporal_acc": _column_mean(frame, "router_temporal_acc"),
                "router_needs_acc": _column_mean(frame, "router_needs_acc"),
            }
            timed_out = _column_mean(frame, "timed_out")
            capped = _column_mean(frame, "capped")
            if timed_out is not None:
                record["timed_out"] = timed_out
            if capped is not None:
                record["capped"] = capped
            dataset_block[system_name] = record
        eval_payload["datasets"][dataset] = dataset_block

        frozen_cache: dict[str, pd.DataFrame | None] = {}
        block: dict = {}
        for system_name in systems:
            if system_name not in learned:
                continue
            for baseline in SIGNIFICANCE_BASELINES:
                if baseline not in frozen_cache:
                    frozen_cache[baseline] = load_frozen_frame(dataset, baseline)
                frozen = frozen_cache[baseline]
                if frozen is None or frozen.empty:
                    print(f"Missing frozen baseline {dataset} {baseline}", flush=True)
                    continue
                block[f"{system_name} vs {baseline}"] = compare_frames(learned[system_name], frozen, rng)
        if "adaptive_learned100k" in learned and "adaptive_fast_learned100k" in learned:
            block["adaptive_learned100k vs adaptive_fast_learned100k"] = compare_frames(
                learned["adaptive_learned100k"],
                learned["adaptive_fast_learned100k"],
                rng,
            )
        significance_payload[dataset] = block

    sig_path, eval_path = statistics_paths(run_dir, limit)
    write_json(sig_path, significance_payload)
    write_json(eval_path, eval_payload)
    print("Wrote", sig_path, flush=True)
    print("Wrote", eval_path, flush=True)
    return [sig_path, eval_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 100k learned router on the frozen Adaptive pipeline.")
    parser.add_argument("--datasets", default="synthetic,heldout,timeqa")
    parser.add_argument("--systems", default="adaptive_learned100k,adaptive_fast_learned100k")
    parser.add_argument("--limit", type=int, default=0, help="Cap queries per dataset. Writes under out-dir/_smoke.")
    parser.add_argument("--router-dir", default="artifacts/router_large_100k")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "general_eval" / TAG))
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute a dataset/system even if its results CSV is complete.")
    args = parser.parse_args()
    args.datasets = _split_csv(args.datasets)
    args.systems = _split_csv(args.systems)
    unknown_datasets = [name for name in args.datasets if name not in DATASETS]
    unknown_systems = [name for name in args.systems if name not in SYSTEMS]
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown_datasets)}")
    if unknown_systems:
        raise SystemExit(f"Unknown systems: {', '.join(unknown_systems)}")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    return args


def _shard(queries):
    spec = os.environ.get("ROUTER_SHARD", "")
    if not spec:
        return queries
    i, n = (int(x) for x in spec.split("/"))
    return queries[i::n]

def main() -> None:
    set_seeds()
    args = parse_args()
    if HYBRID:
        print("HYBRID: frozen resolve_adaptive_routing kept (skip needs learned needs==0 AND trivia whitelist)", flush=True)
    else:
        install_pure_learned_routing()
    if not HYBRID and src.pipelines.resolve_adaptive_routing is not resolve_pure_learned_routing:
        raise RuntimeError("pure learned routing was not installed on src.pipelines")

    router_dir = _resolve(args.router_dir)
    out_dir = _resolve(args.out_dir)
    run_dir = out_dir / "_smoke" if args.limit > 0 else out_dir
    written: list[Path] = []

    if not args.stats_only:
        base_cfg = load_eval_config(CONFIG_PATH)
        print(
            f"Config {base_cfg.name} sufficiency_loop={base_cfg.enable_sufficiency_loop} "
            f"max_sufficiency_rounds={base_cfg.max_sufficiency_rounds} timeout_s={base_cfg.query_timeout_s}",
            flush=True,
        )
        needs_model = False
        expected_by_dataset: dict[str, int] = {}
        for dataset in args.datasets:
            expected_by_dataset[dataset] = len(_shard(load_test_queries(dataset, args.limit)))
            for system_name in args.systems:
                path = results_csv(run_dir, dataset, system_name)
                if args.force or not results_complete(path, expected_by_dataset[dataset]):
                    needs_model = True
        analyzer = None
        if needs_model:
            set_seeds()
            print(f"Loading router {router_dir} on cpu", flush=True)
            analyzer = load_large_router(router_dir, device="cpu")
        timings_path = run_dir / "timings.json"
        for dataset in args.datasets:
            queries = _shard(load_test_queries(dataset, args.limit))
            expected = len(queries)
            pending = []
            for system_name in args.systems:
                path = results_csv(run_dir, dataset, system_name)
                if not args.force and results_complete(path, expected):
                    print(f"Skip {dataset} {system_name}: {path} has {expected} rows", flush=True)
                    continue
                pending.append(system_name)
            if not pending:
                continue
            print(f"Loading index dataset={dataset}", flush=True)
            index, _index_dir = evaluate._index_for(dataset, ARTIFACTS)
            for system_name in args.systems:
                if system_name not in pending:
                    continue
                cfg = config_for_system(base_cfg, system_name)
                print(
                    f"Evaluating {len(queries)} queries dataset={dataset} system={system_name} "
                    f"sufficiency_loop={cfg.enable_sufficiency_loop}",
                    flush=True,
                )
                pipe = build_adaptive(index, analyzer, cfg, system_name)
                if pipe.name != system_name:
                    raise RuntimeError(f"pipeline name is {pipe.name}, expected {system_name}")
                baseline = ROUTING_BASELINE[system_name]
                frozen = load_frozen_frame(dataset, baseline)
                if frozen is None:
                    print(f"WARNING no frozen rows for {dataset} {baseline}", flush=True)
                rows, elapsed = run_system(pipe, queries, dataset, system_name, cfg, frozen)
                frame = pd.DataFrame(rows)
                csv_path = results_csv(run_dir, dataset, system_name)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(csv_path, index=False, encoding="utf-8")
                summary = aggregate_rows(rows)
                summary_path = summary_json(run_dir, dataset, system_name)
                write_json(summary_path, summary)
                update_timings(timings_path, dataset, system_name, elapsed)
                written.extend([csv_path, summary_path, timings_path])
                n_diff = int(sum(int(row["routing_differs"]) for row in rows))
                print(
                    f"{dataset} {system_name}: contains={_fmt(summary['contains'])} "
                    f"em={_fmt(summary['em'])} token_f1={_fmt(summary['token_f1'])} "
                    f"p@5={_fmt(summary['precision_at_k'])} r@5={_fmt(summary['recall_at_k'])} "
                    f"retrieval_rate={_fmt(summary['retrieval_rate'])} "
                    f"mean_latency={_fmt(summary['latency_s'])} median_latency={_fmt(summary['latency_s_median'])} "
                    f"routing_differs={n_diff}",
                    flush=True,
                )
                print("Wrote", csv_path, flush=True)
                print("Wrote", summary_path, flush=True)

    written.extend(run_statistics(args.datasets, args.systems, run_dir, router_dir, args.limit))
    print("Created files:", flush=True)
    seen = set()
    for path in written:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        print(resolved, flush=True)


if __name__ == "__main__":
    main()
