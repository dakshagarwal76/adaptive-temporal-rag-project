"""Paired McNemar tests and bootstrap CIs from saved per-query predictions.

Reads artifacts/general_eval/final. Does not re-run retrieval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "general_eval" / "final"
OUT_JSON = ROOT / "artifacts" / "significance.json"
OUT_MD = ROOT / "artifacts" / "final_results.md"
DATASETS = ("synthetic", "heldout", "timeqa")
METRICS = ("em", "contains")
N_BOOT = 10_000
SEED = 0


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test of p=0.5 (McNemar)."""
    if n <= 0:
        return 1.0
    def pmf(i: int) -> float:
        return math.comb(n, i) / (2**n)

    obs = pmf(k)
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-18)
    return float(min(1.0, total))


def _paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    a = a.astype(float)
    b = b.astype(float)
    n = int(len(a))
    diff = a - b
    point = float(diff.mean()) if n else float("nan")
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    p = _binom_two_sided(n10, n10 + n01)
    if n == 0:
        lo = hi = float("nan")
    else:
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boots = diff[idx].mean(axis=1)
        lo, hi = (float(x) for x in np.quantile(boots, [0.025, 0.975]))
    return {
        "n": n,
        "mean_a": float(a.mean()) if n else float("nan"),
        "mean_b": float(b.mean()) if n else float("nan"),
        "diff_a_minus_b": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_a_only": n10,
        "n_b_only": n01,
        "mcnemar_p": p,
        "bootstrap_resamples": N_BOOT,
        "seed": SEED,
    }


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["query_id"] = df["query_id"].astype(str)
    return df


def _binary(df: pd.DataFrame, metric: str) -> pd.Series:
    return (df.set_index("query_id")[metric].astype(float) >= 0.5).astype(int)


def _compare(name_a: str, frame_a: pd.DataFrame, name_b: str, frame_b: pd.DataFrame, rng) -> dict:
    out = {}
    for metric in METRICS:
        left = _binary(frame_a, metric)
        right = _binary(frame_b, metric)
        joined = pd.concat([left.rename("a"), right.rename("b")], axis=1, join="inner").dropna()
        stats = _paired(joined["a"].to_numpy(), joined["b"].to_numpy(), rng)
        stats["metric"] = metric
        stats["a"] = name_a
        stats["b"] = name_b
        out[metric] = stats
    return out


def _main_frames(dataset: str) -> dict[str, pd.DataFrame]:
    path = FINAL / dataset / "results.csv"
    df = _load_csv(path)
    return {system: group.copy() for system, group in df.groupby("system")}


def _ablation_frame(dataset: str, ablation: str) -> pd.DataFrame | None:
    path = FINAL / "ablations" / dataset / f"results_{ablation}.csv"
    if not path.exists():
        return None
    return _load_csv(path)


def _b4_frame(dataset: str) -> pd.DataFrame | None:
    path = FINAL / dataset / "b4_tempralm_decay" / "results.csv"
    if not path.exists():
        return None
    return _load_csv(path)


def _fmt_p(p: float) -> str:
    if p != p:
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def _fmt_ci(row: dict) -> str:
    return (
        f"{row['diff_a_minus_b']:+.3f} "
        f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] "
        f"p={_fmt_p(row['mcnemar_p'])} "
        f"(only A {row['n_a_only']}, only B {row['n_b_only']})"
    )


def _markdown(payload: dict) -> str:
    lines = [
        "## Significance",
        "",
        "Paired tests on the saved per-query rows. No retrieval was re-run.",
        "McNemar p-values are exact two-sided binomial tests on discordant pairs.",
        f"95% intervals are paired bootstrap percentile intervals of (A − B) with {N_BOOT} resamples, seed {SEED}.",
        "A is the first system named in each row.",
        "",
        "| Dataset | Comparison | Metric | A−B | 95% CI | McNemar p | only A | only B |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for dataset, block in payload["datasets"].items():
        for comparison, metrics in block.items():
            for metric, row in metrics.items():
                ci = f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            dataset,
                            comparison,
                            metric,
                            f"{row['diff_a_minus_b']:+.3f}",
                            ci,
                            _fmt_p(row["mcnemar_p"]),
                            str(row["n_a_only"]),
                            str(row["n_b_only"]),
                        ]
                    )
                    + " |"
                )
    lines.append("")
    lines.append(f"File: `artifacts/significance.json`.")
    lines.append("")
    return "\n".join(lines)


def _append_md(section: str) -> None:
    if not OUT_MD.exists():
        OUT_MD.write_text(section, encoding="utf-8")
        return
    text = OUT_MD.read_text(encoding="utf-8")
    marker = "\n## Significance\n"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    OUT_MD.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)
    payload: dict = {
        "method": "paired McNemar exact + paired bootstrap percentile CI",
        "bootstrap_resamples": N_BOOT,
        "seed": SEED,
        "datasets": {},
    }
    for dataset in DATASETS:
        frames = _main_frames(dataset)
        block: dict = {}
        adaptive = frames["adaptive"]
        for other in ("B2_standard_rag", "B3_fixed_temporal"):
            block[f"adaptive vs {other}"] = _compare("adaptive", adaptive, other, frames[other], rng)
        b4 = _b4_frame(dataset)
        if b4 is not None:
            block["adaptive vs tempralm_decay"] = _compare("adaptive", adaptive, "tempralm_decay", b4, rng)
        if dataset == "heldout":
            cut = _ablation_frame(dataset, "entity_cut_off")
            if cut is not None:
                block["adaptive vs entity_cut_off"] = _compare(
                    "adaptive", adaptive, "entity_cut_off", cut, rng
                )
        if dataset == "timeqa":
            for ablation in ("sentence_selector_off", "entity_cut_off", "window_off", "sufficiency_off"):
                frame = _ablation_frame(dataset, ablation)
                if frame is None:
                    continue
                block[f"adaptive vs {ablation}"] = _compare("adaptive", adaptive, ablation, frame, rng)
        payload["datasets"][dataset] = block
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    section = _markdown(payload)
    _append_md(section)
    print("Wrote", OUT_JSON)
    print("Updated", OUT_MD)
    for dataset, block in payload["datasets"].items():
        for comparison, metrics in block.items():
            for metric, row in metrics.items():
                print(f"{dataset} {comparison} {metric}: {_fmt_ci(row)}")


if __name__ == "__main__":
    main()
