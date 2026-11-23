"""Run frozen Adaptive ablations on synthetic, held-out, and TimeQA full tests."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DATASETS = ("synthetic", "heldout", "timeqa")
ABLATIONS = (
    "router_off",
    "window_off",
    "sufficiency_off",
    "entity_cut_off",
    "sentence_selector_off",
)


def already_done(dataset: str, ablation: str) -> Path | None:
    path = (
        ROOT
        / "artifacts"
        / "general_eval"
        / "final"
        / "ablations"
        / dataset
        / f"results_{ablation}_summary.json"
    )
    return path if path.exists() else None


def main() -> None:
    timings: dict[str, float] = {}
    for dataset in DATASETS:
        for ablation in ABLATIONS:
            key = f"{dataset}/{ablation}"
            existing = already_done(dataset, ablation)
            if existing:
                print(f"SKIP {key} ({existing})")
                continue
            cmd = [
                PY,
                str(ROOT / "scripts" / "evaluate.py"),
                "--dataset",
                dataset,
                "--split",
                "test",
                "--config",
                str(ROOT / "configs" / "final.yaml"),
                "--final",
                "--ablation",
                ablation,
            ]
            print(f"RUN {key}", flush=True)
            t0 = time.perf_counter()
            subprocess.check_call(cmd, cwd=str(ROOT))
            elapsed = time.perf_counter() - t0
            timings[key] = elapsed
            print(f"DONE {key} seconds={elapsed:.1f}", flush=True)
    out = ROOT / "artifacts" / "ablation_timings.json"
    import json

    out.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
