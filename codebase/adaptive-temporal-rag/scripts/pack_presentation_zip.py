"""Build lean atr_presentation.zip for Colab demo (artifacts + code needed for inference)."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "atr_presentation.zip"

# Paths relative to project root
INCLUDE_PREFIXES = [
    "src/",
    "notebooks/01_colab_presentation.ipynb",
    "requirements.txt",
    "README_PRESENTATION.md",
    "docs/COLAB_E2E_FLOW.md",
    "docs/diagrams/01_adaptive_inference_full.png",
    "docs/diagrams/02_policy_router.png",
    "docs/diagrams/03_colab_four_systems_eval.png",
    "docs/diagrams/INFERENCE_NODES.md",
    "ARCHITECTURE.md",
    # Needed when RUN_TRAINING=True in the presentation notebook
    "scripts/build_dataset.py",
    "scripts/evaluate.py",
    "scripts/plot_results.py",
]

ARTIFACT_FILES = [
    "artifacts/analyzer.pt",
    "artifacts/analyzer_meta.json",
    "artifacts/faiss.index",
    "artifacts/embeddings.npy",
    "artifacts/index_meta.json",
    "artifacts/corpus_snapshot.jsonl",
    "artifacts/results_summary.json",
    "artifacts/results.csv",
    "artifacts/results_charts.png",
    "artifacts/comparative_analysis.md",
    "artifacts/comparative_analysis.json",
]

DATA_FILES = [
    "data/corpus.jsonl",
    # queries.jsonl not required for presentation demo (keeps zip lean)
]


def should_include_src(rel: str) -> bool:
    if not rel.startswith("src/"):
        return False
    if "__pycache__" in rel or rel.endswith(".pyc"):
        return False
    return rel.endswith(".py")


def main() -> None:
    must = ARTIFACT_FILES + DATA_FILES + [
        "notebooks/01_colab_presentation.ipynb",
        "src/pipelines.py",
        "src/user_corpus.py",
        "README_PRESENTATION.md",
    ]
    missing = [p for p in must if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"Missing required files:\n" + "\n".join(missing))

    if OUT.exists():
        OUT.unlink()

    count = 0
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # src/*.py
        for path in (ROOT / "src").rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if should_include_src(rel):
                zf.write(path, f"adaptive-temporal-rag/{rel}")
                count += 1

        # fixed files
        for rel in INCLUDE_PREFIXES:
            if rel.endswith("/"):
                continue
            path = ROOT / rel
            if path.exists() and path.is_file():
                zf.write(path, f"adaptive-temporal-rag/{rel}")
                count += 1

        for rel in ARTIFACT_FILES + DATA_FILES:
            path = ROOT / rel
            zf.write(path, f"adaptive-temporal-rag/{rel}")
            count += 1

        # empty markers so folders open cleanly
        zf.writestr("adaptive-temporal-rag/artifacts/.gitkeep", "")
        zf.writestr("adaptive-temporal-rag/data/.gitkeep", "")

    mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT} ({mb:.1f} MB, {count} files)")
    print("Open notebooks/01_colab_presentation.ipynb on Colab after unzip.")


if __name__ == "__main__":
    main()
