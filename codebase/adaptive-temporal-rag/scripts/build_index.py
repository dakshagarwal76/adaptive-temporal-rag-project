"""Build MiniLM + FAISS index from corpus.jsonl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embed_index import build_index_from_corpus
from src.model_config import DEFAULT_EMBED_MODEL


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default=str(ROOT / "data" / "corpus.jsonl"))
    p.add_argument("--artifacts", type=str, default=str(ROOT / "artifacts"))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    args = p.parse_args()
    idx = build_index_from_corpus(
        args.corpus, args.artifacts, model_name=args.embed_model, device=args.device
    )
    print(f"Indexed {len(idx.docs)} docs -> {args.artifacts}")


if __name__ == "__main__":
    main()
