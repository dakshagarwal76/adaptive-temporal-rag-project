"""Train DistilBERT multitask query analyzer (Colab/local)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_schema import load_queries
from src.model_config import DEFAULT_ANALYZER_MODEL
from src.query_analyzer import train_analyzer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=str, default=str(ROOT / "data" / "queries.jsonl"))
    p.add_argument("--artifacts", type=str, default=str(ROOT / "artifacts"))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--freeze-layers", type=int, default=0)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument(
        "--augment-factor",
        type=int,
        default=1,
        help="Multiply train rows with light paraphrases (T4 pilot ~8, A100 large ~12).",
    )
    p.add_argument(
        "--train-seeds-only",
        action="store_true",
        default=True,
        help="Exclude dataset paraphrases from train (recommended).",
    )
    p.add_argument("--no-train-seeds-only", action="store_false", dest="train_seeds_only")
    p.add_argument("--model-name", type=str, default=DEFAULT_ANALYZER_MODEL)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    args = p.parse_args()

    queries = load_queries(args.queries)
    train = [q for q in queries if q.split == "train"]
    if args.train_seeds_only:
        train = [q for q in train if not q.is_paraphrase]
        print(f"Train seeds only: {len(train)} queries")
    val = [q for q in queries if q.split == "val"]
    print(f"Train={len(train)} Val={len(val)} augment_factor={args.augment_factor}")

    result = train_analyzer(
        train_texts=[q.query for q in train],
        train_needs=[q.needs_retrieval for q in train],
        train_temporal=[q.temporal_class for q in train],
        val_texts=[q.query for q in val],
        val_needs=[q.needs_retrieval for q in val],
        val_temporal=[q.temporal_class for q in val],
        artifacts_dir=args.artifacts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_layers=args.freeze_layers,
        max_length=args.max_length,
        augment_factor=args.augment_factor,
        model_name=args.model_name,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )
    print("Best val score:", result["best_val_score"])
    print("Train examples used:", result.get("train_examples"))
    print("Saved analyzer to", args.artifacts)


if __name__ == "__main__":
    main()
