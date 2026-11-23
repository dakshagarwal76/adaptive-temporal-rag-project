"""Train the calibrated DistilBERT router on synthetic questions plus SituatedQA.

Saves artifacts/analyzer_v2.pt and does not replace artifacts/analyzer.pt.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_schema import load_queries
from src.model_config import DEFAULT_ANALYZER_MODEL
from src.query_analyzer import LABEL2ID, train_analyzer


def _rows(path: Path, split: str | None = None):
    queries = load_queries(path)
    if split:
        queries = [query for query in queries if query.split == split]
    return queries


def _fit_temperature(model, tokenizer, texts, temporal, device: str, max_length: int) -> float:
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for text, label in zip(texts, temporal):
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            logits.append(out["temporal_logits"][0].detach().cpu())
            labels.append(LABEL2ID[label])
    stacked = torch.stack(logits)
    target = torch.tensor(labels)
    best_t, best_nll = 1.0, float("inf")
    for step in range(5, 31):
        temperature = step / 10
        log_probs = torch.log_softmax(stacked / temperature, dim=-1)
        nll = float(-log_probs[torch.arange(len(target)), target].mean())
        if nll < best_nll:
            best_t, best_nll = temperature, nll
    print(f"Temperature {best_t:.1f} val temporal NLL {best_nll:.4f}")
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=0, help="Cap training questions. 0 keeps every row.")
    parser.add_argument("--max-val", type=int, default=0, help="Cap validation questions. 0 keeps every row.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freeze-layers", type=int, default=4)
    args = parser.parse_args()

    synthetic = _rows(ROOT / "data" / "queries.jsonl")
    situated_path = ROOT / "data" / "general" / "situatedqa" / "queries.jsonl"
    situated = _rows(situated_path) if situated_path.exists() else []
    train = [q for q in synthetic if q.split == "train"] + [q for q in situated if q.split == "train"]
    val = [q for q in synthetic if q.split == "val"] + [q for q in situated if q.split == "val"]
    if len(val) < 20:
        val = train[:200]
        train = train[200:]
    rng = random.Random(42)
    if args.max_train > 0 and len(train) > args.max_train:
        rng.shuffle(train)
        train = train[: args.max_train]
    if args.max_val > 0 and len(val) > args.max_val:
        rng.shuffle(val)
        val = val[: args.max_val]
    work = ROOT / "artifacts" / "router_v2"
    base_weights = ROOT / "artifacts" / "analyzer.pt"
    print(f"Router train={len(train)} val={len(val)}")
    train_analyzer(
        [q.query for q in train],
        [q.needs_retrieval for q in train],
        [q.temporal_class for q in train],
        [q.query for q in val],
        [q.needs_retrieval for q in val],
        [q.temporal_class for q in val],
        artifacts_dir=work,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=64,
        model_name=DEFAULT_ANALYZER_MODEL,
        freeze_layers=args.freeze_layers,
        augment_factor=1,
        patience=1,
        init_state=base_weights if base_weights.exists() else None,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer

    from src.hf_utils import from_pretrained_local_first
    from src.query_analyzer import MultitaskQueryAnalyzer

    meta = json.loads((work / "analyzer_meta.json").read_text(encoding="utf-8"))
    tokenizer = from_pretrained_local_first(AutoTokenizer.from_pretrained, meta.get("tokenizer_name", DEFAULT_ANALYZER_MODEL))
    model = MultitaskQueryAnalyzer(model_name=meta.get("model_name", DEFAULT_ANALYZER_MODEL))
    state = torch.load(work / "analyzer.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    temperature = _fit_temperature(
        model, tokenizer, [q.query for q in val], [q.temporal_class for q in val], device, 64
    )
    meta["temperature"] = temperature
    meta["blend_confidence"] = False
    artifacts = ROOT / "artifacts"
    shutil.copyfile(work / "analyzer.pt", artifacts / "analyzer_v2.pt")
    (artifacts / "analyzer_v2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (work / "analyzer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", artifacts / "analyzer_v2.pt")


if __name__ == "__main__":
    main()
