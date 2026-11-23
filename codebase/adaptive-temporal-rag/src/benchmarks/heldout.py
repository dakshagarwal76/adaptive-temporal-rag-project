"""Synthetic split where no entity in the test questions appears in train or val."""

from __future__ import annotations

import random
import sys
from pathlib import Path

from ..data_schema import CorpusDoc, QueryItem

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def entity_names() -> list[str]:
    import build_dataset as dataset

    names: list[str] = []
    for block in dataset.OFFICE_HOLDERS + dataset.CEOS + dataset.SPORTS + dataset.AWARDS + dataset.LAWS_POLICIES:
        names.append(block["entity"])
    return names


def heldout_entities(names: list[str], every: int = 5) -> set[str]:
    ordered = sorted(names)
    return {name for index, name in enumerate(ordered) if index % every == 0}


def entity_in_query(query: str, names: list[str]) -> str | None:
    lowered = query.lower()
    matches = [name for name in names if name.lower() in lowered]
    if not matches:
        return None
    return max(matches, key=len)


def build_heldout(seed: int = 42) -> tuple[list[CorpusDoc], list[QueryItem], set[str]]:
    import build_dataset as dataset

    rng = random.Random(seed)
    docs = dataset.build_corpus(rng)
    queries = dataset.build_queries(docs, random.Random(seed), target=1200)
    names = entity_names()
    held = heldout_entities(names)
    train_val = [query for query in queries if query.split in ("train", "val")]
    reassigned: list[QueryItem] = []
    for query in queries:
        found = entity_in_query(query.query, names)
        if found in held:
            query.split = "test"
            reassigned.append(query)
            continue
        if query.split == "test":
            query.split = "train"
        reassigned.append(query)
    # Keep a validation slice among the seen entities.
    seen = [query for query in reassigned if query.split == "train" and not query.is_paraphrase]
    rng.shuffle(seen)
    val_ids = {query.query_id for query in seen[: max(1, len(seen) // 8)]}
    for query in reassigned:
        if query.query_id in val_ids:
            query.split = "val"
    return docs, reassigned, held
