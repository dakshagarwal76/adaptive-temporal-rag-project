"""Sentence-Transformers + FAISS index helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

from .data_schema import CorpusDoc, load_corpus


from .model_config import DEFAULT_EMBED_MODEL, LEGACY_EMBED_MODEL

# Re-export for callers that imported DEFAULT_EMBED_MODEL from here.
__all__ = ["DEFAULT_EMBED_MODEL", "LEGACY_EMBED_MODEL", "EmbeddingIndex", "build_index_from_corpus"]


class EmbeddingIndex:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        from .hf_utils import from_pretrained_local_first, prefer_local_hf_cache

        prefer_local_hf_cache()
        self._device = device
        self.model_name = model_name
        self._load_sentence_model(model_name, device)
        self.docs: list[CorpusDoc] = []
        self.embeddings: np.ndarray | None = None
        self.index = None

    def _load_sentence_model(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        from .hf_utils import from_pretrained_local_first

        try:
            self.model = from_pretrained_local_first(
                lambda n, **kw: SentenceTransformer(n, device=device or self._device, **kw),
                model_name,
            )
        except TypeError:
            self.model = SentenceTransformer(model_name, device=device or self._device)

    def build(self, docs: list[CorpusDoc], batch_size: int = 64) -> None:
        if faiss is None:
            raise ImportError("faiss is required. Install faiss-cpu.")
        self.docs = list(docs)
        texts = [d.text for d in self.docs]
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.embeddings = emb
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self.index = index

    def _format_query(self, query: str) -> str:
        name = (self.model_name or "").lower()
        if "bge" in name:
            return f"Represent this sentence for searching relevant passages: {query}"
        return query

    def encode_query(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [self._format_query(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return q

    def search(self, query: str, top_k: int = 5) -> list[tuple[CorpusDoc, float]]:
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Index not built. Call build() or load().")
        q = self.encode_query(query)
        scores, idxs = self.index.search(q, top_k)
        results: list[tuple[CorpusDoc, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append((self.docs[int(idx)], float(score)))
        return results

    def search_candidates(
        self, query: str, top_k: int = 50
    ) -> list[tuple[CorpusDoc, float]]:
        return self.search(query, top_k=top_k)

    def save(self, artifacts_dir: str | Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.embeddings is None or self.index is None:
            raise RuntimeError("Nothing to save.")
        np.save(artifacts_dir / "embeddings.npy", self.embeddings)
        faiss.write_index(self.index, str(artifacts_dir / "faiss.index"))
        meta = {
            "model_name": self.model_name,
            "doc_ids": [d.doc_id for d in self.docs],
            "n_docs": len(self.docs),
        }
        (artifacts_dir / "index_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        # Persist docs snapshot for reload without re-reading corpus path mismatches
        with (artifacts_dir / "corpus_snapshot.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    def load(self, artifacts_dir: str | Path, corpus_path: str | Path | None = None) -> None:
        if faiss is None:
            raise ImportError("faiss is required. Install faiss-cpu.")
        artifacts_dir = Path(artifacts_dir)
        self.embeddings = np.load(artifacts_dir / "embeddings.npy").astype("float32")
        self.index = faiss.read_index(str(artifacts_dir / "faiss.index"))
        meta_path = artifacts_dir / "index_meta.json"
        if meta_path.exists():
            disk_name = json.loads(meta_path.read_text(encoding="utf-8")).get("model_name")
            if disk_name and disk_name != self.model_name:
                self.model_name = disk_name
                self._load_sentence_model(disk_name, self._device)
        snap = artifacts_dir / "corpus_snapshot.jsonl"
        if snap.exists():
            from .data_schema import read_jsonl

            self.docs = [CorpusDoc(**row) for row in read_jsonl(snap)]
        elif corpus_path is not None:
            self.docs = load_corpus(corpus_path)
        else:
            raise FileNotFoundError("Need corpus_snapshot.jsonl or corpus_path")


def build_index_from_corpus(
    corpus_path: str | Path,
    artifacts_dir: str | Path,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
) -> EmbeddingIndex:
    docs = load_corpus(corpus_path)
    idx = EmbeddingIndex(model_name=model_name, device=device)
    idx.build(docs)
    idx.save(artifacts_dir)
    return idx
