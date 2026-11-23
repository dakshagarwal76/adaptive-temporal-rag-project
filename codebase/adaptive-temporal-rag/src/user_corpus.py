"""Loopable knowledge-base: add many user documents, rebuild/update FAISS, query."""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .data_schema import CorpusDoc, load_corpus, write_jsonl
from .embed_index import DEFAULT_EMBED_MODEL, EmbeddingIndex, build_index_from_corpus

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:48] or "doc"


def docs_from_text_blobs(
    blobs: list[tuple[str, str, str]],
    id_prefix: str = "user",
) -> list[CorpusDoc]:
    """Each blob: (title_or_filename, body_text, timestamp YYYY-MM-DD)."""
    out: list[CorpusDoc] = []
    stamp = uuid.uuid4().hex[:8]
    for i, (title, text, ts) in enumerate(blobs, start=1):
        body = (text or "").strip()
        if not body:
            continue
        doc_id = f"{id_prefix}_{stamp}_{_slug(title)}_{i}"
        out.append(
            CorpusDoc(
                doc_id=doc_id,
                text=body,
                timestamp=ts or date.today().isoformat(),
                title=title or doc_id,
                source="user_upload",
                entity="",
                topic="user",
            )
        )
    return out


def _is_pdf(filename: str, raw: bytes) -> bool:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return True
    return raw[:5] == b"%PDF-"


def extract_text_from_pdf_bytes(raw: bytes) -> list[tuple[int, str]]:
    """
    Extract text per page from a PDF.
    Returns list of (1-based page_number, text). Requires `pypdf`.
    """
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PDF support needs pypdf. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(BytesIO(raw))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """Split long page/file text into overlapping chunks for retrieval."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        # prefer break on sentence/space
        if end < n:
            cut = text.rfind(". ", start, end)
            if cut < start + max_chars // 3:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def docs_from_upload_bytes(
    filename: str,
    raw: bytes,
    default_timestamp: str | None = None,
    *,
    max_chars: int = 1800,
) -> list[CorpusDoc]:
    """Turn one uploaded file (.txt / .md / .pdf) into CorpusDoc chunks."""
    ts = default_timestamp or date.today().isoformat()
    name = filename or "upload"
    lower = name.lower()

    if _is_pdf(name, raw):
        pages = extract_text_from_pdf_bytes(raw)
        if not pages:
            return []
        blobs: list[tuple[str, str, str]] = []
        for page_no, page_text in pages:
            for ci, chunk in enumerate(_chunk_text(page_text, max_chars=max_chars), start=1):
                title = f"{name} · p{page_no}" + (f" · part{ci}" if ci > 1 else "")
                blobs.append((title, chunk, ts))
        return docs_from_text_blobs(blobs, id_prefix="user_pdf")

    # Plain text / markdown / light structured text
    text_exts = (".txt", ".md", ".markdown", ".csv", ".json", ".log", ".text")
    looks_text = lower.endswith(text_exts) or (
        b"\x00" not in raw[:512] and not lower.endswith((".docx", ".doc", ".xlsx", ".zip", ".png", ".jpg"))
    )
    if not looks_text:
        return []

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    if "\x00" in text[:200]:
        return []
    blobs = []
    for ci, chunk in enumerate(_chunk_text(text, max_chars=max_chars), start=1):
        title = name if ci == 1 else f"{name} · part{ci}"
        blobs.append((title, chunk, ts))
    return docs_from_text_blobs(blobs, id_prefix="user_txt")


def docs_from_colab_uploads(
    uploaded: dict[str, bytes],
    default_timestamp: str | None = None,
    *,
    max_chars: int = 1800,
) -> list[CorpusDoc]:
    """`uploaded` is google.colab files.upload() dict: filename -> bytes (.txt / .pdf / .md)."""
    out: list[CorpusDoc] = []
    for fname, raw in uploaded.items():
        out.extend(
            docs_from_upload_bytes(
                fname,
                raw,
                default_timestamp=default_timestamp,
                max_chars=max_chars,
            )
        )
    return out


def merge_corpus_files(
    base_corpus_path: Path,
    extra_docs: list[CorpusDoc],
    out_path: Path,
    *,
    dedupe_by_text: bool = True,
) -> int:
    """Append user docs to existing corpus.jsonl; write merged file."""
    base = load_corpus(base_corpus_path) if Path(base_corpus_path).exists() else []
    seen = {d.text.strip() for d in base} if dedupe_by_text else set()
    merged = list(base)
    for d in extra_docs:
        if dedupe_by_text and d.text.strip() in seen:
            continue
        merged.append(d)
        if dedupe_by_text:
            seen.add(d.text.strip())
    write_jsonl(out_path, merged)
    return len(merged)


def rebuild_index(
    corpus_path: Path,
    artifacts_dir: Path,
    model_name: str | None = None,
) -> EmbeddingIndex:
    kwargs: dict = {}
    if model_name:
        kwargs["model_name"] = model_name
    return build_index_from_corpus(corpus_path, artifacts_dir, **kwargs)


class KnowledgeBase:
    """
    Presentation / Colab knowledge base with a multi-add loop.

    - Live corpus: data/corpus_live.jsonl (starts as copy of corpus.jsonl)
    - Index artifacts: artifacts/faiss.index + embeddings.npy + corpus_snapshot
    - add_documents() can be called many times; each call appends + updates FAISS
    """

    def __init__(
        self,
        data_dir: str | Path,
        artifacts_dir: str | Path,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.model_name = model_name
        self.device = device
        self.live_corpus = self.data_dir / "corpus_live.jsonl"
        self.base_corpus = self.data_dir / "corpus.jsonl"
        self.user_log = self.data_dir / "user_docs_log.jsonl"
        self.index = EmbeddingIndex(model_name=model_name, device=device)

    def ensure_live_corpus(self, reset: bool = False) -> int:
        """Create corpus_live from base corpus if missing (or reset=True)."""
        if reset or not self.live_corpus.exists():
            if not self.base_corpus.exists():
                raise FileNotFoundError(f"Missing base corpus: {self.base_corpus}")
            docs = load_corpus(self.base_corpus)
            write_jsonl(self.live_corpus, docs)
            if self.user_log.exists():
                self.user_log.unlink()
        return len(load_corpus(self.live_corpus))

    def load_or_build_index(self, force_rebuild: bool = False) -> EmbeddingIndex:
        """Load existing FAISS index, or build from live corpus."""
        self.ensure_live_corpus()
        faiss_path = self.artifacts_dir / "faiss.index"
        emb_path = self.artifacts_dir / "embeddings.npy"
        if force_rebuild or not faiss_path.exists() or not emb_path.exists():
            self.index = rebuild_index(self.live_corpus, self.artifacts_dir, self.model_name)
            return self.index
        self.index.load(self.artifacts_dir, corpus_path=self.live_corpus)
        # Prefer live corpus text over stale snapshot when counts mismatch
        live_docs = load_corpus(self.live_corpus)
        if len(live_docs) != len(self.index.docs):
            self.index = rebuild_index(self.live_corpus, self.artifacts_dir, self.model_name)
        return self.index

    def list_docs(self, user_only: bool = False, limit: int = 20) -> list[CorpusDoc]:
        docs = load_corpus(self.live_corpus) if self.live_corpus.exists() else []
        if user_only:
            docs = [d for d in docs if d.source == "user_upload"]
        return docs[-limit:]

    def stats(self) -> dict[str, Any]:
        docs = load_corpus(self.live_corpus) if self.live_corpus.exists() else []
        user = [d for d in docs if d.source == "user_upload"]
        return {
            "total_docs": len(docs),
            "user_docs": len(user),
            "base_docs": len(docs) - len(user),
            "live_corpus": str(self.live_corpus),
            "index_docs": len(self.index.docs) if self.index.docs else 0,
            "model_name": self.model_name,
        }

    def add_documents(
        self,
        docs: list[CorpusDoc],
        *,
        rebuild_full: bool = False,
    ) -> dict[str, Any]:
        """
        Append docs into the live knowledge base and update FAISS.

        Call this in a loop — each batch is added without wiping previous user docs.
        """
        if not docs:
            return {"added": 0, "total": self.stats()["total_docs"], "seconds": 0.0}

        t0 = time.time()
        self.ensure_live_corpus()
        existing = load_corpus(self.live_corpus)
        seen = {d.text.strip() for d in existing}
        new_docs: list[CorpusDoc] = []
        for d in docs:
            key = d.text.strip()
            if not key or key in seen:
                continue
            new_docs.append(d)
            seen.add(key)

        if not new_docs:
            return {
                "added": 0,
                "skipped_dupes": len(docs),
                "total": len(existing),
                "seconds": round(time.time() - t0, 2),
            }

        # Persist corpus + user log
        write_jsonl(self.live_corpus, existing + new_docs)
        with self.user_log.open("a", encoding="utf-8") as f:
            for d in new_docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

        if rebuild_full or not self.index.docs or self.index.embeddings is None:
            self.index = rebuild_index(self.live_corpus, self.artifacts_dir, self.model_name)
        else:
            self._incremental_add(new_docs)

        return {
            "added": len(new_docs),
            "skipped_dupes": len(docs) - len(new_docs),
            "total": len(existing) + len(new_docs),
            "new_ids": [d.doc_id for d in new_docs],
            "seconds": round(time.time() - t0, 2),
        }

    def _incremental_add(self, new_docs: list[CorpusDoc]) -> None:
        """Encode new docs and append into FAISS IndexFlatIP + embeddings.npy."""
        if faiss is None:
            raise ImportError("faiss is required")
        assert self.index.embeddings is not None and self.index.index is not None
        texts = [d.text for d in new_docs]
        emb = self.index.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 8,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.index.index.add(emb)
        self.index.embeddings = np.vstack([self.index.embeddings, emb])
        self.index.docs.extend(new_docs)
        self.index.save(self.artifacts_dir)

    def add_from_paste(
        self,
        title: str,
        text: str,
        timestamp: str,
    ) -> dict[str, Any]:
        return self.add_documents(docs_from_text_blobs([(title, text, timestamp)]))

    def add_from_uploads(
        self,
        uploaded: dict[str, bytes],
        default_timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.add_documents(
            docs_from_colab_uploads(uploaded, default_timestamp=default_timestamp)
        )

    def reset_to_base(self) -> dict[str, Any]:
        """Drop all user docs; restore corpus_live + full index from base corpus."""
        n = self.ensure_live_corpus(reset=True)
        self.index = rebuild_index(self.live_corpus, self.artifacts_dir, self.model_name)
        return {"total_docs": n, "user_docs": 0, "reset": True}
