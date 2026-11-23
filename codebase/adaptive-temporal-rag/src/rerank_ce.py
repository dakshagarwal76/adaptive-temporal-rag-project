"""Cross-encoder reranking over FAISS candidates (lazy-loaded)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .data_schema import CorpusDoc
from .model_config import DEFAULT_RERANKER_MODEL

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_reranker_cache: dict[str, "CrossEncoderReranker"] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._device = device

    def _ensure_model(self) -> "CrossEncoder":
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[CorpusDoc, float]],
        keep: int = 5,
    ) -> list[tuple[CorpusDoc, float]]:
        if not candidates:
            return []
        keep = max(1, min(keep, len(candidates)))
        model = self._ensure_model()
        pairs = [[query, d.text] for d, _ in candidates]
        ce_scores = model.predict(pairs, show_progress_bar=False)
        faiss_scores = [s for _, s in candidates]
        f_min, f_max = min(faiss_scores), max(faiss_scores)
        f_span = max(f_max - f_min, 1e-6)
        blended: list[tuple[CorpusDoc, float]] = []
        for (doc, f_sc), ce in zip(candidates, ce_scores):
            f_norm = (f_sc - f_min) / f_span
            ce_f = float(ce)
            score = 0.35 * f_norm + 0.65 * ce_f
            blended.append((doc, score))
        blended.sort(key=lambda x: x[1], reverse=True)
        return blended[:keep]


def get_cross_encoder_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str | None = None,
    enabled: bool = True,
) -> Optional[CrossEncoderReranker]:
    if not enabled:
        return None
    key = f"{model_name}:{device or 'auto'}"
    if key not in _reranker_cache:
        _reranker_cache[key] = CrossEncoderReranker(model_name=model_name, device=device)
    return _reranker_cache[key]
