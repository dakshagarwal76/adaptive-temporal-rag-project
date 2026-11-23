"""Extractive reader and a small learned sufficiency model.

The reader is a frozen SQuAD 2.0 model. Its no-answer score, not a hand-weighted
sum, decides whether to answer, read again, or abstain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data_schema import CorpusDoc

READER_MODEL = "deepset/tinyroberta-squad2"
ABSTAIN_TEXT = "Insufficient evidence."


@dataclass
class ReaderHit:
    answer: str
    score: float
    no_answer_score: float
    doc_id: str


class SquadReader:
    def __init__(self, model_name: str = READER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            from .hf_utils import from_pretrained_local_first

            self._torch = torch
            self._tokenizer = from_pretrained_local_first(AutoTokenizer.from_pretrained, self.model_name)
            if getattr(self._tokenizer, "vocab_size", 0) < 1000:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=False)
            self._model = from_pretrained_local_first(
                AutoModelForQuestionAnswering.from_pretrained, self.model_name
            )
            self._model.eval()
        return self._model

    def _span(self, question: str, context: str) -> tuple[str, float, float]:
        import math

        model = self._ensure()
        encoded = self._tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0]
        sequence_ids = encoded.sequence_ids(0)
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in ("input_ids", "attention_mask")
        }
        with self._torch.no_grad():
            output = model(**model_inputs)
        start_logits = output.start_logits[0]
        end_logits = output.end_logits[0]
        start_probs = self._torch.softmax(start_logits, dim=-1)
        end_probs = self._torch.softmax(end_logits, dim=-1)
        null_logit = float(start_logits[0] + end_logits[0])
        best_logit = -1e9
        best_span: tuple[int, int] | None = None
        length = int(start_logits.shape[0])
        for start in range(length):
            if sequence_ids[start] != 1:
                continue
            stop = min(start + 32, length)
            for end in range(start, stop):
                if sequence_ids[end] != 1:
                    break
                logit = float(start_logits[start] + end_logits[end])
                if logit > best_logit:
                    best_logit = logit
                    best_span = (start, end)
        margin = best_logit - null_logit
        margin = max(-20.0, min(20.0, margin))
        no_answer = 1.0 - (1.0 / (1.0 + math.exp(-margin)))
        if best_span is None:
            return "", 0.0, no_answer
        start, end = best_span
        text = context[int(offsets[start][0]) : int(offsets[end][1])].strip()
        score = float(start_probs[start] * end_probs[end])
        return text, score, no_answer

    def read(self, question: str, docs: list[CorpusDoc], top_n: int = 3) -> ReaderHit:
        if not docs:
            return ReaderHit(ABSTAIN_TEXT, 0.0, 1.0, "")
        from .generate import _best_dated_sentence, _sentences

        # If the top document is a multi-sentence passage, first read from the
        # date-matched sentence so other years in the same chunk cannot distract the span reader.
        if len(_sentences(docs[0].text)) > 1:
            dated_sent = _best_dated_sentence(question, docs)
            if dated_sent:
                try:
                    ans, sc, na = self._span(question, dated_sent)
                    if ans.strip() and sc >= 0.02:
                        return ReaderHit(ans.strip(" ,.;:"), sc, na, docs[0].doc_id)
                except Exception:
                    pass

        best: ReaderHit | None = None
        for idx, doc in enumerate(docs[:top_n]):
            try:
                answer, score, no_answer = self._span(question, doc.text[:2000])
            except Exception:
                continue
            if not answer.strip():
                continue
            hit = ReaderHit(answer.strip(" ,.;:"), score, no_answer, doc.doc_id)
            # Trust the top-ranked temporal document when it yields a confident span.
            if idx == 0 and hit.score >= 0.05:
                return hit
            if best is None or hit.score > best.score:
                best = hit
        return best or ReaderHit(ABSTAIN_TEXT, 0.0, 1.0, "")


class SufficiencyModel:
    def __init__(self, model: LogisticRegression, threshold: float = 0.5, scaler=None) -> None:
        self.model = model
        self.threshold = threshold
        self.scaler = scaler

    def decide(self, reader_score: float, no_answer_score: float, rank_score: float) -> tuple[bool, float]:
        row = np.asarray([[reader_score, no_answer_score, rank_score]], dtype=np.float32)
        if self.scaler is not None:
            row = self.scaler.transform(row)
        probability = float(self.model.predict_proba(row)[0, 1])
        is_sufficient = probability >= self.threshold or (reader_score >= 0.08 and rank_score >= 1.0)
        return is_sufficient, probability

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str | Path) -> "SufficiencyModel":
        payload = joblib.load(path)
        return cls(
            payload["model"],
            threshold=float(payload.get("threshold", 0.5)),
            scaler=payload.get("scaler"),
        )


def train_sufficiency(rows: list[list[float]], labels: list[int]) -> SufficiencyModel:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(rows, dtype=np.float32))
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(scaled, np.asarray(labels, dtype=np.int64))
    return SufficiencyModel(model, scaler=scaler)
