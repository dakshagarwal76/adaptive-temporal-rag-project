"""Small BM25 index over in-memory corpus texts."""

from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.N = len(texts)
        self.dl: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(texts):
            tf: dict[str, int] = {}
            for tok in tokenize(text):
                tf[tok] = tf.get(tok, 0) + 1
            length = sum(tf.values())
            self.dl.append(length or 1)
            for tok, count in tf.items():
                postings[tok].append((i, count))
        self.postings = dict(postings)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        self.idf = {
            tok: math.log(1.0 + (self.N - len(plist) + 0.5) / (len(plist) + 0.5))
            for tok, plist in self.postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        acc: dict[int, float] = {}
        seen: set[str] = set()
        for tok in tokenize(query):
            if tok in seen:
                continue
            seen.add(tok)
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for doc_i, tf in self.postings[tok]:
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.dl[doc_i] / self.avgdl)
                acc[doc_i] = acc.get(doc_i, 0.0) + idf * (tf * (self.k1 + 1.0) / denom)
        return acc

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]
