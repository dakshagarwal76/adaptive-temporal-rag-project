# How Colab end-to-end actually works

## What you might expect vs what this notebook does

| Expectation | This project (Colab) |
|-------------|----------------------|
| Upload PDFs → ask questions | **§1 auto-builds** `data/corpus.jsonl` (synthetic dated passages). That file **is** the knowledge base. |
| Routing needs your labels | **Routing is learned** (`analyzer.pt`). You only type a **question** at query time. |
| One “chat” cell | **§5** widget demo + **§5½** upload-your-text demo (optional). |

## The real E2E chain

```
§1 build_dataset.py  →  corpus.jsonl + queries.jsonl (labels for train/eval only)
§2 BGE + FAISS      →  artifacts/faiss.index (searchable embeddings)
§3 train_analyzer   →  artifacts/analyzer.pt (needs_retrieval + temporal_class)
§4 evaluate.py      →  B1/B2/B3/Adaptive on TEST queries (metrics)
§5 demo            →  YOU type a question → pipeline runs analyzer → retrieve → answer
§5½ (optional)     →  YOU upload/paste text → re-index → same §5 routing on YOUR docs
```

## At inference (§5 / §5½), for one question

1. **Query analyzer** (`analyzer.pt`) — no gold labels.
2. **Policy** — current / historical / semantic / skip trivial.
3. **BGE + FAISS** → **cross-encoder** → temporal rerank.
4. **Evidence sufficiency** loop (Adaptive only).
5. **Generator** — extractive or API.

Gold `gold_answer` / `gold_doc_ids` are **never** passed into `pipeline.run()`.

## Optional: your own documents

Use **§5½** (or presentation §4) to append pasted text, **`.txt`**, or **`.pdf`** pages to the corpus and **update FAISS**. PDFs are extracted with `pypdf` (one chunk per page/part). Then ask questions over the live KB (base + your uploads).
