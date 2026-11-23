# Report sections to paste into the final project report

Use `artifacts/results_summary.json` + `comparative_analysis.md` from the **2026-10-08** Colab export (`atr_colab_artifacts_20261008_0451`) — **real** held-out test eval, n=77.

## 7. Proposed System Architecture

Adaptive Temporal RAG inserts a **query analyzer** before retrieval. For each query the analyzer predicts:

1. **Retrieval necessity** — whether external evidence is required  
2. **Temporal class** — `current`, `historical`, or `timeless`  
3. **Confidence** — reliability of the above decisions  

A policy module then selects one of:

- **No retrieval** → generator answers from parametric knowledge only  
- **Current** → FAISS top-k candidates re-ranked with a **recency** temporal score  
- **Historical** → **adaptive date window** (starts tight from query specificity — exact date / year / decade / vague — then expands until enough evidence) + temporal re-rank  
- **Low confidence / timeless** → semantic-only ranking (safer fallback)  

B3 Fixed Temporal keeps a rigid ±2 year window for comparison; Adaptive varies the window.

**Evidence sufficiency feedback loop (Adaptive only):** after ranking, a heuristic judge scores coverage, retrieval strength, and temporal fit. If evidence is insufficient, the system applies repairs (widen window, expand candidates, raise top-k, relax to semantic, boost recency) and re-retrieves (up to 2 rounds), then optionally retries once more if the generated answer still looks weak.

Final ranking uses:

`final_score = α · semantic_score + (1 − α) · temporal_score` (default `α ≈ 0.55–0.6`).

Components: BGE-small embeddings, FAISS IndexFlatIP, cross-encoder rerank, DistilBERT (T4) / DeBERTa (A100) multitask analyzer, multi-provider API answers (Groq → OpenRouter → OpenCode), evidence sufficiency loop, Streamlit + Colab demos. Colab-only pipeline diagrams: **ARCHITECTURE.md** (`00_colab_e2e.ipynb` sections 0–6).

## 8. Dataset

- **Corpus:** ~800 short dated passages covering politics, business leadership, sports, awards, laws, and timeless encyclopedia facts. High-churn entities have multiple timestamps so historical vs current ranking differs.  
- **Queries:** ~1050 labeled items with `needs_retrieval`, `temporal_class`, optional `time_hint`, `gold_answer`, and `gold_doc_ids`. Split **70/15/15** train/val/test; train includes paraphrase augmentation.  
- Approx. balance: ~35% current / ~30% historical / ~30% timeless; ~25% no-retrieval.

## 9. Experimental Setup

Four systems compared on the same test split:

| ID | System | Retrieval |
|----|--------|-----------|
| B1 | LLM-only | Never |
| B2 | Standard RAG | Always, semantic only |
| B3 | Fixed Temporal RAG | Always, fixed time policy |
| Proposed | Adaptive Temporal RAG | Conditional + class-specific policy |

Metrics: answer contains/EM/token-F1 vs gold, temporal accuracy (evidence-date heuristic), precision@k / recall@k when gold docs exist, latency, retrieval rate.

## 10. Results (Colab T4 — export `20261008_0451`)

**Authoritative artifacts:** `artifacts/results_summary.json`, `results.csv`, `results_charts.png`, `comparative_analysis.md` from the held-out **test** eval (n=77). These are **real** `evaluate.py` outputs, not synthetic placeholders.

### 10.1 Analyzer training (successful DistilBERT run)

| Field | Value |
|--------|--------|
| Stage | `t4_pilot` |
| Encoder | `distilbert-base-uncased` |
| Train seeds / after augment | 2849 → 17094 |
| Val | 74 |
| Augment / epochs / batch | 6 / 15 (early stop) / 8 |
| Wall time | ~22 min |
| Best val score | **1.0** (small synthetic val — report **test** metrics below) |

Smoke predictions after train: current PM → `current`; US president 2018 → `historical`; photosynthesis / 2+2 → `timeless`.

### 10.2 Comparative test results (real eval)

| System | Contains | EM | Token F1 | Temp Acc | P@5 | R@5 | Ret Rate | Latency (s) |
|--------|----------|-----|----------|----------|-----|-----|----------|-------------|
| B1 LLM-only | 0.117 | 0.104 | 0.145 | 0.247 | 0.000 | 0.000 | 0.000 | 0.432 |
| B2 Standard RAG | 0.688 | 0.377 | 0.505 | 0.636 | 0.390 | 0.649 | 1.000 | 0.572 |
| B3 Fixed Temporal | 0.844 | 0.442 | 0.609 | 0.883 | 0.547 | 0.759 | 1.000 | 0.498 |
| Adaptive (ours) | 0.844 | 0.442 | 0.605 | 0.883 | 0.569 | 0.736 | 0.883 | 0.516 |

**Takeaways:** Adaptive matches B3 on Contains / Temp Acc while retrieving on **88%** of queries (vs 100% for B2/B3). vs B2: **+15.6%** Contains, **+24.7%** Temp Acc. Full narrative in `comparative_analysis.md`.

## 11. Error Analysis

Common remaining errors:

- Ambiguous “current” questions when the corpus’s latest snapshot is slightly stale relative to real world  
- Historical questions without an explicit year (analyzer may mark `current`)  
- Extractive generator truncating multi-token names  

## 12. Conclusion & Future Work

Adaptive Temporal RAG shows that jointly modeling retrieval necessity and temporal intent improves time-aware QA on a compact, evaluable dataset suitable for a minor project. Future work: stronger generators (Phi-3 / API LLMs), learned temporal scorers, larger FreshQA-style corpora, and FastAPI deployment.
