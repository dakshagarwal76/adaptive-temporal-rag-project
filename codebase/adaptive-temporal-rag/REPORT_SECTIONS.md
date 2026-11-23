# Report sections to paste into the final project report

Authoritative numbers: `artifacts/final_results.md` and `artifacts/final_results.json` from the frozen Checkpoint B eval (`configs/final.yaml`). Do not quote the older Colab 0.844 / local 0.600 / Checkpoint A 0.468 figures.

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

## 10. Results (frozen Checkpoint B)

**Authoritative artifacts:** `artifacts/final_results.md`, `artifacts/final_results.json`, `artifacts/general_eval/final/`. Four systems share the extractive reader, answer projection, and entity/surname cut. They differ on skip vs always-retrieve, the time window, and the sufficiency loop.

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

### 10.2 Comparative test results (frozen eval)

Master table from `artifacts/final_results.md`:

| Dataset | System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic | B1 no retrieval | 77 | 0.130 | 0.167 | 0.130 | 0.000 | 0.000 | 0.247 | 0.000 | 0.000 | 0.000 |
| synthetic | B2 standard | 77 | 0.649 | 0.676 | 0.727 | 0.411 | 0.655 | 0.662 | 1.000 | 1.551 | 1.432 |
| synthetic | B3 fixed ±2y | 77 | 0.935 | 0.951 | 0.987 | 0.608 | 0.943 | 1.000 | 1.000 | 1.555 | 1.452 |
| synthetic | Adaptive | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.690 | 1.573 |
| heldout | B1 no retrieval | 193 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | B2 standard | 193 | 0.332 | 0.361 | 0.409 | 0.346 | 0.577 | 0.596 | 1.000 | 1.569 | 1.489 |
| heldout | B3 fixed ±2y | 193 | 0.684 | 0.696 | 0.710 | 0.591 | 0.879 | 1.000 | 1.000 | 1.546 | 1.410 |
| heldout | Adaptive | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 2.584 | 1.715 |
| timeqa | B1 no retrieval | 500 | 0.000 | 0.013 | 0.014 | 0.000 | 0.000 | — | 0.000 | 0.000 | 0.000 |
| timeqa | B2 standard | 500 | 0.006 | 0.077 | 0.260 | 0.235 | 0.468 | — | 1.000 | 6.056 | 6.118 |
| timeqa | B3 fixed ±2y | 500 | 0.010 | 0.089 | 0.270 | 0.237 | 0.392 | — | 1.000 | 6.074 | 6.116 |
| timeqa | Adaptive | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 8.736 | 6.492 |

**Takeaways:** On the templated test Adaptive ties B3 on Contains (0.987) and is one extra exact match (0.948 vs 0.935) while skipping 10 of 77 queries. On the six-entity split it is ahead of B3 on EM (0.762 vs 0.684) and Contains (0.762 vs 0.710) and always retrieves. On TimeQA n=500 it is ahead on Contains (0.296 vs 0.270) and R@5 (0.488 vs 0.392) and behind by one EM (0.008 vs 0.010). Latency does not improve. Ablations: see `artifacts/final_results.md`.

## 11. Error Analysis

Common remaining errors:

- Ambiguous “current” questions when the corpus’s latest snapshot is slightly stale relative to real world  
- Historical questions without an explicit year (analyzer may mark `current`)  
- Extractive generator truncating multi-token names  

## 12. Conclusion & Future Work

Adaptive Temporal RAG shows that jointly modeling retrieval necessity and temporal intent improves time-aware QA on a compact, evaluable dataset suitable for a minor project. Future work: stronger generators (Phi-3 / API LLMs), learned temporal scorers, larger FreshQA-style corpora, and FastAPI deployment.
