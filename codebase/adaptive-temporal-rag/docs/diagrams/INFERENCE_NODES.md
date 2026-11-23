# Inference architecture — node-by-node guide

This document explains **every box and decision** in the inference PNGs under `docs/diagrams/`.  
Implementation lives mainly in `src/pipelines.py` (Adaptive + baselines), `src/query_analyzer.py`, `src/embed_index.py`, `src/rerank_ce.py`, `src/temporal_rank.py`, `src/evidence_sufficiency.py`, and `src/generate.py`.

| Diagram PNG | What it shows |
|-------------|----------------|
| `01_adaptive_inference_full.png` | End-to-end **Adaptive** pipeline |
| `02_policy_router.png` | How analyzer outputs become **retrieve_mode** |
| `03_colab_four_systems_eval.png` | **B1–B4** on the same test queries (Colab §4) |

At inference time the system **never** sees gold labels (`gold_answer`, `gold_doc_ids`). Those exist only for training the analyzer and for **evaluation metrics**.

---

## Diagram 01 — Adaptive inference (full path)

### User query

**What it is:** Raw natural-language text from the Colab demo widget, Streamlit, or a row in `queries.jsonl` (only the `query` field is passed to `pipeline.run()`).

**What it is not:** Not the full `QueryItem` labels during normal demo inference. In batch eval, `time_hint` from the dataset *can* be passed when the caller passes a `QueryItem` object — that helps historical window hints but is **not** the gold answer.

**Code:** `AdaptiveTemporalRAGPipeline.run()` → `_q(query)` extracts text and id.

---

### Trained query analyzer (DistilBERT / DeBERTa multitask)

**What it is:** A small neural model fine-tuned in Colab §3 (`analyzer.pt`). It reads the query once and produces structured routing signals.

**Architecture:**

- **Encoder:** Hugging Face transformer (T4 pilot: `distilbert-base-uncased`; A100 stage may use a larger name from `model_config.py`).
- **Heads:**  
  - *Necessity* — binary: should we treat this as needing external evidence?  
  - *Temporal* — multi-class: `current`, `historical`, or `timeless`.  
  - *Confidence* — scalar logit blended at inference (not the only training loss term).

**Output object (`AnalyzerOutput`):**

| Field | Meaning |
|-------|---------|
| `needs_retrieval` | `0` or `1` (argmax over necessity head) |
| `temporal_class` | `current` / `historical` / `timeless` |
| `confidence` | `0.5 * conf_head + 0.5 * max(nec_prob, temp_prob)` for stability |
| `necessity_prob` | P(needs retrieval) |
| `temporal_probs` | Full distribution over temporal classes |

**Important:** There is **no** separate `has_time_span` head. Years and dates for ranking come from **regex parsing** of the query (`temporal_rank.extract_year_hint`, etc.), not from the analyzer.

**Code:** `QueryAnalyzerService.predict()` in `src/query_analyzer.py`; loaded via `load_query_analyzer(artifacts_dir)`.

---

### needs_retrieval · temporal_class · confidence

**What it is:** The three primary routing signals copied into `PipelineResult.analyzer` and shown in the Colab demo.

**How they are used:**

- `needs_retrieval` + triviality check → hard skip retrieval (see below).
- `needs_retrieval == 0` without triviality → **soft retrieve** (still hits the index, semantic mode).
- `temporal_class` → preferred temporal ranking mode when confidence is high enough.
- `confidence` compared to `conf_threshold` (Colab default often `0.50`–`0.55`) → fallback to semantic retrieval when the model is unsure.

---

### Decision: needs_retrieval = 0 AND trivial query?

**What it is:** The only path that **skips the index entirely**.

**Trivial** means a small allowlist in `is_trivial_query()` — arithmetic (`2+2`), spelling drills, “days in a week”, etc. This is **not** learned; it prevents wasting retrieval on obvious closed-book items when the analyzer also says no retrieval.

**If yes:** Go to **Generator (no docs)** → **Answer + meta (no retrieval)** with `did_retrieve=False`, `policy: no_retrieval`.

**If no:** Continue to **Policy router** even when `needs_retrieval=0` (e.g. “Define photosynthesis” may still retrieve in semantic soft mode).

**Code:** `pipelines.py` lines ~203–216.

---

### Policy router

**What it is:** Deterministic Python logic (not a second neural net) that maps analyzer outputs + thresholds to a **named policy** string and a **retrieve_mode** (`current` | `historical` | `semantic`).

**Rules (simplified):**

1. **Soft timeless:** `needs_retrieval == 0` → retrieve anyway, semantic, policy `soft_retrieve_timeless`.
2. **Low confidence:** `confidence < conf_threshold` → semantic, policy `low_conf_semantic`.
3. **Timeless class** (but retrieval requested): semantic, policy `timeless_semantic`.
4. **Current:** `retrieve_mode = current`, policy `current`.
5. **Historical:** `retrieve_mode = historical`, policy `historical_adaptive_window`.
6. **Else:** semantic.

**Why it exists:** The analyzer proposes a *intent*; the policy enforces **safe fallbacks** when the model is uncertain or when “timeless” should not force a wrong temporal bias.

**Code:** `pipelines.py` ~221–236.

---

### retrieve_mode · current path · historical path · semantic path

**What they are:** Three entry labels into the same **temporal retriever** subgraph, with different post-FAISS behavior.

| Path | When chosen | Intent |
|------|-------------|--------|
| **current** | High-confidence `current` | Prefer **recent** documents (leadership, “who is PM now”, etc.) |
| **historical** | High-confidence `historical` | Prefer docs near a **target year/date** with an **adaptive window** |
| **semantic** | Soft no-retrieve, low confidence, timeless, or default | **Semantic similarity only** (minimal temporal pressure) |

Colab config often sets `ALPHA=0.55` for blending semantic vs temporal scores in ranking.

---

### Temporal retriever (subgraph) — shared first steps

All three paths run the same **candidate generation** before branching on `mode`.

#### BGE embed + FAISS (candidate_k neighbors)

**What it is:**

- **BGE-small** (`BAAI/bge-small-en-v1.5`) encodes the query with an instruction prefix:  
  `Represent this sentence for searching relevant passages: {query}`.
- **FAISS IndexFlatIP** on L2-normalized vectors → inner product ≈ cosine similarity.
- Returns up to **`candidate_k`** passages (Colab often `80`).

**Built in:** Colab §2 (`build_index_from_corpus`). Artifacts: `faiss.index`, `embeddings.npy`, corpus snapshot.

**Code:** `EmbeddingIndex.search_candidates()` in `src/embed_index.py`.

#### Cross-encoder rerank (ms-marco-MiniLM)

**What it is:** A **second-stage ranker** that scores each `(query, passage)` pair jointly — more accurate than embedding similarity alone.

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Keeps top **`ce_rerank_keep`** (default ~40) after FAISS.

**Why it matters:** Your diagram should always show **FAISS → CE** before temporal reranking; B2/B3/Adaptive all use this pattern when a reranker is configured.

**Code:** `CrossEncoderReranker`, `_rerank_faiss_candidates()` in `src/pipelines.py` / `src/rerank_ce.py`.

#### mode branch (inside retriever)

After CE, `_retrieve_rank()` switches on `retrieve_mode`:

---

### Temporal rerank — recency + alpha blend (`current`)

**What it is:** Re-scores CE candidates using:

- **Semantic score** from retrieval/rerank.
- **Temporal score** favoring **newer** timestamps (corpus reference around 2025).

**Formula (conceptual):**  
`final = α · semantic + (1 − α) · temporal`  
with `α` from pipeline config (and sometimes adjusted during sufficiency repairs).

**Code:** `rerank(..., mode="current")` in `src/temporal_rank.py`.

---

### Adaptive date window — expand until enough docs + alpha blend (`historical`)

**What it is:**

1. **`infer_adaptive_window()`** reads query (+ optional `time_hint`) to set:
   - target year or full date,
   - **initial** half-width in years (tight for exact dates, wider for decades/vague cues),
   - **max** half-width cap.
2. **`adaptive_filter_by_window()`** keeps docs whose timestamps fall in `[target ± years]`, **expanding** years until `min_docs` met or max reached.
3. **`rerank(..., mode="historical")`** blends semantic + temporal fit to the target period.

**Metadata:** Exposed in demo as `meta["adaptive_window"]` (target year, final ±years, expansions, reason).

**Repair interaction:** Sufficiency loop can **`widen_window`** via `window_boost` (+1.25 years per repair step).

**Code:** `infer_adaptive_window`, `adaptive_filter_by_window`, `rerank` in `src/temporal_rank.py`; wired in `_retrieve_rank()` historical branch.

---

### Semantic rerank — alpha blend (`semantic`)

**What it is:** Ranking that **does not** push strong recency or historical window filters — mostly semantic similarity (with light lexical overlap boosts inside `rerank`).

**When used:** Low analyzer confidence, timeless class, soft timeless retrieval, or after **`relax_to_semantic`** repair.

During repairs, α may be forced higher (e.g. `0.7`) to trust semantics more.

**Code:** `rerank(..., mode="semantic")`.

---

### Evidence sufficiency judge

**What it is:** A **fast heuristic judge** (no extra ML model) that scores whether retrieved passages are good enough to answer **before** (and optionally after) generation.

**Score components (weights):**

| Signal | Weight | Meaning |
|--------|--------|---------|
| Query term coverage in top docs | 0.35 | Do retrieved texts mention key content words? |
| Top retrieval score | 0.25 | Strength of best passage |
| Mean retrieval score | 0.15 | Overall pool quality |
| Temporal fit | 0.15 | Does top doc’s date match current/historical intent? |
| Lexical overlap boost | 0.10 | Extra overlap on top doc |

**Threshold:** Default `sufficiency_threshold` ≈ `0.55` (pipeline config).

**Outputs:** `sufficient` bool, `score`, `reasons[]`, `suggested_repairs[]`, logged in `meta["evidence_sufficiency"]` and `meta["sufficiency_loop"]`.

**Code:** `assess_evidence_sufficiency()` in `src/evidence_sufficiency.py`.

---

### Repairs (insufficient branch)

**What it is:** Mutations applied to retrieval **hyperparameters** before re-entering the temporal retriever. Up to **`max_sufficiency_rounds`** (default **2**) repair rounds per query.

| Repair action | Effect |
|---------------|--------|
| `expand_candidates` | Increase `candidate_k` (cap 200, ~×1.75 or +30) |
| `raise_topk` | Increase `top_k` passed to generator (cap 12, +3) |
| `widen_window` | Increase historical window half-width (`window_boost += 1.25`) |
| `relax_to_semantic` | Force semantic retrieve mode |
| `boost_recency` | Force recency-oriented ranking for current-like retrieval |

If no repair applies, the loop stops.

**Code:** `_apply_repairs()` in `src/pipelines.py`.

---

### Generator (extractive or API chain)

**What it is:** Turns `(query, optional docs)` into a short answer string.

**Backends (Colab):**

| Backend | Behavior |
|---------|----------|
| **extractive** | Picks/summarizes from top passages locally (no API key) |
| **api** | Tries providers in order: Groq → OpenRouter → OpenCode; falls back to extractive on failure |

**Prompting:** With docs, prompt includes numbered dated evidence lines; without docs, closed-book instruction.

**Output:** `GenerationResult` → `answer`, `backend` name (shown in UI).

**Code:** `Generator.generate()` in `src/generate.py`.

---

### Decision: Post-answer weak?

**What it is:** After the first answer, sufficiency is run again **with the answer text**. If markers like “insufficient evidence” / “I don’t know” appear (`weak_generated_answer`), the pipeline performs **one** extra retrieval:

- Semantic mode, larger `candidate_k`, higher α, wider window boost.
- Regenerates; keeps the retry only if sufficiency **score improves**.

**Policy suffix:** May append `+sufficiency_retry` to policy name in meta.

**Code:** `pipelines.py` ~310–347.

---

### One wider semantic re-retrieve · Regenerate if better

**What they are:** The concrete steps of the post-answer retry (not a full second sufficiency loop with multiple repair rounds).

---

### Answer + retrieved passages · policy · window · sufficiency meta

**What it is:** Final `PipelineResult`:

| Field | Content |
|-------|---------|
| `answer` | Generated string |
| `retrieved_docs` | Top passages (with timestamps, text) |
| `retrieved_scores` | Blended ranking scores |
| `did_retrieve` | Whether index was used |
| `analyzer` | Full analyzer output |
| `meta` | `policy`, `adaptive_window`, `evidence_sufficiency`, `sufficiency_loop`, generator errors |

**Citations:** The demo lists passages as expandable evidence blocks; the generator prompt asks for dates but does not enforce formal citation IDs like `[1]` in all backends.

---

### Generator (no docs) · Answer + meta (no retrieval)

**What they are:** Terminal nodes for the **trivial skip** path — fastest latency, `retrieved_docs` empty, `sufficiency_loop` empty list.

---

## Diagram 02 — Policy router (detail)

This diagram zooms into the box **Policy router** + **retrieve_mode** from diagram 01.

### Analyzer output

Same as **Trained query analyzer** outputs above — input to all following diamonds.

### trivial and no retrieve?

Combines `needs_retrieval == 0` with `is_trivial_query()`. **Yes** → **Skip retrieval** (generator only). **No** → continue.

### confidence low OR timeless OR soft no-retrieve?

**Yes** → **semantic** retrieve mode (safe fallback or intentional soft RAG).

- *Low confidence:* model unsure → do not trust aggressive temporal filtering.  
- *Timeless:* definitional / stable facts → semantic.  
- *Soft no-retrieve:* analyzer said no retrieval but query is not trivial → still retrieve semantically.

### temporal_class (when confident enough)

| Branch | retrieve_mode |
|--------|----------------|
| **current** | Recency-biased ranking |
| **historical** | Adaptive window + historical temporal score |
| **else** | **semantic** |

### Skip retrieval

Terminal for trivial path — not shown on diagram 02 but implied as the “yes” exit from the first diamond.

---

## Diagram 03 — Colab four-system eval

**Context:** Colab **§4** runs `evaluate.py --split test` so every system sees the **same** test queries. Gold labels are used **only** when computing metrics afterward.

### Same test query text

Each pipeline’s `run()` receives the query string (or `QueryItem` without feeding gold into routing/generation).

### B1: Generator only

**Pipeline:** `LLMOnlyPipeline`.

**Flow:** Query → generator with **no** FAISS, **no** analyzer, **no** temporal logic.

**Purpose:** Baseline “parametric knowledge only” — shows value of retrieval on temporal/factual items.

### B2: FAISS then CE then Generator

**Pipeline:** `StandardRAGPipeline`.

**Flow:** FAISS candidates → cross-encoder rerank → top-k docs → generator.

**Purpose:** Classic RAG without temporal adaptation.

### B3: FAISS then CE then fixed window then Generator

**Pipeline:** `FixedTemporalRAGPipeline`.

**Flow:** Same retrieval as B2, then **fixed** temporal policy (e.g. ±2 year window when a year appears, else recency-style ranking) → generator.

**Purpose:** Shows **rigid** temporal RAG vs Adaptive’s learned routing + adaptive window.

### B4 Adaptive: analyzer + temporal retriever + sufficiency

**Pipeline:** `AdaptiveTemporalRAGPipeline` — entire diagram 01.

**Purpose:** Proposed system combining learned routing, adaptive historical windows, and evidence sufficiency loop.

### Metrics vs gold (not used at run time)

**What it is:** After all systems answer, `evaluate.py` compares predictions to `gold_answer` / `gold_doc_ids` / temporal labels:

- Answer quality: contains, exact match, token F1  
- Retrieval: precision/recall @5  
- `temporal_acc` where applicable  
- Latency, retrieval rate  

**Outputs:** `results.csv`, `results_summary.json`, charts, `comparative_analysis.md`.

**Critical for report honesty:** High **validation** accuracy during analyzer training does not replace this **test** comparison across B1–B4.

---

## Configuration knobs (Colab ↔ inference)

These Colab variables directly affect nodes above:

| Knob | Typical effect |
|------|----------------|
| `TOP_K` | How many passages reach the generator |
| `CANDIDATE_K` | FAISS pool size before CE |
| `ALPHA` | Semantic vs temporal blend in rerank |
| `CONF_THRESHOLD` | Policy router semantic fallback |
| `GENERATOR` | `api` vs `extractive` |
| `max_sufficiency_rounds` | Repair loop length (pipeline default 2) |

---

## Quick map: node → source file

| Node | Primary file |
|------|----------------|
| Analyzer | `src/query_analyzer.py` |
| Policy + Adaptive pipeline | `src/pipelines.py` |
| FAISS / BGE | `src/embed_index.py` |
| Cross-encoder | `src/rerank_ce.py` |
| Temporal + adaptive window | `src/temporal_rank.py` |
| Sufficiency + repairs | `src/evidence_sufficiency.py` |
| Generator + trivial check | `src/generate.py` |
| Eval harness | `scripts/evaluate.py` |

---

## Related assets

- PNGs: `01_adaptive_inference_full.png`, `02_policy_router.png`, `03_colab_four_systems_eval.png`
- Mermaid sources: `*.mmd` in this folder
- Regenerate PNGs: `.\scripts\render_diagram_pngs.ps1`
