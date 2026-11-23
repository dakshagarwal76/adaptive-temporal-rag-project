# Colab pipeline architecture

Single end-to-end path: **`notebooks/00_colab_e2e.ipynb`** on a GPU runtime (T4 pilot or A100 large).  
Upload **`adaptive-temporal-rag-colab.zip`** to Drive (or upload the zip in the setup cell), then run cells top to bottom.

---

## 0 — Runtime layout

```mermaid
flowchart LR
    ZIP[adaptive-temporal-rag-colab.zip\non Google Drive]
    COLAB[Colab GPU runtime\nT4 or A100]
    ROOT[Project ROOT\n/content/.../adaptive-temporal-rag]
    ART[(artifacts/)]
    DATA[(data/)]

    ZIP --> COLAB
    COLAB --> ROOT
    ROOT --> ART
    ROOT --> DATA
```

| Path | Purpose |
|------|---------|
| `ROOT/` | Unzipped project (`src/`, `scripts/`, notebook) |
| `ROOT/data/` | `corpus.jsonl`, `queries.jsonl` (section 1) |
| `ROOT/artifacts/` | Index, trained analyzer, eval outputs (sections 2–6) |

**Setup cell:** find `ROOT`, `pip install`, check `torch.cuda`, `os.chdir(ROOT)`.

---

## Colab stages (notebook order)

```mermaid
flowchart TB
    S0[0 Setup + Config + API keys]
    S1[1 Build dataset]
    S2[2 Build BGE + FAISS index]
    S3[3 Train query analyzer]
    S4[4 Evaluate B1/B2/B3/Adaptive]
    S5[5 Interactive demo widgets]
    S6[6 Zip download artifacts + data]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

| Section | What runs | Main outputs |
|---------|-----------|----------------|
| **Config** | `RUN_STAGE` (`t4_pilot` / `a100_large`), `TARGET_QUERIES`, `AUGMENT_FACTOR`, `EPOCHS`, `BATCH_SIZE`, `GENERATOR`, `TOP_K`, `ALPHA`, `CANDIDATE_K` | Console config + GPU name |
| **API keys** | Groq → OpenRouter → OpenCode; if none → `GENERATOR = extractive` | Provider chain healthcheck |
| **1 — Dataset** | `python scripts/build_dataset.py --target-queries … --out-dir data/` | `data/corpus.jsonl`, `data/queries.jsonl` |
| **2 — Index** | `build_index_from_corpus(corpus.jsonl, artifacts/)` | `faiss.index`, `embeddings.npy`, snapshot meta |
| **3 — Train** | `train_analyzer(...)` on train/val splits from `queries.jsonl` | `analyzer.pt`, `analyzer_meta.json` |
| **4 — Eval** | `evaluate.py --split test` + `plot_results.py` | `results.csv`, `results_summary.json`, charts, comparative report |
| **5 — Demo** | ipywidgets: pick B1/B2/B3/Adaptive, run one query | Answer + analyzer + retrieved docs in notebook |
| **6 — Download** | Zip `artifacts/` + `data/` → `files.download` | `atr_colab_artifacts_*.zip` |
| **8 — Inference trace** | `src/inference_trace.py` — B1–B4 + step log | `artifacts/inference_walkthrough.json` |

---

## Section 1 — Dataset (inside Colab)

```mermaid
flowchart LR
    TQ[TARGET_QUERIES\n3000 T4 / 7000 A100]
    BD[build_dataset.py]
    CORP[corpus.jsonl\ndated + timeless docs]
    QRY[queries.jsonl\nlabels + splits]

    TQ --> BD
    BD --> CORP
    BD --> QRY
```

- **Splits:** train / val / test (stratified on seed templates; paraphrases extend train size).
- **Labels** (for training analyzer + eval metrics only): `needs_retrieval`, `temporal_class`, `time_hint`, `gold_answer`, `gold_doc_ids`.
- **Inference** (sections 4–5) does **not** read gold labels.

---

## Section 2 — Index (no neural training)

```mermaid
flowchart TB
    CORP[corpus.jsonl]
    BGE[BAAI/bge-small-en-v1.5\nencode all passages]
    FAISS[FAISS IndexFlatIP]
    ART[artifacts/]

    CORP --> BGE --> FAISS --> ART
```

Used by B2, B3, and Adaptive at eval/demo time. Cross-encoder rerank loads later at inference (not a separate Colab section).

---

## Section 3 — Train analyzer (only learned weights in Colab)

```mermaid
flowchart TB
    QRY[queries.jsonl]
    TR[split == train]
    VA[split == val]
    AUG[AUGMENT_FACTOR\nsynthetic paraphrase prefixes]
    DS[QueryDataset + DataLoader]
    ENC[DistilBERT T4 pilot\nDeBERTa on A100 stage]
    H[Multitask heads:\nnecessity + temporal + confidence]
    LOSS[Weighted CE loss]
    ES[Early stop on val F1]
    PT[analyzer.pt]

    QRY --> TR --> AUG --> DS
    QRY --> VA
    DS --> ENC --> H --> LOSS --> ES --> PT
    VA --> ES
```

| `RUN_STAGE` | Typical encoder | `TARGET_QUERIES` | Notes |
|-------------|-----------------|------------------|--------|
| `t4_pilot` | `distilbert-base-uncased` | 3000 | ~2–3h target on T4 |
| `a100_large` | larger preset in `model_config` | 7000 | ~6h final-quality run |

After training, the notebook runs a few **smoke predictions** (current PM, 2018 president, define term, arithmetic).

---

## Section 4 — Eval (batch inference on test split)

```mermaid
flowchart TB
    LOAD[Load faiss.index + analyzer.pt]
    PIP[Four pipelines:\nB1, B2, B3, Adaptive]
    TEST[queries split == test]
    GEN[GENERATOR:\napi chain or extractive]

    LOAD --> PIP
    TEST --> PIP
    GEN --> PIP
    PIP --> MET[Per-system metrics]
    MET --> OUT[results.csv\nresults_summary.json\nresults_charts.png]
```

**Adaptive path in eval** (same as demo, no gold at `run()`):

```mermaid
flowchart LR
    Q[test query text] --> AN[analyzer.pt]
    AN --> POL[policy:\ncurrent / historical / semantic / skip trivial]
    POL --> RET[BGE + FAISS + CE rerank]
    RET --> TR[temporal re-rank\nadaptive window if historical]
    TR --> SUF[evidence sufficiency\n≤2 repair rounds]
    SUF --> G[generator]
    G --> A[predicted answer]
```

| System | Colab behavior |
|--------|----------------|
| **B1** | Generator only, no retrieval |
| **B2** | FAISS + CE rerank → generator |
| **B3** | Fixed temporal window + α blend → generator |
| **Adaptive** | Analyzer routing + adaptive historical window + sufficiency loop → generator |

---

## Inference diagrams (PNG for report / slides)

Ready-made PNGs (regenerate: `.\scripts\render_diagram_pngs.ps1`):

| PNG | Use |
|-----|-----|
| [01_adaptive_inference_full.png](docs/diagrams/01_adaptive_inference_full.png) | Full Adaptive inference path |
| [02_policy_router.png](docs/diagrams/02_policy_router.png) | Policy router detail |
| [03_colab_four_systems_eval.png](docs/diagrams/03_colab_four_systems_eval.png) | B1–B4 eval in Colab §4 |

Sources: `docs/diagrams/*.mmd`  
**Node-by-node guide:** [docs/diagrams/INFERENCE_NODES.md](docs/diagrams/INFERENCE_NODES.md)

---

## Section 5 — Demo (single-query inference)

Same four systems as section 4, driven by widgets (`system_dd`, `query_box`).  
Shows **answer**, **analyzer fields**, **policy / window / sufficiency meta**, and **retrieved passages**.

---

## Section 6 — Artifact bundle

Download zip includes at minimum:

- `artifacts/analyzer.pt`, `faiss.index`, `embeddings.npy`
- `artifacts/results.csv`, `results_charts.png` (after section 4)
- `data/corpus.jsonl`, `data/queries.jsonl`

---

## Models touched in Colab (summary)

| Step | Model | Trained in Colab? |
|------|--------|-------------------|
| Index | BGE-small + FAISS | No — encode once |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | No — used at eval/demo |
| Analyzer | DistilBERT (T4) / DeBERTa (A100) | **Yes** — section 3 |
| Answers | API failover or extractive | No |

Routing is **learned analyzer only** (no rule-based router).

---

## Viva one-liner (Colab only)

On Colab we **build data → index with BGE → fine-tune the multitask analyzer → run the same test queries through four pipelines → demo interactively → download the artifact zip** for the report.
