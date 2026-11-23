# Adaptive Temporal RAG

Colab-first implementation of **Adaptive Temporal RAG**: decide *whether* to retrieve and *what temporal policy* to use for time-sensitive QA.

**New to this project?** Start here → **[HOW_TO_RUN.md](HOW_TO_RUN.md)** (step-by-step beginner guide).  
**Colab pipeline architecture:** **[ARCHITECTURE.md](ARCHITECTURE.md)**.  
**How E2E works (corpus vs upload vs query):** **[docs/COLAB_E2E_FLOW.md](docs/COLAB_E2E_FLOW.md)**.  
**Presentation demo (no retrain):** upload **`atr_presentation.zip`** → open **`notebooks/01_colab_presentation.ipynb`** ([README_PRESENTATION.md](README_PRESENTATION.md)).

Based on the USAR minor-project synopsis: *When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering*.

## What this repo includes

| Piece | Details |
|-------|---------|
| Dataset | Dated corpus + labeled queries (Colab E2E uses compact ~700 queries; local full build ~1050) |
| Colab | **`notebooks/00_colab_e2e.ipynb`** — full pipeline + in-notebook demo + artifact download |
| Index | **BGE-small** embeddings + FAISS + **cross-encoder rerank** |
| Analyzer | **DistilBERT** (T4) / **DeBERTa-v3** (A100) multitask router |
| Systems | B1 LLM-only · B2 Standard RAG · B3 Fixed Temporal RAG · Adaptive (adaptive window + sufficiency loop) |
| Demo | Streamlit app with system toggle + **Comparative analysis** page |
| Comparison | `scripts/plot_results.py` → charts + `comparative_analysis.md` |

## Quick start (local)

```bash
cd adaptive-temporal-rag
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_dataset.py --target-queries 1050
python scripts/build_index.py
python scripts/train_analyzer.py --epochs 6
python scripts/evaluate.py --split test --generator extractive
streamlit run app/streamlit_app.py
```

## Colab workflow (T4 pilot ~2–3h → optional A100 ~6h)

1. Set API keys in `.env` (Groq / OpenRouter / OpenCode — automatic failover).
2. Upload this folder (or a zip) to Drive / Colab.
3. Runtime → GPU **T4** for the pilot.
4. Open **`notebooks/00_colab_e2e.ipynb`** with `RUN_STAGE = "t4_pilot"`:
   - **3000 queries**, seed-only train, `AUGMENT_FACTOR=4`, `EPOCHS=20`, DeBERTa-small (~1–2h train on T4)
5. Later for final quality: set `RUN_STAGE = "a100_large"` on **A100** (~6h).
6. Download artifacts → local:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Defaults

- Embedding: `BAAI/bge-small-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Analyzer: `distilbert-base-uncased` (T4 pilot) / `deberta-v3-base` (A100)
- `k=5`, `α=0.6`, confidence threshold `0.55`
- Generator default for offline eval: extractive stub (swap to `hf` / `api` for LLM answers)

## Project layout

```
adaptive-temporal-rag/
  app/streamlit_app.py
  data/
  notebooks/
  scripts/
  src/
  artifacts/          # created after index/train/eval
  REPORT_SECTIONS.md  # text you can paste into the final report
```

## License / notes

Synthetic educational dataset for coursework. Not affiliated with cited news organizations; timelines are curated for temporal ranking experiments.
