# Adaptive Temporal RAG

Research code for *When and What to Retrieve*. The repository overview, frozen numbers, and the Vercel demo are in the [root README](../../README.md).

## Run

Commands below assume this directory (`codebase/adaptive-temporal-rag`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

`data/corpus.jsonl` (803 passages) and `data/queries.jsonl` (3,000 questions) are already here. `artifacts/analyzer.pt` and `artifacts/faiss.index` are not in git. If they are missing locally:

```powershell
python scripts/build_index.py
python scripts/train_analyzer.py --epochs 6
```

Step-by-step setup, Colab, and common errors: [HOW_TO_RUN.md](HOW_TO_RUN.md). Pipeline map: [ARCHITECTURE.md](ARCHITECTURE.md). Saved tables: [artifacts/final_results.md](artifacts/final_results.md).

## Layout

```
app/streamlit_app.py     local demo
src/                     policy, ranker, reader, pipelines
scripts/                 dataset, index, train, evaluate
data/                    corpus and question splits
configs/final.yaml       frozen Checkpoint B settings
artifacts/               local checkpoints and result tables
notebooks/               Colab and training notebooks
docs/                    diagrams and run notes
```

## Defaults

- Embeddings: `BAAI/bge-small-en-v1.5`, FAISS, then `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Analyzer: `distilbert-base-uncased`
- Retrieval depth `k = 5`, blend `α = 0.6`, confidence and sufficiency thresholds `0.55`
- Offline generator: extractive. API answers are optional via `.env.example`

## Learned router

The r4 learned router lives in [router_v2/](router_v2/). A presentable Colab notebook is [notebooks/train_learned_router.ipynb](notebooks/train_learned_router.ipynb). On the frozen eval it ties the rule-based Adaptive system (templated EM 0.948, held-out 0.762, TimeQA Contains 0.296). See [router_v2/README.md](router_v2/README.md) for the full table and how to retrain.
