# Adaptive Temporal RAG

When and what to retrieve for time-sensitive questions.

A question about who holds an office now should not return a true sentence from another year. This project puts a temporal policy on one retrieval index: skip trivia, keep a date window around a past year, and prefer the latest passage for a current office. The same reader is used for every system, so the comparison is the retrieval decision, not a different generator.

Live demo: [adaptive-temporal-rag.vercel.app](https://adaptive-temporal-rag.vercel.app)

## What is in this repo

| Path | Contents |
|------|----------|
| `codebase/adaptive-temporal-rag/` | Research code, synthetic corpus, notebooks, and saved evaluation tables |
| `web/` | Next.js demo deployed on Vercel. Lexical retrieval, the temporal policy, and an extractive reader |

The frozen numbers are in `codebase/adaptive-temporal-rag/artifacts/final_results.md`.

## Systems

All four share the extractive reader and the answer projection.

| System | Retrieval |
|--------|-----------|
| Closed book | Never retrieve. Hardcoded trivia still answers `2 + 2` |
| Standard RAG | Always retrieve, rank by text overlap |
| Fixed window | Always retrieve. A year in the question keeps a ±2 year window; otherwise rank by recency |
| Adaptive | Skip trivia, widen a window around a past year until a passage hits, and prefer the latest passage for a current office |

## Frozen test, synthetic questions (n = 77)

Same Checkpoint B configuration for every row. Source: `artifacts/general_eval/final/synthetic/`.

| System | EM | Token F1 | Contains | P@5 | Temporal | Retrieved |
|--------|---:|---------:|---------:|-----:|---------:|----------:|
| Closed book | 0.130 | 0.167 | 0.130 | 0.000 | 0.247 | 0% |
| Standard RAG | 0.649 | 0.676 | 0.727 | 0.411 | 0.662 | 100% |
| Fixed ±2 year | 0.935 | 0.951 | 0.987 | 0.608 | 1.000 | 100% |
| Adaptive | 0.948 | 0.960 | 0.987 | 0.668 | 1.000 | 87% |

On the six-entity held-out split (n = 193), Adaptive exact match is 0.762 against 0.684 for the fixed window. On the TimeQA subset (n = 500), exact match stays near zero for every system; Adaptive containment is 0.296 against 0.270 for the fixed window. Full tables, latency, and ablations: `artifacts/final_results.md`.

## Learned router

Round 4 trains a DeBERTa router that decides whether to retrieve and which temporal class a question has. On the same frozen eval it ties the Adaptive rules: templated exact match 0.948, held-out exact match 0.762, TimeQA containment 0.296, with 0/0 discordant answers and McNemar *p* = 1. It does not beat the rules.

Colab training notebook: [notebooks/train_learned_router.ipynb](codebase/adaptive-temporal-rag/notebooks/train_learned_router.ipynb). Scripts, bundle metadata, and result JSON files: [router_v2/](codebase/adaptive-temporal-rag/router_v2/). Checkpoints and the 80k-row training splits are not in git.

## Data

The main file is generated from timelines of office holders, executives, sports winners, awards, and a few laws.

| File | Size |
|------|------|
| `data/corpus.jsonl` | 803 dated passages |
| `data/queries.jsonl` | 3,000 questions: 2,849 train, 74 validation, 77 test |
| `data/heldout/` | Same generator, 1,200 questions. Test is 193 questions that name six held-out entities |
| `data/general/timeqa/` | 5,931 chunks and 900 questions (400 train, 500 test), built from TimeQA |
| `data/router_large/` | Larger router-training split used for the experimental 100k run |

The query analyzer was trained on the 2,849 templated training questions with six augmentations (17,094 rows, 7 epochs). The retriever and the reader are rules plus off-the-shelf models. They were not trained on these questions.

## Run the research code

From `codebase/adaptive-temporal-rag`, with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dataset is in the repo. The DistilBERT checkpoint, FAISS index, and embedding matrix are not: each checkpoint is larger than GitHub’s 100 MB file limit. A local copy that still has `artifacts/analyzer.pt` and `artifacts/faiss.index` can open the Streamlit demo immediately. A fresh clone needs those files rebuilt:

```powershell
python scripts/build_index.py
python scripts/train_analyzer.py --epochs 6
```

Training overwrites `artifacts/analyzer.pt`. The longer guide is `HOW_TO_RUN.md`. Frozen evaluation uses `configs/final.yaml` and `python scripts/evaluate.py`.

Optional API answers read `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `OPENCODE_API_KEY`. Copy `.env.example` to `.env`. The committed `.env.example` has placeholders only. Offline evaluation uses the extractive reader and does not need a key.

## Run the web demo locally

The Vercel app does not load the GPU models. It answers from the 803-passage corpus with BM25, the same temporal rules, and the extractive reader.

```powershell
cd web
npm install
npm run check
npm run dev
```

`npm run check` must print the four demo answers: Narendra Modi, Donald Trump, the photosynthesis definition, and 4 with retrieval skipped.

## Not in git

`*.pt`, `*.npy`, `*.index`, `*.safetensors`, zip archives, virtualenvs, and `web/node_modules`.
