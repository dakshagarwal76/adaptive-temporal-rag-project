# Adaptive Temporal RAG — laptop transfer

Copied on 25 October 2026 from `C:\Users\Assessment\Documents\cursor`.

This PC has no `E:` drive. The external disk is the Seagate Backup Plus drive, letter **D:**. Everything is in this folder:

`D:\Adaptive-Temporal-RAG-Laptop-Transfer`

Treat this folder as private. `codebase\adaptive-temporal-rag\.env` was copied and may contain API keys.

## What is inside

| Folder | What it is |
|--------|------------|
| `codebase\adaptive-temporal-rag\` | Full project: source, notebooks, data, docs, trained `artifacts\analyzer.pt`, FAISS index, embeddings, eval results, `.env` |
| `docs-and-reports\` | Mid-term DOCX/PDFs, Arsh format baseline, workspace chat exports, USAR helpers |
| `cursor-chats\agent-transcripts\` | Cursor agent chats for this workspace (JSONL). They do not auto-load into Cursor on the new laptop; open them as text, or point a new chat at this folder |
| `archives\` | Zip snapshots: full project zip, Colab zip, Colab artifacts zip (`atr_colab_artifacts_20261008_0451.zip`), presentation zip |

Not copied on purpose: `DOC\_extract\py` (a downloaded Python installer dump, not project code). Recreate a venv on the new laptop; there was no `.venv` in the source tree.

## Open this on the other laptop

1. Copy this whole folder off the Seagate drive onto the new machine, for example `Documents\cursor\`.
2. In Cursor: File → Open Folder → `codebase\adaptive-temporal-rag`.
3. Python 3.10+:

```powershell
cd codebase\adaptive-temporal-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

You do **not** need to retrain if `artifacts\analyzer.pt` and `artifacts\faiss.index` are present. `analyzer.pt` is the saved fine-tuned DistilBERT weights. Training, when you do it, starts again from Hugging Face DistilBERT on labeled `data\queries.jsonl` and **overwrites** `analyzer.pt`. It does not train *on* the `.pt` file.

Colab path: `notebooks\00_colab_e2e.ipynb`. Read `HOW_TO_RUN.md`, `ARCHITECTURE.md`, `docs\COLAB_E2E_FLOW.md`, and `docs\diagrams\INFERENCE_NODES.md`.

## Cursor chats

| File | Chat |
|------|------|
| `cursor-chats\agent-transcripts\e2f1ee83-1845-4349-9f00-7418f8c02d10\` | This transfer chat, progress report, architecture walkthrough, Colab routing questions |
| `cursor-chats\agent-transcripts\a54ad015-322f-47b2-8e31-8677e42c757e\` | Earlier project chat (larger transcript) |
| `docs-and-reports\cursor_comparative_analysis_results.md` | Exported chat: baselines, Colab, adaptive window, sufficiency loop |
| `docs-and-reports\cursor_training_plan_for_dataset.md` | Exported chat: dataset / training plan |

## Project in one page

**Title:** When and What to Retrieve: Adaptive Temporal RAG (ARP-455, USAR).  
**People:** Shashank Kumar (08619051623), Daksh Agarwal (10819051623), supervisor Dr. Annu Priya.  
**Idea:** Before retrieval, a multitask analyzer predicts `needs_retrieval`, `temporal_class` (`current` / `historical` / `timeless`), and `confidence`. A policy then picks no-retrieval, recency, an adaptive historical date window, or semantic-only. Only Adaptive has an evidence-sufficiency repair loop.

**Four systems (same corpus, index, generator):**

- B1 LLM-only — never retrieve
- B2 Standard RAG — always semantic
- B3 Fixed Temporal RAG — always retrieve, fixed ±2 year window
- Adaptive — analyzer + adaptive window + sufficiency loop

**Stack:** BGE-small + FAISS + cross-encoder `ms-marco-MiniLM`, DistilBERT analyzer (DeBERTa planned for a larger run), generator extractive or Groq → OpenRouter → OpenCode.

**Frozen Checkpoint B bake-off** (`codebase/adaptive-temporal-rag/artifacts/final_results.md`, config `configs/final.yaml`). Shared reader, projection, and entity cut. Do not quote older 0.600 / 0.844 / 0.468 numbers.

| Dataset | Adaptive EM | Adaptive Contains | B3 EM | B3 Contains | Adaptive ret | Adaptive mean s |
|---------|-------------|-------------------|-------|-------------|--------------|-----------------|
| synthetic n=77 | 0.948 | 0.987 | 0.935 | 0.987 | 0.870 | 1.690 |
| held-out n=193 | 0.762 | 0.762 | 0.684 | 0.710 | 1.000 | 2.584 |
| TimeQA n=500 | 0.008 | 0.296 | 0.010 | 0.270 | 1.000 | 8.736 |

Adaptive ties B3 on templated containment, wins on the six-entity split, and wins TimeQA containment while losing one exact-match question. Full P@5 / R@5 / F1 / median latency: `artifacts/final_results.md`.

**Known training issue (say this honestly):** the Colab T4 pilot’s temporal accuracy stayed near chance (~0.33). Sanity queries often all came out `timeless`. `2+2` as `timeless` is correct (no time dimension). Skipping retrieval is a **different** field: `needs_retrieval == 0` **and** `is_trivial_query()` in `src\generate.py` (hardcoded phrases such as `2 + 2`). Identical ~75% confidence is expected when the confidence head is not in the training loss (`0.5 * ~0.5 + 0.5 * ~1.0 = 0.75`).

**Report:** `docs-and-reports\DOC\ARP455_Progress_Report_Adaptive_Temporal_RAG.docx` (Arsh Ahmad section layout, GGSIPU logo on page 1, each major section starts on a new page). Backup of the older draft is the `*_BACKUP.docx` next to it. Format sample: `Minor Project Progress Report - Arsh Ahmad and Garv Jaiswal.pdf`.

## Key source files

| File | Role |
|------|------|
| `src\query_analyzer.py` | Train/load analyzer, `predict()` |
| `src\adaptive_policy.py` | Policy router |
| `src\pipelines.py` | B1, B2, B3, Adaptive |
| `src\temporal_rank.py` | Adaptive date window vs B3 fixed ±2y |
| `src\evidence_sufficiency.py` | Sufficiency score and repairs |
| `src\generate.py` | Extractive / API failover, trivial-query list |
| `src\embed_index.py` | BGE + FAISS |
| `scripts\evaluate.py` | Four-system eval |
| `app\streamlit_app.py` | Local demo |
