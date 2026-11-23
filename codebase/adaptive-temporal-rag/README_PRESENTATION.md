# Adaptive Temporal RAG — Presentation package

Lean Colab package: **trained analyzer + FAISS index + real test metrics + interactive KB**.

## Quick start (Google Colab)

1. Runtime → **GPU (T4)**  
2. Upload / unzip **`atr_presentation.zip`** (or put folder on Drive)  
3. Open **`notebooks/01_colab_presentation.ipynb`**  
4. Run cells top → bottom  

**Default: no training.** In the notebook config cell set:

```python
RUN_TRAINING = False   # load analyzer.pt + FAISS (demo)
RUN_TRAINING = True    # rebuild dataset + index + train analyzer
```

## What’s inside

| Path | Purpose |
|------|---------|
| `artifacts/analyzer.pt` | Trained DistilBERT query analyzer |
| `artifacts/faiss.index` + `embeddings.npy` | Retrieval index |
| `artifacts/results_*.json/csv/png` | **Real** held-out test comparison (n=77) |
| `data/corpus.jsonl` | Base dated knowledge base |
| `notebooks/01_colab_presentation.ipynb` | Demo notebook |
| `src/` | Inference + loopable KB code |

## Notebook flow

0. Setup  
1. Load artifacts  
2. Show **real** B1/B2/B3/Adaptive comparison + charts  
3. KB snapshot + year histogram  
4. **Add documents in a loop** (paste / `.txt` / `.pdf` upload → FAISS update)  
5. Ask questions with Adaptive routing + evidence  
6. Same query on all four systems  
7. Download updated bundle  

## Honesty notes

- Comparison numbers come from `evaluate.py` on the **test split** (`results_summary.json`).  
- Base corpus is a **synthetic dated benchmark** (standard for this project).  
- Documents you add in §4 are **your real text** and are searched like any other passage.
