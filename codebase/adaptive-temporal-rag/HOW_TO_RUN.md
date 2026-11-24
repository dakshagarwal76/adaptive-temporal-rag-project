# How to Run This Project (Beginner Guide)

This guide assumes you are on **Windows** and have never run this project before.

You will open a **terminal**, install Python packages, then open a **web demo** in your browser.

---

## What you need installed first

1. **Python 3.10 or newer**  
   - Check: open PowerShell and type:
     ```powershell
     python --version
     ```
   - If that fails, try:
     ```powershell
     py --version
     ```
   - If Python is missing, install it from [https://www.python.org/downloads/](https://www.python.org/downloads/)  
     During install, tick **“Add Python to PATH”**.

2. This project folder:
   ```text
   codebase\adaptive-temporal-rag
   ```

---

## Path A — Fastest: just open the demo (recommended first)

The question files and the 803-passage corpus are in `data/`.
A machine that already has `artifacts/analyzer.pt` and `artifacts/faiss.index` can start the demo after installing packages.
Those checkpoints are not in git. On a fresh clone, build them with Path B before opening Streamlit.

### Step 1 — Open PowerShell at the repository root, then enter the project

```powershell
cd codebase\adaptive-temporal-rag
```

### Step 2 — Create a virtual environment (one-time)

```powershell
python -m venv .venv
```

If `python` does not work:

```powershell
py -m venv .venv
```

### Step 3 — Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If Windows blocks activation, run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

You should see `(.venv)` at the start of your terminal line.

### Step 4 — Install packages (one-time, takes a few minutes)

```powershell
pip install -r requirements.txt
```

### Step 5 — Start the demo website

```powershell
streamlit run app/streamlit_app.py
```

By default the demo uses **extractive** answers (works offline from your retrieved docs).  
Groq (`api`) is optional. Current working model in `.env`:

```text
GROQ_MODEL=openai/gpt-oss-20b
```

If Groq fails, keep **Answer generator = extractive**.

If the demo was already running, stop it with `Ctrl+C` and start it again so code changes load.

### Re-run the automated E2E smoke test anytime

```powershell
python scripts/e2e_smoke_test.py
```

You want `FAIL=0` in the summary.

### Step 6 — Open it in your browser

- Streamlit usually opens automatically.
- If not, go to: [http://localhost:8501](http://localhost:8501)

### Step 7 — Try these questions

In the demo:

1. Choose system **`adaptive`** (left sidebar).
2. Ask:

   - `Who is the current Prime Minister of India?`
   - `Who was the President of the United States in 2018?`
   - `Define photosynthesis.`
   - `What is 2 + 2?`

3. Look at:
   - **Answer**
   - **Analyzer** (needs retrieval? current / historical / timeless?)
   - **Retrieved evidence** (dated documents)

4. In the sidebar, switch to **Comparative analysis** to see how Adaptive scores against B1 / B2 / B3 on the test set.

### Stop the demo

In the terminal, press:

```text
Ctrl + C
```

---

## Path B — Full run from scratch (dataset → train → evaluate → demo)

Use this when you want to rebuild everything yourself (good for project viva / Colab training story).

### Step 1 — Go to the folder and activate venv

```powershell
cd codebase\adaptive-temporal-rag
.\.venv\Scripts\Activate.ps1
```

If `.venv` does not exist yet, create it and install packages using Path A Steps 2–4 first.

### Step 2 — Build the dataset

```powershell
python scripts/build_dataset.py --target-queries 1050
```

Creates:

- `data/corpus.jsonl`
- `data/queries.jsonl`

### Step 3 — Build the search index

```powershell
python scripts/build_index.py
```

Creates FAISS files in `artifacts/` (this downloads the MiniLM model the first time).

### Step 4 — Train the query analyzer

**On your laptop (CPU):** start with fewer epochs (faster):

```powershell
python scripts/train_analyzer.py --epochs 3
```

**For the full ~2–3 hour Colab-style train:**

```powershell
python scripts/train_analyzer.py --epochs 6
```

Creates:

- `artifacts/analyzer.pt`

### Step 5 — Evaluate all 4 systems + comparative analysis

```powershell
python scripts/evaluate.py --split test --generator extractive
python scripts/plot_results.py
```

Creates:

- `artifacts/results.csv`
- `artifacts/results_summary.json`
- `artifacts/results_charts.png` (4-panel comparison figure)
- `artifacts/comparative_analysis.md` (tables for your report)
- `artifacts/comparative_analysis.json`

### Step 6 — Open the demo

```powershell
streamlit run app/streamlit_app.py
```

In the sidebar, open **Comparative analysis** to see scoreboards, deltas vs B1/B2/B3, and charts.

---

## Path C — Google Colab (T4 pilot ~2–3h, then optional A100 ~6h)

Use **one notebook** for dataset → index → train → eval → in-Colab demo → download.

### Stage A — T4 pilot (do this first, ~2–3 hours train)
1. Copy `.env.example` → `.env` and add API keys (Groq / OpenRouter / OpenCode).
2. Upload project zip / Drive folder to Colab.
3. **Runtime → GPU (T4)**
4. Open **`notebooks/00_colab_e2e.ipynb`**
5. Keep `RUN_STAGE = "t4_pilot"` (default):
   - Dataset: **`TARGET_QUERIES=3000`**
   - Train: **`AUGMENT_FACTOR=8`**, **`EPOCHS=12`**, **`BATCH_SIZE=8`**
6. Run all cells → demo → download artifacts.

Viva line: *“Pilot mid-size training on T4 to validate routing, adaptive windows, sufficiency loop, and metrics.”*

### Stage B — A100 large (later, ~6 hours train)
1. Same notebook, set `RUN_STAGE = "a100_large"`
2. Runtime → **A100**
3. Rebuild dataset + retrain (`TARGET_QUERIES=7000`, `AUGMENT_FACTOR=12`, `EPOCHS=15`, `BATCH_SIZE=32`)
4. Re-evaluate + download final artifacts.

Viva line: *“After the pilot, we scaled data/augmentation on A100 for stronger final results.”*

Local after download:
```powershell
cd codebase\adaptive-temporal-rag
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

---

## What each important folder means

| Folder / file | Meaning |
|---------------|---------|
| `app/streamlit_app.py` | The demo website |
| `data/` | Dataset (documents + questions) |
| `artifacts/` | Trained model + search index + results |
| `scripts/` | Commands you run in terminal |
| `notebooks/00_colab_e2e.ipynb` | **Preferred** one-shot Colab pipeline + demo + download |
| `notebooks/` | Optional older step-by-step Colab notebooks |
| `src/` | Project code (pipelines, ranking, analyzer) |
| `REPORT_SECTIONS.md` | Text you can paste into your college report |
| `HOW_TO_RUN.md` | This beginner guide |

---

## Common problems and fixes

### `python` is not recognized

Try:

```powershell
py --version
```

Then use `py` instead of `python` in all commands.

### `Activate.ps1` is blocked

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

### `streamlit` is not recognized

Make sure venv is activated (`(.venv)` visible), then:

```powershell
pip install streamlit
streamlit run app/streamlit_app.py
```

### Terminal shows many `No module named 'torchvision'` lines

Those are usually Streamlit file-watcher warnings, not a crashed app. Fix:

```powershell
pip install torchvision
streamlit run app/streamlit_app.py
```

### Demo says artifacts failed to load

You are missing index/model files. Run:

```powershell
python scripts/build_index.py
python scripts/train_analyzer.py --epochs 3
```

Then start Streamlit again.

### Package install is very slow / fails

Make sure internet works, then retry:

```powershell
pip install -r requirements.txt
```

First-time downloads of `torch`, `transformers`, and models can take several minutes.

### Wrong folder

Always run commands from:

```powershell
cd codebase\adaptive-temporal-rag
```

---

## Smallest “I just want to see it work” checklist

```powershell
cd codebase\adaptive-temporal-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If Streamlit says the index or analyzer failed to load, the checkpoints are not in this clone. Run Path B steps 3 and 4, then start Streamlit again.

Then open [http://localhost:8501](http://localhost:8501) and ask:

`Who is the current Prime Minister of India?`
