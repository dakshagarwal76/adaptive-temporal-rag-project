# Comparative analysis results
_Exported on 10/7/2026 at 17:16:15 GMT+5:30 from Cursor (3.22.12)_

---

**User**

i want to also show the results, as to how well am i doing as compared to the current ones existing. a compartivie analysis, so could you help me out on that as well.

---

**Cursor**

I'll look at your Adaptive Temporal RAG project to see what results you already have and how we can add a comparative analysis against existing approaches.

You already have eval numbers vs B1/B2/B3 — I'll wire a proper comparative analysis into the demo and strengthen the charts/report so the comparison is easy to show.

I'll add a shared comparison module, richer charts, a Streamlit analysis tab, and a report section you can paste into your write-up.

I'll drop the `tabulate` dependency and write a small markdown table helper instead.

Comparative analysis is wired in against B1 / B2 / B3 using your existing test results.

### Where to see it

1. **Streamlit** — sidebar → **Comparative analysis**  
   Scoreboard, deltas, by-class charts, efficiency frontier.

```powershell
cd D:\cursor\adaptive-temporal-rag
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

2. **Charts + report files** (already generated):
   - `artifacts/results_charts.png`
   - `artifacts/comparative_analysis.md` ← paste into your report
   - `REPORT_SECTIONS.md` §10 — expanded comparison write-up

Regenerate anytime with:
```powershell
python scripts/plot_results.py
```

### Headline numbers (75 test queries)

| System | Contains | Temp Acc | Retrieval |
|--------|----------|----------|-----------|
| B1 LLM-only | 0.027 | 0.307 | 0% |
| B2 Standard RAG | 0.427 | 0.573 | 100% |
| B3 Fixed Temporal | 0.773 | 0.867 | 100% |
| **Adaptive (yours)** | **0.600** | **0.840** | **68%** |

**Story for viva/report:** vs Standard RAG you gain **+17pp Contains** and **+27pp Temporal Acc** while retrieving **~32% less**. On current/historical questions you **match Fixed Temporal**; the overall Contains gap vs B3 is mostly timeless items where Adaptive correctly skips retrieval (extractive eval then has nothing to quote).

---

**User**

instead of showing it like on the streamlit have a colab nb one e2e wherein i run all the things and then download whatever artifcats or whatever is needed and then i can store it here and run the same thin locally, also on the colab i want it to be trained on something which isnt a huge dataset and should take 2 hours not more than that. Also in the colab once the training is done i want it to have something wherein i put the query and it is able to show like we are shoing on streamlit but inside colab itself.

---

**Cursor**

I'll check your existing Colab notebooks and training setup, then build one end-to-end notebook: train (~2h), evaluate, interactive query demo in-notebook, and artifact download for local reuse.

I'll create one Colab E2E notebook with compact ~2h training, an in-notebook query demo, and artifact download for local use.

PowerShell mangled the inline script — I'll write the notebook generator as a Python file and run that instead.

I'll harden the Colab setup so it finds the project on Drive more reliably before asking for a zip upload.

You now have one Colab end-to-end notebook instead of the Streamlit-on-Colab path.

### Use this notebook
**`notebooks/00_colab_e2e.ipynb`**

It runs everything in order:
1. Setup / install (finds project on Drive or lets you upload a zip)
2. Compact dataset (`TARGET_QUERIES=700`)
3. FAISS index
4. Train DistilBERT (`EPOCHS=5`, `FREEZE_LAYERS=2`) — aimed at **≤ ~2 hours** on T4
5. Eval + comparative charts
6. **In-Colab query UI** (system dropdown, question box, answer / analyzer / evidence — same idea as Streamlit)
7. **Download zip** of `artifacts/` + `data/`

### Colab steps
1. Runtime → GPU (T4)
2. Open `00_colab_e2e.ipynb`
3. Runtime → Run all
4. In section 5, type queries in the widget
5. In section 6, download the zip

### On your PC after download
Overwrite local folders with the zip contents, then:

```powershell
cd D:\cursor\adaptive-temporal-rag
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

No retrain needed if `analyzer.pt` and the FAISS files are in `artifacts/`.

If training still feels long, in the config cell set `FREEZE_LAYERS = 4` or `EPOCHS = 4`. Details are in `HOW_TO_RUN.md` Path C.

---

**User**

what exactly, have you made that compartive analysis for and what does it comtain exactly, is that even real data?

---

**Cursor**

Checking where those comparison numbers come from so I can answer precisely.

**What it’s for:** to show how *your* Adaptive system scores against the three baselines built into this project (B1 LLM-only, B2 Standard RAG, B3 Fixed Temporal), on the same test questions — for the report/viva/demo, not for comparing to published papers or public leaderboards.

**What it contains (exactly):**
- **Overall table** — Contains, EM, Token F1, Temporal Acc, P@5, R@5, retrieval rate, latency  
- **By temporal class** — current / historical / timeless  
- **Deltas** — Adaptive minus each baseline  
- **Charts** — same metrics + efficiency (quality vs how often it retrieves)  
- **Highlights** — short auto-written takeaways  

Source files: `artifacts/results.csv` → rolled into `comparative_analysis.md` / `.json` / `results_charts.png`, and the Streamlit **Comparative analysis** page.

**Is it “real” data?**  
**Real measured runs on this project’s own dataset — not fake numbers, and also not a real-world news/web benchmark.**

| Piece | What it actually is |
|--------|----------------------|
| Numbers | Real outputs of `scripts/evaluate.py` on 75 held-out **test** queries (300 rows = 75 × 4 systems) |
| Corpus / questions | **Synthetic educational** set (~800 dated passages, labeled queries) built for coursework — curated timelines, not FreshQA / Natural Questions |
| Answers in that eval | Mostly **extractive** (pull from retrieved docs), not a big LLM |
| “Real world” claims | Don’t claim SOTA on public datasets; claim “on our time-aware test split, Adaptive vs B1/B2/B3 …” |

So: the comparative analysis is a **honest project-internal bake-off**. The metrics are computed for real from your pipelines; the underlying corpus is synthetic for the assignment, which the README already notes.

---

**User**

in the colab what setups and all have you used? which odels and all how can i improve them?

---

**Cursor**

Here’s exactly what the Colab E2E notebook (`00_colab_e2e.ipynb`) uses, and how to improve it.

### Setup in Colab
| Piece | Default |
|--------|---------|
| Runtime | **GPU T4** |
| Dataset | **`TARGET_QUERIES = 700`** (compact, not the full ~1050) |
| Train | **`EPOCHS = 5`**, **`BATCH_SIZE = 16`**, **`LR = 2e-5`**, **`FREEZE_LAYERS = 2`** |
| Retrieval | **`TOP_K = 5`**, **`ALPHA = 0.6`** (semantic vs temporal mix), **`CONF_THRESHOLD = 0.55`** |
| Eval answers | **`GENERATOR = "extractive"`** (pulls text from docs; no big LLM) |
| Index | FAISS (`IndexFlatIP`) over embeddings |

### Models
| Role | Model |
|------|--------|
| Embeddings / search | **`sentence-transformers/all-MiniLM-L6-v2`** |
| Query analyzer (trained) | **`distilbert-base-uncased`** multitask (need retrieval? + current/historical/timeless + confidence) |
| Answer generator (eval/demo) | **Extractive stub** — not Phi-3 / Groq unless you change it |
| Optional in code (not Colab default) | HF: `microsoft/Phi-3-mini-4k-instruct`; API: Groq via `.env` |

So Colab is intentionally **small + fast**: MiniLM + DistilBERT + extractive answers.

---

### How to improve (best bang for effort)

**1. Fairer / better answers (biggest for “Contains”)**  
In the config cell:
```python
GENERATOR = "hf"   # needs GPU RAM; uses Phi-3 in src/generate.py
```
Or locally use Groq (`api`) with a key. This especially helps **timeless / no-retrieval** questions where extractive looks weak.

**2. Stronger analyzer (quality vs time)**  
```python
TARGET_QUERIES = 900   # or 1050
EPOCHS = 6
FREEZE_LAYERS = 0      # train full DistilBERT
```
Stay under ~2h if possible; if it overruns, use `FREEZE_LAYERS = 2` again.

**3. Better retrieval**  
- Raise `TOP_K` to `8`–`10`  
- Tune `ALPHA` (e.g. `0.5` = more temporal, `0.7` = more semantic)  
- Swap embedder to something stronger, e.g. `sentence-transformers/all-mpnet-base-v2` (slower, larger index) in `src/embed_index.py` / `build_index_from_corpus(..., model_name=...)`

**4. Better ranking**  
Current temporal score is heuristic. Improvements: learn a temporal scorer, tighter date windows, or entity-aware filtering — that’s real “research upgrade” work.

**5. Better data**  
Synthetic corpus is fine for coursework. For stronger claims: more entities, fresher “current” snapshots, or a public time-sensitive set (e.g. FreshQA-style) — then re-run eval.

**Practical order:** `GENERATOR = "hf"` or Groq → more train data / `FREEZE_LAYERS = 0` → tune `ALPHA`/`TOP_K` → stronger embedder.

---

**User**

yes ill use groq/openrouter/opencode if one of them hits the limit switch to the other one and then ill train in on a100 for 2 hours maximum so give based on that and improve it to give amazing results a instead of fixed date window make the window adaptive too,

---

**Cursor**

I'll wire multi-provider LLM fallback (Groq → OpenRouter → OpenCode), tune A100 training for a 2h budget, and replace the fixed historical date window with an adaptive one.

Implementing provider failover, adaptive historical windows, and A100-oriented Colab/training defaults.

Cleaning up the adaptive-window meta plumbing in the pipeline — the last edit was messy.

Here’s what changed for your A100 + multi-API + adaptive-window setup.

### 1. API failover: Groq → OpenRouter → OpenCode
If one hits a rate limit / outage / block, the next is tried, then extractive.

Copy `.env.example` → `.env` and fill what you have:

```env
GENERATOR_BACKEND=api
LLM_PROVIDER_ORDER=groq,openrouter,opencode
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
OPENCODE_API_KEY=...
OPENCODE_BASE_URL=https://opencode.ai/zen/v1   # or your Console gateway URL
```

### 2. Adaptive historical date window (not fixed ±2y)
Adaptive only (B3 stays fixed ±2y as baseline):

| Query cue | Starts at | Expands up to |
|-----------|-----------|----------------|
| Exact date `2022-01-20` | ±0.35y | 1.5y |
| Year `in 2018` | ±1y | 4y |
| Decade `2010s` | ±5y | 8y |
| Vague historical | ±3y | 8y |

Window grows until enough docs land inside; UI shows `adaptive_window` meta.

### 3. Colab A100 ≤ ~2h preset (`00_colab_e2e.ipynb`)
| Knob | Value |
|------|--------|
| Queries | 1050 |
| Epochs | 8 |
| Batch | 64 |
| Freeze | 0 |
| Max length | 96 |
| AMP | on |
| Generator | `api` failover |
| α / top_k | 0.55 / 8 |

Paste keys in the notebook API cell, Runtime → **A100**, Run all, download artifacts.

### What you should do
1. Put keys in `.env` (and/or the Colab cell).  
2. Run `notebooks/00_colab_e2e.ipynb` on A100.  
3. Download zip → overwrite local `artifacts/` + `data/`.  
4. Locally: `GENERATOR_BACKEND=api` then `streamlit run app/streamlit_app.py`.

---

**User**

also add in a vidence suffiecncy feedback loop as well to enahnce the result.

---

**Cursor**

I'll add an evidence-sufficiency feedback loop to the Adaptive pipeline: check retrieved evidence, and if it's weak, expand retrieval and retry.

Fixing a parameter name typo that would break the pipeline.

Added an **evidence sufficiency feedback loop** to Adaptive only.

### How it works
After each retrieve/rank step, a judge scores evidence on:
- query-term coverage  
- top/mean retrieval score  
- temporal fit  
- (optional) weak answer text  

If score &lt; 0.55, it applies repairs and retries (up to **2** rounds), then one **post-answer** retry if the answer still looks empty/weak:

| Repair | Effect |
|--------|--------|
| `widen_window` | Broader historical date window |
| `expand_candidates` | Pull more FAISS candidates |
| `raise_topk` | Keep more final docs |
| `relax_to_semantic` | Drop strict temporal filter |
| `boost_recency` | Prefer newer docs for “current” |

### Where you’ll see it
- Streamlit analyzer panel → `evidence_sufficiency` + expander **Evidence sufficiency feedback loop**
- Colab demo → same fields under the answer
- Meta on results: `sufficiency_loop` lists each round + repairs

New module: `src/evidence_sufficiency.py` · wired in `AdaptiveTemporalRAGPipeline`.

---

**User**

can you smoke test if everything is working fine and check if the feeback loop is also working with proper routing between all the three api keys, in the e2e ipynb i have.

---

**Cursor**

I'll run a full smoke test: core pipeline, sufficiency feedback loop, and API failover routing (Groq → OpenRouter → OpenCode) against your E2E setup.

Only Groq is in `.env` (OpenRouter/OpenCode unset). I'll extend the smoke test for the sufficiency loop and failover routing, then run it — keys won't be printed.

Updating the smoke test for the new failover generator and sufficiency loop, then running it.

**Smoke test: PASS=72 · FAIL=0 · WARN=2**

Ran `python scripts/e2e_smoke_test.py` (covers the same pieces as `00_colab_e2e.ipynb`).

### What’s working
| Area | Result |
|------|--------|
| Dataset / index / analyzer | Pass |
| B1 / B2 / B3 / Adaptive answers | Pass |
| Adaptive date window | Pass (`year` + `exact_date`) |
| Evidence sufficiency loop | Pass — **3 rounds**, repairs applied (`widen_window` ×2) on a hard query |
| Colab E2E notebook cells | Pass (keys, train, sufficiency UI, download) |
| API failover logic | Pass — tried `fake_dead` then **`groq`**, got live answer |

Failover proof from the run:
```text
failover_tries_first_provider — ['fake_dead', 'groq']
failover_reached_live_provider — api:groq:openai/gpt-oss-20b
```

### Warnings (keys)
Only **Groq** is set in `.env`.  
**OpenRouter** and **OpenCode** are missing, so full 3-provider live routing couldn’t be exercised end-to-end.

Add to `.env` (and paste in the Colab API cell):
```env
OPENROUTER_API_KEY=...
OPENCODE_API_KEY=...
OPENCODE_BASE_URL=https://opencode.ai/zen/v1
```
Then re-run:
```powershell
python scripts/e2e_smoke_test.py
```
You should see `provider_chain_loaded — groq -> openrouter -> opencode` with no those two WARNs.

---

**User**

actually ill train it on t4 give me dataset which will take me 2 hours on t4 gpu, i will then tell that ok this was to see how the results were, and now we will being training for large dataset to get good results, probably 6 hours on a100 gpu, right now we will train for about 2-3 hours on t4 gpu

---

**Cursor**

I'll size a T4 “pilot” dataset/training config for ~2–3 hours, and document the later A100 ~6h large run in the Colab E2E notebook.

I'll lift the 1200-query cap, add train-time augmentation so a T4 run can land near 2–3 hours, and retarget the Colab notebook as a T4 pilot with an A100 large-run follow-up.

Configured for a **T4 pilot first**, then an optional **A100 large** run later.

### What to use now (T4, ~2–3h)
In `notebooks/00_colab_e2e.ipynb` keep:

```python
RUN_STAGE = "t4_pilot"
```

| Setting | Value |
|---------|--------|
| Dataset | **3000 queries** |
| Train augment | **×8** (~22.8k train rows) |
| Epochs | **12** |
| Batch | **8** |
| Freeze | **0** |

Runtime → **GPU (T4)** → Run all.

**Viva line:** *“We trained a mid-size pilot on T4 for ~2–3 hours to validate the pipeline, adaptive window, sufficiency loop, and metrics before scaling.”*

### Later (A100, ~6h)
```python
RUN_STAGE = "a100_large"
```

| Setting | Value |
|---------|--------|
| Dataset | **7000 queries** |
| Augment | **×12** |
| Epochs | **15** |
| Batch | **32** |

**Viva line:** *“After the pilot, we scaled data/augmentation on A100 for ~6 hours for final quality.”*

### Tuning if time is off
- Finishes **&lt; 90 min** on T4 → raise `AUGMENT_FACTOR` to `10` or `EPOCHS` to `14`
- Goes **&gt; 3.5h** → `AUGMENT_FACTOR=6` or `EPOCHS=10`

---

**User**

whats augment factor?'

---

**Cursor**

**`AUGMENT_FACTOR`** multiplies your **training queries** with light paraphrases before DistilBERT trains.

Example: you have **2,850** train questions and `AUGMENT_FACTOR = 8`:

- Each question is kept once, then rewritten ~7 more times with prefixes like “Please answer: …”, “Tell me: …”, etc.
- Same labels (current/historical/timeless, needs retrieval)
- Effective train size ≈ **2,850 × 8 ≈ 22,800** examples

So it does **not** add new real facts — it stretches the same labeled set so the T4 run takes longer and sees more wording variety.

| Value | Effect |
|------|--------|
| `1` | No extra copies (fastest) |
| `8` | T4 pilot (~2–3h) |
| `12` | A100 large (~6h) |

