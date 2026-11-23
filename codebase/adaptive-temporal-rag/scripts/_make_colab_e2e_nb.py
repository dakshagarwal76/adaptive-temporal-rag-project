"""Generate notebooks/00_colab_e2e.ipynb (run once; not part of user workflow)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "00_colab_e2e.ipynb"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": [],
}


def md(src: str) -> None:
    nb["cells"].append(
        {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
    )


def code(src: str) -> None:
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }
    )


md(
    """# Adaptive Temporal RAG — Colab End-to-End

**One notebook** for the full pipeline. Default preset = **T4 pilot (~2–3h train)** to inspect results first.
Later you can switch to **A100 large (~6h)** for the final high-quality run.

1. Setup project + install deps + API keys (Groq → OpenRouter → OpenCode failover)
2. Build **pilot dataset** (~3000 queries)
3. Build FAISS index (**BGE-small**)
4. Train DistilBERT analyzer (**T4 target ~2–3 hours**)
5. Evaluate B1 / B2 / B3 / Adaptive + comparative charts
6. Interactive query demo
7. Download artifacts zip
8. **Full inference architecture** — step-by-step trace (B1–B4)
9. Use bundle on your PC (Streamlit)

After training: **§4** → **§5** → **§8** (trace JSON) → **§6** (download zip).

### Before you start
- Runtime → **Change runtime type** → **GPU (T4)** → Save
- Put this project on Colab (Drive / zip), then open **this** notebook
"""
)

md("## 0 — Setup project root + install packages")

code(
    r'''import os, sys, time, shutil, zipfile
from pathlib import Path

IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

print("IN_COLAB =", IN_COLAB)

def find_root(start: Path):
    for p in [start, *start.parents]:
        if (p / "src" / "pipelines.py").exists() and (p / "scripts" / "evaluate.py").exists():
            return p
    return None

ROOT = find_root(Path.cwd())

# Also probe common Colab locations before asking the user to upload.
if ROOT is None:
    for cand in [
        Path("/content/adaptive-temporal-rag"),
        Path("/content/drive/MyDrive/adaptive-temporal-rag"),
        Path("/content/drive/MyDrive/cursor/adaptive-temporal-rag"),
    ]:
        ROOT = find_root(cand) if cand.exists() else None
        if ROOT is not None:
            break

if ROOT is None and IN_COLAB:
    from google.colab import files, drive

    # Try Drive mount + search (quiet fail if user skips)
    DRIVE_PROJECT = "/content/drive/MyDrive/adaptive-temporal-rag"  # edit if your path differs
    try:
        drive.mount("/content/drive", force_remount=False)
        for cand in [
            Path(DRIVE_PROJECT),
            Path("/content/drive/MyDrive/adaptive-temporal-rag"),
            Path("/content/drive/MyDrive/cursor/adaptive-temporal-rag"),
        ]:
            hit = find_root(cand) if cand.exists() else None
            if hit is not None:
                ROOT = hit
                break
        if ROOT is None:
            # shallow search under MyDrive for pipelines.py
            mydrive = Path("/content/drive/MyDrive")
            if mydrive.exists():
                for p in mydrive.rglob("pipelines.py"):
                    if p.parent.name == "src":
                        ROOT = find_root(p.parent.parent)
                        if ROOT is not None:
                            break
    except Exception as exc:
        print("Drive mount/search skipped:", exc)

if ROOT is None and IN_COLAB:
    from google.colab import files
    print("Project still not found. Upload a ZIP of the adaptive-temporal-rag folder.")
    uploaded = files.upload()
    assert uploaded, "No zip uploaded"
    zip_name = next(iter(uploaded))
    extract_dir = Path("/content/atr_project")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_name) as zf:
        zf.extractall(extract_dir)
    ROOT = find_root(extract_dir)
    if ROOT is None:
        kids = [p for p in extract_dir.iterdir() if p.is_dir()]
        ROOT = find_root(kids[0]) if kids else None
    assert ROOT is not None, "Could not find project root inside the zip"

if ROOT is None:
    raise FileNotFoundError(
        "Could not find adaptive-temporal-rag. Open this notebook from the project folder, "
        "or upload the project zip in Colab."
    )

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("ROOT =", ROOT)

%pip install -q pandas scikit-learn matplotlib plotly sentence-transformers faiss-cpu accelerate tqdm python-dateutil requests ipywidgets pypdf
%pip install -q "transformers>=4.38"

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU — training will be much slower. Enable T4 runtime.")
'''
)

md(
    """## Config — training stage

### Stage A (default): T4 pilot — see results first (~2–3h train)
Use this now. Story for viva/report: *“We first trained a mid-size pilot on T4 to validate the pipeline and metrics.”*

| Knob | T4 pilot | Meaning |
|------|----------|---------|
| `RUN_STAGE` | `"t4_pilot"` | Active preset |
| `TARGET_QUERIES` | **3000** | Mid-size labeled set + paraphrases |
| `AUGMENT_FACTOR` | **8** | Extra light train paraphrases (longer GPU run) |
| `EPOCHS` | **12** | DistilBERT multitask epochs |
| `BATCH_SIZE` | **8** | Smaller batch → more steps on T4 |
| `FREEZE_LAYERS` | **0** | Full DistilBERT fine-tune |
| `MAX_LENGTH` | **96** | Query tokenizer length |

### Stage B (later): A100 large — best results (~6h train)
After the pilot, set `RUN_STAGE = "a100_large"` and re-run dataset + train for the final model.

| Knob | A100 large |
|------|------------|
| `TARGET_QUERIES` | **7000** |
| `AUGMENT_FACTOR` | **12** |
| `EPOCHS` | **15** |
| `BATCH_SIZE` | **32** |

If T4 pilot finishes **under ~90 min**, raise `AUGMENT_FACTOR` to `10` or `EPOCHS` to `14`.  
If it **overruns ~3.5h**, set `AUGMENT_FACTOR = 6` or `EPOCHS = 10`.
"""
)

code(
    r'''# ===== master switches =====
# False = skip dataset/index/train; load existing artifacts/
# True  = rebuild dataset + FAISS + train analyzer
RUN_TRAINING = True

# "t4_pilot"  -> ~2-3h on Colab T4  (use this first)
# "a100_large" -> ~6h on Colab A100 (final quality run later)
RUN_STAGE = "t4_pilot"

if RUN_STAGE == "a100_large":
    TARGET_QUERIES = 7000
    AUGMENT_FACTOR = 12
    EPOCHS = 15
    BATCH_SIZE = 32
    FREEZE_LAYERS = 0
    stage_note = "A100 large (~6h) — final quality training"
else:
    # T4 pilot defaults
    TARGET_QUERIES = 3000
    AUGMENT_FACTOR = 8
    EPOCHS = 12
    BATCH_SIZE = 8
    FREEZE_LAYERS = 0
    stage_note = "T4 pilot (~2-3h) — validate pipeline + metrics first"

LR = 2e-5
MAX_LENGTH = 96
GENERATOR = "api"        # failover; falls back to extractive if no keys
TOP_K = 8
ALPHA = 0.55
CONF_THRESHOLD = 0.50
CANDIDATE_K = 80

ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
ARTIFACTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

print("STAGE:", RUN_STAGE, "-", stage_note)
print("Config:", dict(
    TARGET_QUERIES=TARGET_QUERIES,
    AUGMENT_FACTOR=AUGMENT_FACTOR,
    EPOCHS=EPOCHS,
    BATCH_SIZE=BATCH_SIZE,
    FREEZE_LAYERS=FREEZE_LAYERS,
    MAX_LENGTH=MAX_LENGTH,
    GENERATOR=GENERATOR,
))
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU — enable T4 runtime before training.")

# Rough step budget (helps justify the 2-3h / 6h story)
approx_train = int(TARGET_QUERIES * 0.85 * AUGMENT_FACTOR)  # most paraphrases land in train
steps_per_epoch = max(1, approx_train // BATCH_SIZE)
total_steps = steps_per_epoch * EPOCHS
print(f"Approx train rows~{approx_train} | steps/epoch~{steps_per_epoch} | total_steps~{total_steps}")
'''
)

md(
    """## API keys (Groq → OpenRouter → OpenCode)

Paste keys below (runtime only). On rate-limit / 429 / outage the next provider is tried automatically.
"""
)

code(
    r'''import os
from src.generate import load_provider_chain, groq_healthcheck

# Fill any you have — missing ones are skipped
os.environ["LLM_PROVIDER_ORDER"] = "groq,openrouter,opencode"

# --- Groq ---
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")  # paste key between quotes if unset
os.environ["GROQ_MODEL"] = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# --- OpenRouter ---
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENROUTER_MODEL"] = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# --- OpenCode (Console / Zen OpenAI-compatible base URL) ---
os.environ["OPENCODE_API_KEY"] = os.environ.get("OPENCODE_API_KEY", "")
os.environ["OPENCODE_MODEL"] = os.environ.get("OPENCODE_MODEL", "opencode/gpt-5-nano")
os.environ["OPENCODE_BASE_URL"] = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")

# Quick edit helpers (uncomment + paste):
# os.environ["GROQ_API_KEY"] = "gsk_..."
# os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
# os.environ["OPENCODE_API_KEY"] = "..."

chain = load_provider_chain()
print("Provider chain:", " -> ".join(p.name for p in chain) if chain else "(none — extractive fallback)")
if chain:
    ok, msg = groq_healthcheck()
    print("Healthcheck:", "OK" if ok else "FAIL", msg)
else:
    print("No API keys set — GENERATOR will fall back to extractive answers.")
    GENERATOR = "extractive"
'''
)

md("## 1 — Build dataset (pilot mid-size / later large)")

code(
    r'''from collections import Counter
from src.data_schema import load_corpus, load_queries

print(f"Building dataset with TARGET_QUERIES={TARGET_QUERIES} ({RUN_STAGE})")
!python {ROOT}/scripts/build_dataset.py --target-queries {TARGET_QUERIES} --out-dir {DATA}

docs = load_corpus(DATA / "corpus.jsonl")
queries = load_queries(DATA / "queries.jsonl")
print(f"Corpus docs: {len(docs)}")
print(f"Queries: {len(queries)}")
print("split:", Counter(q.split for q in queries))
print("temporal:", Counter(q.temporal_class for q in queries))
print("needs_retrieval:", Counter(q.needs_retrieval for q in queries))
'''
)

md("## 2 — Build BGE-small + FAISS index")

code(
    r'''from src.embed_index import build_index_from_corpus

t0 = time.time()
index = build_index_from_corpus(DATA / "corpus.jsonl", ARTIFACTS)
print(f"Indexed {len(index.docs)} docs in {time.time()-t0:.1f}s")
print("Wrote:", ARTIFACTS / "faiss.index", ARTIFACTS / "embeddings.npy")
'''
)

md("## 3 — Train query analyzer (T4 pilot ~2–3h)")

code(
    r'''from src.data_schema import load_queries
from src.query_analyzer import train_analyzer, QueryAnalyzerService

queries = load_queries(DATA / "queries.jsonl")
train = [q for q in queries if q.split == "train"]
val = [q for q in queries if q.split == "val"]
print(
    f"STAGE={RUN_STAGE}  raw_train={len(train)}  val={len(val)}  "
    f"augment={AUGMENT_FACTOR}  epochs={EPOCHS}  batch={BATCH_SIZE}"
)

t0 = time.time()
result = train_analyzer(
    train_texts=[q.query for q in train],
    train_needs=[q.needs_retrieval for q in train],
    train_temporal=[q.temporal_class for q in train],
    val_texts=[q.query for q in val],
    val_needs=[q.needs_retrieval for q in val],
    val_temporal=[q.temporal_class for q in val],
    artifacts_dir=ARTIFACTS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    freeze_layers=FREEZE_LAYERS,
    max_length=MAX_LENGTH,
    augment_factor=AUGMENT_FACTOR,
)
elapsed_min = (time.time() - t0) / 60
print(f"Best val score: {result['best_val_score']:.4f}")
print(f"Train examples used: {result.get('train_examples')}")
print(f"Training wall time: {elapsed_min:.1f} minutes ({elapsed_min/60:.2f} hours)")
if RUN_STAGE == "t4_pilot":
    if elapsed_min < 90:
        print("Finished early for a 2-3h target — raise AUGMENT_FACTOR to 10 or EPOCHS to 14 and re-train.")
    elif elapsed_min > 210:
        print("Over ~3.5h — next pilot use AUGMENT_FACTOR=6 or EPOCHS=10.")
    else:
        print("In the intended ~2-3h T4 pilot band.")
else:
    if elapsed_min < 240:
        print("A100 large finished under 4h — optional: raise AUGMENT_FACTOR/EPOCHS for fuller 6h budget.")
    elif elapsed_min > 420:
        print("Over ~7h — lower AUGMENT_FACTOR or EPOCHS slightly next time.")

svc = QueryAnalyzerService.from_pretrained_bundle(ARTIFACTS)
for q in [
    "Who is the current Prime Minister of India?",
    "Who was the President of the United States in 2018?",
    "Define photosynthesis.",
    "What is 2 + 2?",
]:
    out = svc.predict(q)
    print(f"[{out.temporal_class} need={out.needs_retrieval} conf={out.confidence:.2f}] {q}")
'''
)

md(
    """## 3½ — Inference architecture (preview before eval)

| System | Inference path |
|--------|----------------|
| **B1** | Query → generator only |
| **B2** | Query → BGE+FAISS → cross-encoder → generator |
| **B3** | B2 → fixed temporal window / recency → generator |
| **Adaptive** | Query → **analyzer** → policy → FAISS → CE → temporal rank → **sufficiency loop** → generator |

PNG diagrams: `docs/diagrams/`. Node guide: `docs/diagrams/INFERENCE_NODES.md`.  
After training, run **§8** (after §7 download or before — trace writes into `artifacts/`).
"""
)

code(
    r'''from IPython.display import display, Markdown, Image
from pathlib import Path

diag = ROOT / "docs" / "diagrams"
for name, caption in [
    ("01_adaptive_inference_full.png", "### Adaptive inference (proposed system)"),
    ("02_policy_router.png", "### Policy router"),
    ("03_colab_four_systems_eval.png", "### §5 eval — four systems on test split"),
]:
    path = diag / name
    if not path.exists():
        print("MISSING", path, "— use the latest adaptive-temporal-rag-colab.zip")
        continue
    display(Markdown(caption))
    display(Image(filename=str(path)))
'''
)

md("## 4 — Evaluate all systems + comparative analysis")

code(
    r'''!python {ROOT}/scripts/evaluate.py --split test --generator {GENERATOR} --out {ARTIFACTS}/results.csv
!python {ROOT}/scripts/plot_results.py --results {ARTIFACTS}/results.csv --out {ARTIFACTS}/results_charts.png --report {ARTIFACTS}/comparative_analysis.md --json {ARTIFACTS}/comparative_analysis.json

from IPython.display import display, Markdown, Image
import json

summary = json.loads((ARTIFACTS / "results_summary.json").read_text(encoding="utf-8"))
rows = []
for sys_name, m in summary.items():
    rows.append(
        "| {} | {:.3f} | {:.3f} | {:.2f} | {:.3f} |".format(
            sys_name, m["contains"], m["temporal_acc"], m["retrieval_rate"], m["latency_s"]
        )
    )
display(Markdown(
    "### Test summary\n\n"
    "| System | Contains | Temp Acc | Ret Rate | Latency |\n"
    "|--------|----------|----------|----------|---------|\n"
    + "\n".join(rows)
))
display(Image(filename=str(ARTIFACTS / "results_charts.png")))
display(Markdown((ARTIFACTS / "comparative_analysis.md").read_text(encoding="utf-8")))
'''
)

md(
    """## 5 — Interactive demo (inside Colab)

Same idea as Streamlit: pick a system, type a query, see **answer**, **analyzer**, and **retrieved evidence**.
"""
)

code(
    r'''import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from src.embed_index import EmbeddingIndex
from src.generate import Generator
from src.pipelines import (
    AdaptiveTemporalRAGPipeline,
    FixedTemporalRAGPipeline,
    LLMOnlyPipeline,
    StandardRAGPipeline,
)
from src.query_analyzer import load_query_analyzer

_index = EmbeddingIndex()
_index.load(ARTIFACTS)
_analyzer = load_query_analyzer(ARTIFACTS)
_analyzer_name = "trained neural query analyzer"
_generator = Generator(backend=GENERATOR)
print(f"Demo ready · docs={len(_index.docs)} · analyzer={_analyzer_name} · generator={GENERATOR}")

def run_query(query: str, system: str = "adaptive", top_k: int = TOP_K, alpha: float = ALPHA, conf: float = CONF_THRESHOLD):
    query = (query or "").strip()
    if not query:
        print("Enter a question.")
        return
    if system == "B1_llm_only":
        pipe = LLMOnlyPipeline(generator=_generator)
    elif system == "B2_standard_rag":
        pipe = StandardRAGPipeline(index=_index, generator=_generator, top_k=top_k)
    elif system == "B3_fixed_temporal":
        pipe = FixedTemporalRAGPipeline(index=_index, generator=_generator, top_k=top_k, alpha=alpha)
    else:
        pipe = AdaptiveTemporalRAGPipeline(
            analyzer=_analyzer, index=_index, generator=_generator,
            top_k=top_k, candidate_k=CANDIDATE_K, alpha=alpha, conf_threshold=conf,
        )
    result = pipe.run(query)

    html = [f"<h3>Answer</h3><p style='font-size:16px'>{result.answer}</p>"]
    html.append(
        f"<p><i>Latency: {result.latency_s:.3f}s · Backend: {result.backend} · System: {system}</i></p>"
    )
    if result.analyzer is not None:
        html.append("<h3>Analyzer</h3><ul>")
        html.append(f"<li><b>needs_retrieval</b>: {result.analyzer.needs_retrieval}</li>")
        html.append(f"<li><b>temporal_class</b>: {result.analyzer.temporal_class}</li>")
        html.append(f"<li><b>confidence</b>: {result.analyzer.confidence:.3f}</li>")
        html.append(f"<li><b>policy</b>: {result.meta.get('policy')}</li>")
        if result.meta.get("adaptive_window"):
            w = result.meta["adaptive_window"]
            html.append(
                f"<li><b>adaptive_window</b>: ±{w.get('final_years')}y around {w.get('target_year')} "
                f"(reason={w.get('reason')}, expansions={w.get('expansions')})</li>"
            )
        if result.meta.get("evidence_sufficiency"):
            s = result.meta["evidence_sufficiency"]
            html.append(
                f"<li><b>evidence_sufficiency</b>: score={s.get('score')} "
                f"sufficient={s.get('sufficient')} reasons={s.get('reasons')}</li>"
            )
        html.append("</ul>")
        if result.meta.get("sufficiency_loop"):
            html.append("<h4>Sufficiency feedback rounds</h4><pre style='white-space:pre-wrap;font-size:12px'>")
            html.append(str(result.meta["sufficiency_loop"]))
            html.append("</pre>")
    elif result.meta:
        html.append(f"<h3>Meta</h3><pre>{result.meta}</pre>")
    html.append("<h3>Retrieved evidence</h3>")
    if not result.retrieved_docs:
        html.append("<p><i>No retrieval</i></p>")
    else:
        for i, (doc, score) in enumerate(zip(result.retrieved_docs, result.retrieved_scores), start=1):
            title = doc.title or doc.entity or doc.doc_id
            open_attr = " open" if i == 1 else ""
            html.append(
                f"<details{open_attr}>"
                f"<summary><b>[{i}]</b> {doc.timestamp} · score={score:.3f} · {title}</summary>"
                f"<p>{doc.text}</p>"
                f"<p><code>doc_id={doc.doc_id}</code></p></details>"
            )
    display(HTML("".join(html)))

system_dd = widgets.Dropdown(
    options=["adaptive", "B1_llm_only", "B2_standard_rag", "B3_fixed_temporal"],
    value="adaptive",
    description="System:",
)
query_box = widgets.Textarea(
    value="Who is the current Prime Minister of India?",
    description="Query:",
    layout=widgets.Layout(width="90%", height="80px"),
)
run_btn = widgets.Button(description="Run query", button_style="primary")
examples = widgets.ToggleButtons(
    options=[
        "Who is the current Prime Minister of India?",
        "Who was the President of the United States in 2018?",
        "Define photosynthesis.",
        "What is 2 + 2?",
    ],
    description="Try:",
)
out = widgets.Output()

def _on_example(change):
    if change["name"] == "value":
        query_box.value = change["new"]

def _on_click(_):
    with out:
        clear_output(wait=True)
        run_query(query_box.value, system=system_dd.value)

examples.observe(_on_example, names="value")
run_btn.on_click(_on_click)

display(widgets.VBox([
    widgets.HTML("<b>Interactive Adaptive Temporal RAG demo</b>"),
    system_dd,
    examples,
    query_box,
    run_btn,
    out,
]))

# Fallback without widgets:
# run_query("Who is the current Chancellor of Germany?", system="adaptive")
'''
)

md(
    """## 5½ — Upload / paste **your** docs → re-index → query with routing

1. Paste text or upload `.txt` / `.pdf` / `.md` with a **date** (`YYYY-MM-DD`).
2. **Merge** + **update FAISS** (PDF pages chunked via `pypdf`).
3. **Adaptive** run with step-by-step trace.

See `docs/COLAB_E2E_FLOW.md` for how this fits §1–§5.
"""
)

code(
    r'''import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown, HTML
from src.user_corpus import docs_from_text_blobs, docs_from_colab_uploads, merge_corpus_files, rebuild_index
from src.embed_index import EmbeddingIndex
from src.generate import Generator
from src.pipelines import AdaptiveTemporalRAGPipeline
from src.query_analyzer import load_query_analyzer
from src.inference_trace import format_trace_text
from src.model_config import DEFAULT_EMBED_MODEL

CORPUS_LIVE = DATA / "corpus_live.jsonl"
EMBED_MODEL = globals().get("EMBED_MODEL", DEFAULT_EMBED_MODEL)
_live_index = EmbeddingIndex()
_live_analyzer = load_query_analyzer(ARTIFACTS)
_live_gen = Generator(backend=GENERATOR)

paste_title = widgets.Text(value="My company note", description="Title:")
paste_date = widgets.Text(value="2018-06-15", description="Date:")
paste_body = widgets.Textarea(
    value="On 2018-06-15, Jane Doe was appointed CEO of Acme Corporation according to internal records.",
    description="Text:",
    layout=widgets.Layout(width="90%", height="120px"),
)
live_query = widgets.Textarea(
    value="Who was the CEO of Acme Corporation in 2018?",
    description="Question:",
    layout=widgets.Layout(width="90%", height="70px"),
)
upload_btn = widgets.Button(description="1) Merge pasted text + re-index", button_style="info")
ask_btn = widgets.Button(description="2) Run Adaptive (with trace)", button_style="primary")
live_out = widgets.Output()

def _merge_and_reindex():
    user_docs = docs_from_text_blobs([(paste_title.value, paste_body.value, paste_date.value)])
    if IN_COLAB:
        try:
            from google.colab import files
            up = files.upload()
            if up:
                user_docs.extend(docs_from_colab_uploads(up, default_timestamp=paste_date.value))
        except Exception as e:
            print("Upload skipped:", e)
    n = merge_corpus_files(DATA / "corpus.jsonl", user_docs, CORPUS_LIVE)
    rebuild_index(CORPUS_LIVE, ARTIFACTS, model_name=EMBED_MODEL)
    _live_index.load(ARTIFACTS)
    return n, len(user_docs)

def _on_merge(_):
    with live_out:
        clear_output(wait=True)
        n, added = _merge_and_reindex()
        print(f"Merged corpus: {n} docs total ({added} new user chunk(s))")
        print("Index rebuilt at", ARTIFACTS / "faiss.index")

def _on_ask(_):
    with live_out:
        clear_output(wait=True)
        if not CORPUS_LIVE.exists():
            print("Click step 1 first to merge + re-index.")
            return
        if not _live_index.docs:
            _live_index.load(ARTIFACTS)
        q = live_query.value.strip()
        pipe = AdaptiveTemporalRAGPipeline(
            analyzer=_live_analyzer, index=_live_index, generator=_live_gen,
            top_k=TOP_K, candidate_k=CANDIDATE_K, alpha=ALPHA, conf_threshold=CONF_THRESHOLD,
        )
        trace = []
        result = pipe.run(q, inference_trace=trace)
        display(Markdown(f"### Answer\n\n{result.answer}"))
        display(Markdown(
            f"**policy** `{result.meta.get('policy')}` · **retrieve** {result.did_retrieve} · "
            f"**backend** {result.backend} · **latency** {result.latency_s:.2f}s"
        ))
        if result.analyzer:
            display(Markdown(
                f"**Analyzer:** need={result.analyzer.needs_retrieval} "
                f"class={result.analyzer.temporal_class} conf={result.analyzer.confidence:.2f}"
            ))
        print(format_trace_text(trace))
        if result.retrieved_docs:
            d = result.retrieved_docs[0]
            display(HTML(f"<p><b>{d.timestamp}</b> {d.title}<br>{d.text[:500]}...</p>"))

upload_btn.on_click(_on_merge)
ask_btn.on_click(_on_ask)
display(widgets.VBox([
    widgets.HTML("<b>Your docs → FAISS → Adaptive routing</b>"),
    paste_title, paste_date, paste_body,
    widgets.HTML("<i>On Colab, step 1 also opens a file picker for .txt / .pdf / .md uploads.</i>"),
    upload_btn, live_query, ask_btn, live_out,
]))
'''
)

md(
    """## 6 — Download artifacts for local use

Zips `artifacts/` + `data/` for your PC / report.

Run **§8** before this cell if you want `inference_walkthrough.json` in the zip (or re-run §6 after §8).
"""
)

code(
    r'''import datetime as dt

stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
bundle_name = f"atr_colab_artifacts_{stamp}"
bundle_dir = Path("/content") / bundle_name if IN_COLAB else (ROOT / bundle_name)
if bundle_dir.exists():
    shutil.rmtree(bundle_dir)
bundle_dir.mkdir(parents=True)

for folder_name in ("artifacts", "data"):
    src = ROOT / folder_name
    dst = bundle_dir / folder_name
    shutil.copytree(src, dst)

zip_path = Path(str(bundle_dir) + ".zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in bundle_dir.rglob("*"):
        if path.is_file():
            zf.write(path, arcname=str(path.relative_to(bundle_dir.parent)))

print("Created", zip_path)
print("Size MB:", round(zip_path.stat().st_size / 1e6, 1))

must_have = [
    "artifacts/analyzer.pt",
    "artifacts/faiss.index",
    "artifacts/embeddings.npy",
    "artifacts/results.csv",
    "artifacts/results_charts.png",
    "artifacts/inference_walkthrough.json",
    "data/corpus.jsonl",
    "data/queries.jsonl",
]
for rel in must_have:
    ok = (bundle_dir / rel).exists()
    print(("OK  " if ok else "MISS"), rel)

if IN_COLAB:
    from google.colab import files
    files.download(str(zip_path))
else:
    print("Not on Colab — zip left at", zip_path)
'''
)

md(
    """## 8 — Full inference architecture (code walkthrough)

Runs **the same production code** as §4–§5 but logs every Adaptive stage:

1. Query analyzer → 2. Policy router → 3. FAISS → 4. Cross-encoder → 5. Temporal rerank  
6. Evidence sufficiency (+ repairs) → 7. Generator → 8. Optional post-answer retry  

Compares **B1 / B2 / B3 / Adaptive** on one query. Writes `artifacts/inference_walkthrough.json`.

If you already ran **§6 download**, run **§6** again after this cell to refresh the zip.
"""
)

code(
    r'''from src.inference_trace import (
    load_walkthrough_from_artifacts,
    print_architecture_walkthrough,
    save_walkthrough_json,
    format_trace_text,
)
from IPython.display import display, Markdown, Image
from pathlib import Path

diag = ROOT / "docs" / "diagrams"
for name in ("01_adaptive_inference_full.png", "02_policy_router.png"):
    p = diag / name
    if p.exists():
        display(Image(filename=str(p)))

WALKTHROUGH_QUERY = "Who was the President of the United States in 2018?"

walk = load_walkthrough_from_artifacts(
    WALKTHROUGH_QUERY,
    ARTIFACTS,
    generator_backend=GENERATOR,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
    alpha=ALPHA,
    conf_threshold=CONF_THRESHOLD,
)
out_json = save_walkthrough_json(walk, ARTIFACTS / "inference_walkthrough.json")
print_architecture_walkthrough(walk)
print("\nSaved:", out_json)

adaptive_trace = walk["systems"]["adaptive"].get("trace") or []
display(Markdown("### Adaptive step-by-step log"))
print(format_trace_text(adaptive_trace))
'''
)

md(
    r"""## 9 — Use the downloaded bundle on your PC

1. Unzip the download. You should see `artifacts/` and `data/`.
2. Copy them into your local project (overwrite):

```text
D:\cursor\adaptive-temporal-rag\artifacts\   ← from zip
D:\cursor\adaptive-temporal-rag\data\        ← from zip
```

3. Locally:

```powershell
cd D:\cursor\adaptive-temporal-rag
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

4. In Streamlit sidebar open **Comparative analysis** for the Colab eval charts/tables.

You do **not** need to retrain on the laptop if `artifacts/analyzer.pt` and the FAISS files are present.
"""
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT, "cells=", len(nb["cells"]))
