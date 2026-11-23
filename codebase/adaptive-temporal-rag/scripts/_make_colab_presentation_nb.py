"""Generate notebooks/01_colab_presentation.ipynb — presentation-ready Colab (no retrain)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_colab_presentation.ipynb"

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
    """# Adaptive Temporal RAG — Presentation Demo (Colab)

Set **`RUN_TRAINING`** in the next config cell:

| Value | Behavior |
|-------|----------|
| `False` (default) | Load existing `analyzer.pt` + FAISS — **demo / viva** (fast) |
| `True` | Rebuild dataset + index + **train** analyzer on GPU (~20–90+ min on T4) |

| Section | What you do |
|---------|-------------|
| 0 | Setup + install |
| 0b | **Config** — `RUN_TRAINING = True/False` |
| 1a | *(only if training)* dataset → index → train analyzer |
| 1b | Load analyzer + FAISS + live KB |
| 2 | **Real** B1/B2/B3/Adaptive comparison (held-out test) |
| 3 | Explore knowledge base |
| 4 | **Add documents in a loop** |
| 5 | Ask questions → routing + answer |
| 6 | Same query on all 4 systems |
| 7 | Download updated KB + artifacts |

**Honesty:** Comparison tables/charts come from `evaluate.py` on the **test split** when present. Base corpus is a dated synthetic benchmark; §4 uploads are your real text.
"""
)

md("## 0 — Setup")

code(
    r'''import os, sys, time, shutil, zipfile, json
from pathlib import Path
from collections import Counter

IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

def find_root(start: Path):
    for p in [start, *start.parents]:
        if (p / "src" / "pipelines.py").exists() and (p / "artifacts").exists():
            return p
    return None

ROOT = find_root(Path.cwd())
if ROOT is None:
    for cand in [
        Path("/content/adaptive-temporal-rag"),
        Path("/content/atr_presentation"),
        Path("/content/drive/MyDrive/adaptive-temporal-rag"),
        Path("/content/drive/MyDrive/atr_presentation"),
    ]:
        ROOT = find_root(cand) if cand.exists() else None
        if ROOT is not None:
            break

if ROOT is None and IN_COLAB:
    from google.colab import files, drive
    print("Mount Drive or upload atr_presentation.zip …")
    try:
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", e)
    for cand in [
        Path("/content/drive/MyDrive/atr_presentation"),
        Path("/content/drive/MyDrive/adaptive-temporal-rag"),
    ]:
        ROOT = find_root(cand) if cand.exists() else None
        if ROOT:
            break
    if ROOT is None:
        uploaded = files.upload()
        assert uploaded, "Upload atr_presentation.zip"
        zname = next(iter(uploaded))
        extract_dir = Path("/content/atr_presentation")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(zname) as zf:
            zf.extractall(extract_dir)
        ROOT = find_root(extract_dir)
        if ROOT is None and any(extract_dir.iterdir()):
            kids = [p for p in extract_dir.iterdir() if p.is_dir()]
            ROOT = find_root(kids[0]) if kids else None

assert ROOT is not None, "Could not find project root (need src/ + artifacts/)"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("ROOT =", ROOT)
print("IN_COLAB =", IN_COLAB)

%pip install -q pandas scikit-learn matplotlib plotly sentence-transformers faiss-cpu accelerate tqdm python-dateutil requests ipywidgets pypdf
%pip install -q "transformers>=4.38"

import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'''
)

md(
    """## 0b — Config (`RUN_TRAINING`)

```python
RUN_TRAINING = False   # demo: load artifacts only
RUN_TRAINING = True    # full train path (needs GPU; overwrites analyzer.pt)
```
"""
)

code(
    r'''# ========== MASTER SWITCH ==========
RUN_TRAINING = False   # True = train analyzer; False = load existing weights
# ==================================

from src.model_config import DEFAULT_EMBED_MODEL, T4_ANALYZER_MODEL

ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
ARTIFACTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

GENERATOR = "extractive"  # or "api" if keys set below
TOP_K = 8
ALPHA = 0.55
CONF_THRESHOLD = 0.50
CANDIDATE_K = 80
EMBED_MODEL = DEFAULT_EMBED_MODEL

# Used only when RUN_TRAINING = True
TARGET_QUERIES = 3000
AUGMENT_FACTOR = 6
EPOCHS = 15
BATCH_SIZE = 8
LR = 2e-5
MAX_LENGTH = 96
FREEZE_LAYERS = 0
ANALYZER_MODEL = T4_ANALYZER_MODEL

# Optional API keys (only if GENERATOR = "api")
# os.environ["GROQ_API_KEY"] = "gsk_..."
# os.environ["LLM_PROVIDER_ORDER"] = "groq,openrouter,opencode"

print("RUN_TRAINING =", RUN_TRAINING)
if RUN_TRAINING:
    print("Will rebuild dataset + FAISS + train", ANALYZER_MODEL)
    if not torch.cuda.is_available():
        print("WARNING: no GPU — training will be very slow.")
else:
    print("Will LOAD existing analyzer.pt + faiss.index (no training).")
'''
)

md(
    """## 1a — Train path (runs only if `RUN_TRAINING = True`)

Skipped automatically when `RUN_TRAINING = False`. When True: build dataset → FAISS → fine-tune analyzer.
"""
)

code(
    r'''from src.data_schema import load_corpus, load_queries
from src.embed_index import build_index_from_corpus
from src.query_analyzer import train_analyzer, QueryAnalyzerService

if not RUN_TRAINING:
    print("Skipped §1a (RUN_TRAINING=False). Jump to §1b.")
else:
    print(f"Building dataset TARGET_QUERIES={TARGET_QUERIES}")
    !python {ROOT}/scripts/build_dataset.py --target-queries {TARGET_QUERIES} --out-dir {DATA}
    docs = load_corpus(DATA / "corpus.jsonl")
    queries = load_queries(DATA / "queries.jsonl")
    print(f"Corpus={len(docs)} queries={len(queries)}")

    t0 = time.time()
    build_index_from_corpus(DATA / "corpus.jsonl", ARTIFACTS, model_name=EMBED_MODEL)
    print(f"Index built in {time.time()-t0:.1f}s")

    train_all = [q for q in queries if q.split == "train"]
    train = [q for q in train_all if not q.is_paraphrase]
    val = [q for q in queries if q.split == "val"]
    print(f"Train seeds={len(train)} val={len(val)} augment={AUGMENT_FACTOR} epochs={EPOCHS}")

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
        model_name=ANALYZER_MODEL,
    )
    print(f"Best val={result['best_val_score']:.4f} wall_min={(time.time()-t0)/60:.1f}")
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

md("## 1b — Load analyzer + FAISS + live knowledge base")

code(
    r'''from src.generate import Generator
from src.query_analyzer import load_query_analyzer
from src.user_corpus import KnowledgeBase

assert (ARTIFACTS / "analyzer.pt").exists(), (
    "Missing analyzer.pt — set RUN_TRAINING=True once, or use atr_presentation.zip with weights."
)
assert (ARTIFACTS / "faiss.index").exists(), "Missing faiss.index — run §1a with RUN_TRAINING=True or restore zip."

kb = KnowledgeBase(DATA, ARTIFACTS, model_name=EMBED_MODEL)
# After training, refresh live corpus from new base; after demo-only, keep prior user docs
kb.ensure_live_corpus(reset=bool(RUN_TRAINING))
index = kb.load_or_build_index(force_rebuild=bool(RUN_TRAINING))
analyzer = load_query_analyzer(ARTIFACTS)
generator = Generator(backend=GENERATOR)

meta = json.loads((ARTIFACTS / "analyzer_meta.json").read_text(encoding="utf-8"))
print("RUN_TRAINING was", RUN_TRAINING)
print("Analyzer:", meta.get("model_name"), "| best_val_score:", round(float(meta.get("best_val_score", 0)), 4))
print("KB stats:", kb.stats())
print("Ready.")
'''
)

md(
    """## 2 — Real comparative results (held-out test)

Loaded from your Colab eval run (`results_summary.json`, `results.csv`, charts).

- **77 test queries**, same split for B1 / B2 / B3 / Adaptive  
- Metrics from `scripts/evaluate.py` — **not hand-written**  
- Generator used in that run: check notebook notes / comparative report (often extractive)
"""
)

code(
    r'''from IPython.display import display, Markdown, Image
import pandas as pd
import matplotlib.pyplot as plt

summary = json.loads((ARTIFACTS / "results_summary.json").read_text(encoding="utf-8"))
order = ["B1_llm_only", "B2_standard_rag", "B3_fixed_temporal", "adaptive"]
labels = {
    "B1_llm_only": "B1 LLM-only",
    "B2_standard_rag": "B2 Standard RAG",
    "B3_fixed_temporal": "B3 Fixed Temporal",
    "adaptive": "Adaptive (ours)",
}

rows = []
for k in order:
    m = summary[k]
    rows.append({
        "System": labels[k],
        "N": int(m["n"]),
        "Contains": round(m["contains"], 3),
        "EM": round(m["em"], 3),
        "Token F1": round(m["token_f1"], 3),
        "Temp Acc": round(m["temporal_acc"], 3),
        "P@5": round(m["precision_at_k"], 3),
        "R@5": round(m["recall_at_k"], 3),
        "Ret Rate": round(m["retrieval_rate"], 3),
        "Latency (s)": round(m["latency_s"], 3),
    })
df = pd.DataFrame(rows)
display(Markdown("### Test-split summary (real eval)"))
display(df)

# Bar charts
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics = [("Contains", "Contains"), ("Temp Acc", "Temporal accuracy"), ("Ret Rate", "Retrieval rate")]
colors = ["#6b7280", "#3b82f6", "#0d9488", "#b45309"]
for ax, (col, title) in zip(axes, metrics):
    ax.bar(df["System"], df[col], color=colors)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(df[col]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
fig.suptitle("Held-out test (n=77) — from results_summary.json", fontsize=12)
fig.tight_layout()
plt.show()

charts = ARTIFACTS / "results_charts.png"
if charts.exists():
    display(Markdown("### Saved comparative chart (from evaluate.py)"))
    display(Image(filename=str(charts)))

report = ARTIFACTS / "comparative_analysis.md"
if report.exists():
    display(Markdown(report.read_text(encoding="utf-8")))

# Per-class viz from results.csv if present
csv_path = ARTIFACTS / "results.csv"
if csv_path.exists():
    rdf = pd.read_csv(csv_path)
    if {"system", "temporal_class", "contains"}.issubset(rdf.columns):
        pivot = rdf.groupby(["system", "temporal_class"])["contains"].mean().unstack()
        display(Markdown("### Contains by temporal class (from results.csv)"))
        display(pivot.round(3))
        ax = pivot.reindex(order).plot(kind="bar", figsize=(10, 4), rot=20)
        ax.set_ylabel("Contains")
        ax.set_title("Contains by temporal class")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.show()
'''
)

md("## 3 — Knowledge base snapshot")

code(
    r'''from src.data_schema import load_corpus

docs = load_corpus(kb.live_corpus)
user_docs = [d for d in docs if d.source == "user_upload"]
print(f"Total passages: {len(docs)} | User-added: {len(user_docs)} | Base: {len(docs)-len(user_docs)}")

years = []
for d in docs:
    try:
        years.append(int(str(d.timestamp)[:4]))
    except Exception:
        pass
if years:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(years, bins=range(min(years), max(years) + 2), color="#0d9488", edgecolor="white")
    ax.set_title("Document timestamps in live KB (year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

display(Markdown("### Sample base docs"))
for d in docs[:3]:
    print(f"[{d.timestamp}] {d.title or d.entity or d.doc_id}")
    print(d.text[:220].replace('\\n', ' '), "…\n")

if user_docs:
    display(Markdown("### Your added docs (latest)"))
    for d in user_docs[-5:]:
        print(f"[{d.timestamp}] {d.title} ({d.doc_id})")
        print(d.text[:220], "…\n")
'''
)

md(
    """## 4 — Add documents in a loop (knowledge base)

Use this cell **many times**. Each click **appends** to the live KB and updates FAISS (does not wipe earlier uploads).

1. Edit title / date / text, **or** upload `.txt` / `.pdf` / `.md` when the picker opens.  
2. Click **Add to knowledge base** (paste) or **Upload .txt / .pdf + add**.  
3. Repeat — each add appends (PDF pages are split into chunks).  
4. Click **Reset to base corpus** only if you want to drop all user docs.
"""
)

code(
    r'''import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

title_w = widgets.Text(value="Acme CEO note", description="Title:", layout=widgets.Layout(width="90%"))
date_w = widgets.Text(value="2018-06-15", description="Date:", layout=widgets.Layout(width="90%"))
body_w = widgets.Textarea(
    value="On 2018-06-15, Jane Doe was appointed CEO of Acme Corporation according to internal records.",
    description="Text:",
    layout=widgets.Layout(width="90%", height="140px"),
)
add_btn = widgets.Button(description="Add to knowledge base", button_style="primary")
upload_btn = widgets.Button(description="Upload .txt / .pdf + add", button_style="info")
reset_btn = widgets.Button(description="Reset to base corpus", button_style="danger")
status = widgets.Output()

def _show_stats():
    s = kb.stats()
    print(f"KB total={s['total_docs']} | user={s['user_docs']} | base={s['base_docs']} | index={s['index_docs']}")

def _on_add(_):
    with status:
        clear_output(wait=True)
        info = kb.add_from_paste(title_w.value, body_w.value, date_w.value)
        print("Add result:", info)
        _show_stats()

def _on_upload(_):
    with status:
        clear_output(wait=True)
        if not IN_COLAB:
            print("File picker only works on Google Colab. Use paste add locally, or pass PDF bytes via docs_from_upload_bytes().")
            return
        from google.colab import files
        print("Select .txt and/or .pdf files …")
        up = files.upload()
        if not up:
            print("No files selected.")
            return
        info = kb.add_from_uploads(up, default_timestamp=date_w.value or None)
        print("Upload add result:", info)
        for name in up:
            kind = "PDF" if str(name).lower().endswith(".pdf") else "text"
            print(f"  · {name} ({kind}, {len(up[name])} bytes)")
        _show_stats()

def _on_reset(_):
    with status:
        clear_output(wait=True)
        info = kb.reset_to_base()
        print("Reset:", info)
        _show_stats()

add_btn.on_click(_on_add)
upload_btn.on_click(_on_upload)
reset_btn.on_click(_on_reset)

display(widgets.VBox([
    widgets.HTML("<b>Loop: add as many documents as you want</b>"),
    title_w, date_w, body_w,
    widgets.HBox([add_btn, upload_btn, reset_btn]),
    status,
]))
with status:
    _show_stats()
'''
)

md(
    """## 5 — Ask a question (Adaptive routing)

Uses the **live KB** (base + everything you added in §4). Shows analyzer → policy → sufficiency → answer + top evidence.
"""
)

code(
    r'''from src.pipelines import AdaptiveTemporalRAGPipeline
from src.inference_trace import format_trace_text
from IPython.display import HTML

q_box = widgets.Textarea(
    value="Who was the CEO of Acme Corporation in 2018?",
    description="Query:",
    layout=widgets.Layout(width="90%", height="80px"),
)
examples = widgets.ToggleButtons(
    options=[
        "Who was the CEO of Acme Corporation in 2018?",
        "Who is the current Prime Minister of India?",
        "Who was the President of the United States in 2018?",
        "Define photosynthesis.",
        "What is 2 + 2?",
    ],
    description="Try:",
)
run_btn = widgets.Button(description="Run Adaptive", button_style="success")
out = widgets.Output()

def _run(_=None):
    with out:
        clear_output(wait=True)
        # refresh index object after adds
        index = kb.index
        if not index.docs:
            kb.load_or_build_index()
            index = kb.index
        pipe = AdaptiveTemporalRAGPipeline(
            analyzer=analyzer,
            index=index,
            generator=generator,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            alpha=ALPHA,
            conf_threshold=CONF_THRESHOLD,
        )
        q = q_box.value.strip()
        trace = []
        result = pipe.run(q, inference_trace=trace)
        display(Markdown("### Answer\n\n" + (result.answer or "")))
        if result.analyzer:
            a = result.analyzer
            display(Markdown(
                "**Analyzer** need=`{}` · class=`{}` · conf=`{:.2f}` · **policy** `{}` · "
                "retrieve={} · latency={:.2f}s".format(
                    a.needs_retrieval, a.temporal_class, a.confidence,
                    result.meta.get("policy"), result.did_retrieve, result.latency_s,
                )
            ))
        print(format_trace_text(trace))
        if result.retrieved_docs:
            display(Markdown("### Retrieved evidence"))
            for i, (doc, score) in enumerate(zip(result.retrieved_docs[:5], result.retrieved_scores[:5]), 1):
                src = "USER" if doc.source == "user_upload" else "base"
                display(HTML(
                    f"<details {'open' if i==1 else ''}><summary><b>[{i}]</b> {doc.timestamp} · "
                    f"score={score:.3f} · <i>{src}</i> · {doc.title or doc.doc_id}</summary>"
                    f"<p>{doc.text}</p></details>"
                ))
        # small score chart
        if result.retrieved_scores:
            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.barh(
                [f"#{i+1}" for i in range(len(result.retrieved_scores[:5]))][::-1],
                list(result.retrieved_scores[:5])[::-1],
                color="#b45309",
            )
            ax.set_xlabel("Score")
            ax.set_title("Top retrieved scores")
            plt.tight_layout()
            plt.show()

def _ex(change):
    if change["name"] == "value":
        q_box.value = change["new"]

examples.observe(_ex, names="value")
run_btn.on_click(_run)
display(widgets.VBox([examples, q_box, run_btn, out]))
'''
)

md("## 6 — Same query on all four systems (live demo)")

code(
    r'''from src.inference_trace import trace_all_systems, print_architecture_walkthrough, save_walkthrough_json

demo_q = q_box.value.strip() or "Who is the current Prime Minister of India?"
walk = trace_all_systems(
    demo_q,
    index=kb.index,
    generator=generator,
    analyzer=analyzer,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
    alpha=ALPHA,
    conf_threshold=CONF_THRESHOLD,
)
save_walkthrough_json(walk, ARTIFACTS / "inference_walkthrough.json")
print_architecture_walkthrough(walk)

# compact comparison table for this one query
rows = []
for name, info in walk["systems"].items():
    rows.append({
        "System": name,
        "Retrieve": info.get("did_retrieve"),
        "n_docs": info.get("n_docs"),
        "Latency": info.get("latency_s"),
        "Answer": (info.get("answer_preview") or "")[:120],
    })
display(pd.DataFrame(rows))
'''
)

md(
    """## 7 — Download presentation bundle

Downloads `artifacts/` + `data/` (including `corpus_live.jsonl` and any user docs you added).
"""
)

code(
    r'''import datetime as dt

stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
bundle_name = f"atr_presentation_bundle_{stamp}"
bundle_dir = Path("/content") / bundle_name if IN_COLAB else (ROOT / bundle_name)
if bundle_dir.exists():
    shutil.rmtree(bundle_dir)
bundle_dir.mkdir(parents=True)

for folder_name in ("artifacts", "data"):
    src = ROOT / folder_name
    if src.exists():
        shutil.copytree(src, bundle_dir / folder_name)

zip_path = Path(str(bundle_dir) + ".zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in bundle_dir.rglob("*"):
        if path.is_file():
            zf.write(path, arcname=str(path.relative_to(bundle_dir.parent)))

print("Created", zip_path, "MB:", round(zip_path.stat().st_size / 1e6, 1))
if IN_COLAB:
    from google.colab import files
    files.download(str(zip_path))
'''
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT, "cells=", len(nb["cells"]))
