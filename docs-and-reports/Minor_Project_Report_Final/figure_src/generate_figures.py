"""Generate the 9 report figures from verified_facts.json (commit 44938a7). 300 DPI PNG into ../assets/figures."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
F = json.load(open(ROOT / "verified_facts.json"))
FR = F["final_results"]["datasets"]
SIG = F["significance"]["datasets"]

NAVY, TEAL, SLATE, AMBER, CRIMSON = "#1B365D", "#2B7A78", "#4A5568", "#D97706", "#B91C1C"
LIGHT = {"navy": "#E3EAF4", "teal": "#DDF0EE", "slate": "#EDF0F4", "amber": "#FDF0DC", "crimson": "#FBE5E5", "green": "#E6F2E1"}
EDGE = {"navy": NAVY, "teal": TEAL, "slate": SLATE, "amber": AMBER, "crimson": CRIMSON, "green": "#3F7D20"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.edgecolor": SLATE, "axes.labelcolor": "#1F2937",
    "xtick.color": "#1F2937", "ytick.color": "#1F2937", "axes.grid": False, "savefig.dpi": 300})

SYS = ["B1_llm_only", "B2_standard_rag", "B3_fixed_temporal", "tempralm_decay", "adaptive", "adaptive_fast"]
SYSLAB = {"B1_llm_only": "B1 No retrieval", "B2_standard_rag": "B2 Standard RAG", "B3_fixed_temporal": "B3 Fixed ±2y",
          "tempralm_decay": "B4 TempRALM-decay", "adaptive": "Adaptive", "adaptive_fast": "Adaptive-Fast"}
SYSCOL = {"B1_llm_only": "#A0AEC0", "B2_standard_rag": SLATE, "B3_fixed_temporal": AMBER, "tempralm_decay": CRIMSON,
          "adaptive": NAVY, "adaptive_fast": TEAL}
DS = ["synthetic", "heldout", "timeqa"]
DSLAB = {"synthetic": "Templated (n=77)", "heldout": "Held-out (n=193)", "timeqa": "TimeQA (n=500)"}

def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig); print("saved", name)

def box(ax, x, y, w, h, text, fc="navy", fs=9, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.8", fc=LIGHT[fc], ec=EDGE[fc], lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111827",
            fontweight="bold" if bold else "normal", linespacing=1.25)

def diamond(ax, cx, cy, w, h, text, fs=8.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(plt.Polygon(pts, fc=LIGHT["amber"], ec=AMBER, lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, linespacing=1.2)

def arrow(ax, p1, p2, text=None, col=SLATE, fs=8, toff=(0, 0), ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, color=col, lw=1.2, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + toff[0], (p1[1] + p2[1]) / 2 + toff[1], text, fontsize=fs, ha="center", va="center",
                color="#111827", bbox=dict(fc="white", ec="none", pad=0.6))

def poly(ax, pts, col=SLATE, text=None, tpos=None, fs=8):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=col, lw=1.2)
    arrow(ax, pts[-2], pts[-1], col=col)
    if text: ax.text(*tpos, text, fontsize=fs, ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.6))

def canvas(w, h, W=100, H=100):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); return fig, ax

def fig01():
    fig, ax = canvas(11, 13.5, 100, 132)
    ax.text(50, 130, "Adaptive Temporal RAG — frozen Checkpoint B pipeline (configs/final.yaml)", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    box(ax, 35, 120, 30, 6, "User query  $q$", "slate", bold=True, fs=10)
    box(ax, 25, 108, 50, 8.5, "Query Analyser (distilbert-base-uncased, multitask)\nnecessity head $p^{nec}$ · temporal head $p^{tmp}$ (current / historical / timeless)\nconfidence $\\gamma$ (blended; head untrained)", "navy", fs=8.8)
    arrow(ax, (50, 120), (50, 116.5))
    diamond(ax, 50, 99, 34, 9, "necessity = 0  AND\ntrivia whitelist match?")
    arrow(ax, (50, 108), (50, 103.5))
    box(ax, 4, 94.5, 20, 9, "No-retrieval path\nextractive reader,\nempty document list\n(policy no_retrieval)", "green", fs=8.3)
    arrow(ax, (33, 99), (24, 99), "yes", toff=(0, 1.6))
    box(ax, 33, 84, 34, 7.5, "Policy Router\n$\\gamma < 0.55$ → semantic; else temporal class", "crimson", fs=8.8)
    arrow(ax, (50, 94.5), (50, 91.5), "no", toff=(2.5, 0))
    ax.add_patch(FancyBboxPatch((3, 47), 94, 33, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#F7FAFC", ec=TEAL, lw=1.2, ls="--"))
    ax.text(5, 77.6, "Temporal retriever", ha="left", fontsize=10, fontweight="bold", color=TEAL)
    arrow(ax, (50, 84), (50, 75.8))
    box(ax, 6, 66, 26, 9, "Dense retrieval\nBAAI/bge-small-en-v1.5\nFAISS IndexFlatIP · 80 candidates", "teal", fs=8.3)
    box(ax, 37, 66, 26, 9, "Cross-encoder rerank\nms-marco-MiniLM-L-6-v2 · keep 40\n$s_i=0.35\\,\\tilde s^f_i+0.65\\,s^{ce}_i$", "teal", fs=8.3)
    box(ax, 68, 66, 26, 9, "Entity / surname cut\n(historical branch;\nshared by B2, B3, Adaptive)", "teal", fs=8.3)
    poly(ax, [(50, 75.8), (19, 75.8), (19, 75)])
    arrow(ax, (32, 70.5), (37, 70.5)); arrow(ax, (63, 70.5), (68, 70.5))
    box(ax, 6, 50, 26, 11.5, "Current mode\nlexical tie, sort by latest $\\tau_i$\n$g^{cur}_i=\\max(0,\\,1-\\max(r-\\tau_i,0)/3650)$", "navy", fs=8.2)
    box(ax, 37, 50, 26, 11.5, "Historical mode — adaptive window\n$h\\leftarrow\\min(h_{max},1.5h+0.25)$\nuntil an in-window survivor\n$g^{hist}_i$ from $\\delta(d_i,t)$", "amber", fs=8.2)
    box(ax, 68, 50, 26, 11.5, "Semantic mode\n(low confidence / soft timeless)\n$f_i=\\alpha\\tilde s_i+(1-\\alpha)g_i\\tilde s_i+b_i$\n$\\alpha=0.6$", "slate", fs=8.2)
    poly(ax, [(81, 66), (81, 63.5), (19, 63.5), (19, 61.5)]); arrow(ax, (50, 63.5), (50, 61.5)); arrow(ax, (81, 63.5), (81, 61.5))
    box(ax, 30, 35, 40, 8, "Evidence-sufficiency judge (top $k=5$)\n$S=0.35C+0.25\\min(1,s_{(1)})+0.15\\bar s+0.15T+0.10\\min(1,b_{(1)}/0.25)$", "navy", fs=8.2)
    poly(ax, [(19, 50), (19, 46), (50, 46), (50, 43)]); arrow(ax, (50, 50), (50, 43)); poly(ax, [(81, 50), (81, 46), (50, 46), (50, 43)])
    box(ax, 74, 30.5, 24, 13.5, "Repairs (≤ 2 rounds = 3 retrieves;\n60 s wall-clock budget)\nwiden_window · expand_candidates\nraise_topk · relax_to_semantic\nboost_recency", "crimson", fs=7.9)
    arrow(ax, (70, 39), (74, 39), "S < 0.55", toff=(0, 2.0), fs=7.5)
    poly(ax, [(96, 44), (96, 47.8)], col=CRIMSON)
    ax.text(97.2, 46, "retry", rotation=90, fontsize=7.5, va="center", color=CRIMSON)
    box(ax, 4, 31, 23, 9, "Adaptive-Fast variant\nsufficiency loop disabled\n(ablation-derived)", "green", fs=8.2, ls="--")
    arrow(ax, (27, 35.5), (30, 37.5), col="#3F7D20", ls="--")
    box(ax, 25, 19, 50, 9.5, "Temporal sentence selector (date agreement, overlap, rank bonus;\n<24-char fragments joined, <48-char dated clauses prefixed, 400-char cap)\n→ extractive reader → answer projection (src/answer_projection.py)", "teal", fs=8.2)
    arrow(ax, (50, 35), (50, 28.5), "S ≥ 0.55 or budget spent", toff=(0, 0.2), fs=7.5)
    box(ax, 30, 7, 40, 7.5, "Answer $\\hat y$ + metadata\npolicy · adaptive_window · evidence_sufficiency · latency", "slate", fs=8.5)
    arrow(ax, (50, 19), (50, 14.5))
    poly(ax, [(4, 97), (1.2, 97), (1.2, 10.75), (30, 10.75)], col="#3F7D20")
    box(ax, 74, 6, 24, 11, "Baseline B4 (tempralm_decay)\ndense $\\cup$ BM25, equal blend\n× $\\exp(-|\\Delta y|/2)$\nno hard window, no CE,\nno sufficiency loop", "slate", fs=7.8, ls=":")
    save(fig, "fig01_system_architecture.png")

def fig02():
    fig, ax = canvas(13, 11.6, 130, 105); ax.set_ylim(-12, 105)
    ax.text(65, 103, "Experimental design and decision flow", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(30, 98.5, "(a) Per-query routing and guardrails (Adaptive)", ha="center", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.text(97, 98.5, "(b) Frozen evaluation protocol", ha="center", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.plot([62, 62], [-11, 97], color="#CBD5E0", lw=1)
    box(ax, 18, 89, 24, 6, "Query → Analyser", "slate", bold=True)
    diamond(ax, 30, 80, 30, 8, "Temporal class / confidence?", fs=8.5)
    arrow(ax, (30, 89), (30, 84))
    box(ax, 1, 63, 13, 9, "Timeless\n(whitelist → skip;\nelse soft lookup)", "green", fs=7.8)
    box(ax, 16, 63, 12.5, 9, "Current\nlatest $\\tau$,\n$g^{cur}$", "navy", fs=7.8)
    box(ax, 30.5, 63, 14, 9, "Historical —\nexact date / year", "amber", fs=7.8)
    box(ax, 46.5, 63, 14, 9, "Historical —\nvague (decade /\nno year)", "amber", fs=7.8)
    for x in (7.5, 22.25, 37.5, 53.5): poly(ax, [(30, 76), (30, 74), (x, 74), (x, 72)])
    ax.text(48, 79.5, "$\\gamma<0.55$ → semantic", fontsize=7.8, color=CRIMSON)
    tb = [("Cue", "init h", "max h"), ("Exact calendar date", "0.35", "1.5"), ("Early/mid/late in year", "1.5", "3.5"),
          ("Bare year", "1.0", "4.0"), ("Decade", "5.0", "8.0"), ("Historical, no year", "3.0", "8.0")]
    y0 = 58
    ax.add_patch(Rectangle((31, y0 - 6 * 3.2 - 0.6), 30, 6 * 3.2 + 1.2, fc="white", ec=AMBER, lw=1))
    for i, (a, b, c) in enumerate(tb):
        yy = y0 - 1.4 - i * 3.2; w = "bold" if i == 0 else "normal"
        ax.text(32, yy, a, fontsize=7.6, va="center", fontweight=w); ax.text(52.5, yy, b, fontsize=7.6, va="center", ha="center", fontweight=w); ax.text(58.2, yy, c, fontsize=7.6, va="center", ha="center", fontweight=w)
    ax.text(46, y0 + 1.5, "Window half-widths (years)", fontsize=7.8, ha="center", style="italic")
    poly(ax, [(37.5, 63), (37.5, 61.5), (46, 61.5), (46, 60.2)]); poly(ax, [(53.5, 63), (53.5, 61.5), (46, 61.5)])
    diamond(ax, 40, 31, 18, 8, "In-window\nsurvivor?\n($\\delta\\leq365.25h$)", fs=7.4)
    poly(ax, [(40, 38.6), (40, 35)])
    box(ax, 51, 27, 10, 8, "grow $h$:\n$\\min(h_{max},$\n$1.5h+0.25)$", "amber", fs=7.2)
    arrow(ax, (49, 31), (51, 31)); ax.text(50, 32.4, "no", fontsize=7.4, ha="center")
    poly(ax, [(56, 35), (56, 37), (40.2, 37)], col=AMBER)
    box(ax, 2, 16, 58, 6, "rank top $k=5$ (all retrieving modes) → evidence-sufficiency score $S$", "navy", fs=8)
    arrow(ax, (40, 27), (40, 22), "yes", toff=(2.2, 0.3))
    poly(ax, [(7.5, 63), (7.5, 22)]); poly(ax, [(22.25, 63), (22.25, 22)])
    diamond(ax, 20, 8, 32, 8.5, "$S\\geq0.55$ or rounds = 2\nor 60 s elapsed?", fs=8)
    arrow(ax, (20, 16), (20, 12.25))
    box(ax, 40, 4.5, 19, 7, "repair & re-retrieve\n(max 3 retrieves)", "crimson", fs=7.6)
    arrow(ax, (36, 8), (40, 8), "no", toff=(0, 1.5), fs=7.5)
    poly(ax, [(49.5, 11.5), (49.5, 16)], col=CRIMSON)
    box(ax, 5, -10.5, 30, 7, "sentence selector → reader →\nanswer projection → $\\hat y$ (logged)", "teal", fs=7.8)
    arrow(ax, (20, 3.75), (20, -3.5), "yes", toff=(2.2, 0))
    box(ax, 70, 86, 54, 8, "Frozen config  configs/final.yaml  (checkpoint_b_final)\nshared reader · answer projection · entity/surname cut · sentence selector", "slate", fs=8.3)
    box(ax, 66, 71, 18, 9, "Templated test\nn = 77\n(803 passages)", "navy", fs=8)
    box(ax, 86, 71, 18, 9, "Held-out test\nn = 193\n(six unseen entities)", "navy", fs=8)
    box(ax, 106, 71, 18, 9, "TimeQA test\nn = 500\n(5,931 chunks)", "navy", fs=8)
    for x in (75, 95, 115): poly(ax, [(97, 86), (97, 83), (x, 83), (x, 80)])
    box(ax, 66, 55, 58, 10, "Six systems: B1 no retrieval · B2 standard RAG · B3 fixed ±2y window\nB4 TempRALM-style decay · Adaptive · Adaptive-Fast (sufficiency off, ablation-derived)", "teal", fs=8.3)
    poly(ax, [(75, 71), (75, 67), (95, 67), (95, 65)]); arrow(ax, (95, 71), (95, 65)); poly(ax, [(115, 71), (115, 67), (95, 67), (95, 65)])
    box(ax, 66, 40, 28, 10, "Metrics per query\nEM · token F1 · Contains\nP@5 · R@5 · temporal acc.\nretrieval rate · mean/median s", "slate", fs=8)
    box(ax, 96, 40, 28, 10, "15 ablation runs (Adaptive)\nrouter · adaptive window ·\nsufficiency loop · entity cut ·\nsentence selector  × 3 sets", "amber", fs=8)
    poly(ax, [(95, 55), (95, 52.5), (80, 52.5), (80, 50)]); poly(ax, [(95, 52.5), (110, 52.5), (110, 50)])
    box(ax, 66, 18, 58, 10, "Paired significance on saved per-query rows\nexact two-sided McNemar (discordant pairs)\n10,000-resample paired bootstrap 95% CI (seed 0)", "crimson", fs=8.3)
    poly(ax, [(80, 40), (80, 34), (95, 34), (95, 28)]); poly(ax, [(110, 40), (110, 34), (95, 34)])
    box(ax, 66, -9, 58, 10, "artifacts/final_results.md · final_results.json · significance.json\nscripts/e2e_smoke_test.py: PASS = 72, FAIL = 0 · eval_timings.json", "green", fs=8.3)
    arrow(ax, (95, 18), (95, 1))
    save(fig, "fig02_experiment_flowchart.png")

def ext(ax, x, y, w, h, t, fs=8.5):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=NAVY, lw=1.6)); ax.add_patch(Rectangle((x + 0.8, y - 0.8), w, h, fc="none", ec=NAVY, lw=0.6))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold", color=NAVY)
def proc(ax, cx, cy, r, t, fs=8):
    ax.add_patch(Ellipse((cx, cy), 2 * r * 1.35, 2 * r, fc=LIGHT["teal"], ec=TEAL, lw=1.4))
    ax.text(cx, cy, t, ha="center", va="center", fontsize=fs, linespacing=1.2)
def store(ax, x, y, w, h, t, fs=7.8):
    ax.add_patch(Rectangle((x, y), w, h, fc=LIGHT["slate"], ec="none", zorder=0))
    ax.plot([x, x + w], [y + h, y + h], color=SLATE, lw=1.3); ax.plot([x, x + w], [y, y], color=SLATE, lw=1.3); ax.plot([x, x], [y, y + h], color=SLATE, lw=1.3)
    ax.text(x + 1, y + h / 2, t, va="center", fontsize=fs)
def fig03():
    fig, ax = canvas(13, 15, 130, 150)
    ax.text(65, 148, "Data flow diagrams of Adaptive Temporal RAG", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(65, 143, "(a) Level-0 context diagram", ha="center", fontsize=10.5, fontweight="bold", color=SLATE)
    ext(ax, 4, 122, 22, 9, "User / Evaluator")
    proc(ax, 65, 126, 9, "0.0\nAdaptive\nTemporal RAG", fs=9)
    ext(ax, 104, 131, 23, 8, "Dated corpus\n(templated / TimeQA)")
    ext(ax, 104, 114, 23, 8, "Benchmark query sets\n(gold answers, doc ids)")
    arrow(ax, (26, 129), (52.5, 129), "question, system choice", toff=(0, 1.6))
    arrow(ax, (52.5, 123), (26, 123), "answer, evidence, router meta", toff=(0, -1.8))
    arrow(ax, (104, 135), (77.5, 129), "passages + timestamps", toff=(-1, 2.6))
    arrow(ax, (104, 118), (77.5, 123), "queries for evaluation", toff=(0, -2.4))
    ax.plot([3, 127], [108, 108], color="#CBD5E0", lw=1)
    ax.text(65, 104, "(b) Level-1 data flow diagram", ha="center", fontsize=10.5, fontweight="bold", color=SLATE)
    ext(ax, 2, 86, 18, 8, "User")
    proc(ax, 36, 90, 6.5, "1.0\nAnalyse\nquery"); proc(ax, 66, 90, 6.5, "2.0\nRoute\npolicy"); proc(ax, 98, 90, 6.5, "3.0\nRetrieve &\nrerank")
    proc(ax, 98, 60, 6.5, "4.0\nTemporal\nwindow / cut"); proc(ax, 66, 60, 6.5, "5.0\nSufficiency\ncheck"); proc(ax, 36, 60, 6.5, "6.0\nSelect, read\n& project")
    proc(ax, 66, 26, 6.5, "7.0\nEvaluate &\ntest")
    arrow(ax, (20, 90), (27.2, 90), "query", toff=(0, 1.6))
    arrow(ax, (44.8, 90), (57.2, 90), "class, $\\gamma$", toff=(0, 1.6))
    arrow(ax, (74.8, 90), (89.2, 90), "mode, k", toff=(0, 1.6))
    arrow(ax, (98, 83.5), (98, 66.5), "top-40 pool", toff=(5.5, 0))
    arrow(ax, (89.2, 60), (74.8, 60), "top-5 docs", toff=(0, 1.6))
    arrow(ax, (57.2, 60), (44.8, 60), "accepted evidence", toff=(0, 1.6))
    arrow(ax, (72, 65), (92, 85), "repair (≤2)", col=CRIMSON, toff=(-3, 1), rad=-0.15)
    arrow(ax, (66, 83.5), (40, 66), "no_retrieval", col="#3F7D20", toff=(-2, 2))
    arrow(ax, (29, 62), (14, 86), "answer + meta", toff=(-5, 0))
    arrow(ax, (40, 54), (60, 30.5), "predictions, latency", toff=(-7, 0))
    store(ax, 8, 98, 22, 4.5, "D3  analyzer weights"); store(ax, 104, 98, 24, 4.5, "D2  FAISS index / BM25")
    store(ax, 104, 46, 24, 4.5, "D1  dated corpus (CorpusDoc)"); store(ax, 40, 98, 22, 4.5, "D4  configs/final.yaml")
    store(ax, 4, 36, 24, 4.5, "D5  benchmark queries (QueryItem)"); store(ax, 90, 18, 34, 4.5, "D6  artifacts/general_eval/final, significance.json")
    arrow(ax, (19, 98), (32, 95.5), ls="--"); arrow(ax, (51, 98), (62, 95.8), ls="--")
    arrow(ax, (112, 98), (103, 95.5), ls="--"); arrow(ax, (112, 50.5), (102, 54.5), ls="--")
    arrow(ax, (16, 36), (60, 26), "gold answers, gold ids", ls="--", toff=(0, 3))
    arrow(ax, (72.5, 24), (90, 20.25), "tables, CIs, p-values", toff=(0, -2.5))
    ax.text(3, 6, "Notation: rounded ellipses = processes; double-edged boxes = external entities; open-ended bars = data stores; dashed = store reads.", fontsize=8, style="italic", color=SLATE)
    save(fig, "fig03_dfd_level0_level1.png")

def entity(ax, x, y, w, name, attrs, col="navy"):
    h = 3.2 + 2.55 * len(attrs); c = EDGE[col]
    ax.add_patch(Rectangle((x, y - h), w, h, fc="white", ec=c, lw=1.4)); ax.add_patch(Rectangle((x, y - 3.2), w, 3.2, fc=c, ec="none"))
    ax.text(x + w / 2, y - 1.6, name, ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    for i, a in enumerate(attrs):
        ax.text(x + 0.8, y - 3.2 - 1.3 - i * 2.55, a, fontsize=7.4, va="center", fontweight="bold" if a.startswith("PK") else "normal", style="italic" if a.startswith("FK") else "normal")
def card(ax, p, q, t):
    dx, dy = q[0] - p[0], q[1] - p[1]; L = math.hypot(dx, dy); vert = abs(dy) > abs(dx)
    ax.text(p[0] + dx / L * 1.6 + (1.4 if vert else 0), p[1] + dy / L * 1.6 + (0 if vert else 1.0), t, fontsize=7.6, fontweight="bold", color=CRIMSON, ha="left" if vert else "center", va="center")
def lbl(ax, pos, t):
    ax.text(*pos, t, fontsize=7.2, ha="center", va="center", style="italic", bbox=dict(boxstyle="round,pad=0.25", fc=LIGHT["amber"], ec=AMBER, lw=0.8))
def rel(ax, p1, p2, label, c1, c2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=SLATE, lw=1.1); lbl(ax, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2), label)
    card(ax, p1, p2, c1); card(ax, p2, p1, c2)
def relpath(ax, pts, label, lpos, c1, c2):
    xs, ys = zip(*pts); ax.plot(xs, ys, color=SLATE, lw=1.1); lbl(ax, lpos, label)
    card(ax, pts[0], pts[1], c1); card(ax, pts[-1], pts[-2], c2)
def fig04():
    fig, ax = canvas(13.3, 11, 133, 110); ax.set_ylim(-4, 110)
    ax.text(65, 108.5, "Entity–relationship diagram: temporal knowledge store, dual index, query analysis and frozen evaluation", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    entity(ax, 3, 102, 25, "DOCUMENT (CorpusDoc)", ["PK doc_id", "text", "timestamp τ (YYYY-MM-DD)", "title · source", "entity · topic"], "navy")
    entity(ax, 3, 78, 25, "TEMPORAL_EXPRESSION", ["PK expr_id", "FK doc_id", "kind (point / interval)", "start u · end v", "rule (ISO, month-year, season)"], "teal")
    entity(ax, 3, 50, 25, "INDEX_ENTRY", ["PK (index, doc_id)", "FK doc_id", "index: FAISS IndexFlatIP", "   or BM25 (B4 only)", "embedder bge-small-en-v1.5"], "slate")
    entity(ax, 52, 102, 26, "QUERY (QueryItem)", ["PK query_id", "query · split", "needs_retrieval · temporal_class", "time_hint", "gold_answers · gold_doc_ids", "unanswerable"], "navy")
    entity(ax, 52, 72, 26, "ANALYZER_OUTPUT", ["PK (query_id, run_id)", "FK query_id", "needs_retrieval · temporal_class", "confidence γ · necessity_prob", "temporal_probs"], "teal")
    entity(ax, 52, 46, 26, "POLICY_DECISION", ["PK (query_id, run_id)", "policy (current / historical_", "   adaptive_window / no_retrieval)", "target_year · initial/final_years", "max_years · expansions"], "amber")
    entity(ax, 100, 102, 27, "PIPELINE_RESULT", ["PK (system, query_id)", "FK config_name", "answer · did_retrieve", "latency_s · backend", "meta (sufficiency loop)"], "crimson")
    entity(ax, 100, 74, 27, "RETRIEVED_EVIDENCE", ["PK (system, query_id, rank)", "FK doc_id", "rank (≤ k = 5)", "score"], "slate")
    entity(ax, 100, 52, 27, "SYSTEM_CONFIG", ["PK name (checkpoint_b_final)", "top_k 5 · candidate_k 80", "ce_rerank_keep 40 · alpha 0.6", "thresholds 0.55 · rounds 2", "query_timeout_s 60 · enable_*"], "navy")
    entity(ax, 100, 24, 27, "EVAL_SUMMARY", ["PK (dataset, system)", "n · em · token_f1 · contains", "precision_at_k · recall_at_k", "temporal_acc · retrieval_rate", "latency_s · latency_s_median"], "teal")
    entity(ax, 52, 20, 26, "SIGNIFICANCE_TEST", ["PK (dataset, a, b, metric)", "diff_a_minus_b", "ci95_low · ci95_high", "n_a_only · n_b_only", "mcnemar_p (10,000 resamples)"], "crimson")
    rel(ax, (15.5, 85.0), (15.5, 78), "contains", "1", "N")
    rel(ax, (15.5, 60.8), (15.5, 50), "indexed as", "1", "N")
    rel(ax, (65, 83.8), (65, 72), "analysed into", "1", "1..N")
    rel(ax, (65, 56.8), (65, 46), "decides", "1", "1")
    rel(ax, (78, 92), (100, 92), "answered by", "1", "N")
    rel(ax, (113.5, 85.8), (113.5, 74), "cites", "1", "0..5")
    relpath(ax, [(100, 63), (95, 63), (95, 105), (15.5, 105), (15.5, 102)], "refers to", (45, 105), "N", "1")
    relpath(ax, [(127, 95), (129.5, 95), (129.5, 15), (127, 15)], "aggregated into", (129.5, 55), "N", "1")
    rel(ax, (113.5, 35.4), (113.5, 24), "configures", "1", "N")
    rel(ax, (100, 15), (78, 15), "compared in", "2", "N")
    ax.text(3, -2, "Field names follow src/data_schema.py (CorpusDoc, QueryItem), PipelineResult in src/pipelines.py, configs/final.yaml and artifacts/significance.json.", fontsize=8, style="italic", color=SLATE)
    save(fig, "fig04_er_diagram.png")

def clean(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

def fig05():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharey="row")
    for r, (met, mlab) in enumerate([("em", "Exact Match (EM)"), ("contains", "Contains")]):
        for c, ds in enumerate(DS):
            ax = axes[r, c]; vals = [FR[ds][s][met] for s in SYS]; x = np.arange(len(SYS))
            bars = ax.bar(x, vals, color=[SYSCOL[s] for s in SYS], edgecolor="white", width=0.75)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, rotation=90)
            ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticks(x); ax.set_xticklabels(["B1", "B2", "B3", "B4", "Adapt.", "Fast"], fontsize=8.5)
            ax.yaxis.grid(True, color="#E2E8F0", lw=0.8); ax.set_axisbelow(True); clean(ax)
            if r == 0: ax.set_title(DSLAB[ds], fontsize=11, fontweight="bold", color=NAVY)
            if c == 0: ax.set_ylabel(mlab, fontsize=10.5)
    def bracket(ax, i, j, y, text):
        ax.plot([i, i, j, j], [y - 0.03, y, y, y - 0.03], color="#111827", lw=0.9)
        ax.text((i + j) / 2, y + 0.015, text, ha="center", va="bottom", fontsize=7.8)
    s = SIG["heldout"]["adaptive vs B3_fixed_temporal"]
    bracket(axes[0, 1], 2, 4, 1.17, f"vs B3: +{s['em']['diff_a_minus_b']:.3f}, p = {s['em']['mcnemar_p']:.2e}")
    bracket(axes[1, 1], 2, 4, 1.17, f"vs B3: +{s['contains']['diff_a_minus_b']:.3f}, p = {s['contains']['mcnemar_p']:.3f}")
    t = SIG["timeqa"]["adaptive vs tempralm_decay"]["contains"]
    bracket(axes[1, 2], 3, 4, 1.17, f"vs B4: +{t['diff_a_minus_b']:.3f}, p = {t['mcnemar_p']:.2e}")
    fig.legend([Rectangle((0, 0), 1, 1, color=SYSCOL[s]) for s in SYS], [SYSLAB[s] for s in SYS], loc="lower center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Answer accuracy of the six systems on the three frozen test sets (commit 44938a7)", fontsize=12.5, fontweight="bold", color=NAVY, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); save(fig, "fig05_master_accuracy.png")

def fig06():
    sy = ["B2_standard_rag", "B3_fixed_temporal", "tempralm_decay", "adaptive"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
    for ax, (met, lab) in zip(axes, [("precision_at_k", "Precision@5"), ("recall_at_k", "Recall@5")]):
        x = np.arange(len(DS)); w = 0.2
        for i, s in enumerate(sy):
            vals = [FR[d][s][met] for d in DS]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, color=SYSCOL[s], label=SYSLAB[s], edgecolor="white")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([DSLAB[d] for d in DS], fontsize=9.5); ax.set_title(lab, fontsize=11.5, fontweight="bold", color=NAVY)
        ax.set_ylim(0, 1.22); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.yaxis.grid(True, color="#E2E8F0"); ax.set_axisbelow(True); clean(ax)
    dp = round(FR["heldout"]["adaptive"]["precision_at_k"], 3) - round(FR["heldout"]["B3_fixed_temporal"]["precision_at_k"], 3)
    dr = round(FR["timeqa"]["adaptive"]["recall_at_k"], 3) - round(FR["timeqa"]["B3_fixed_temporal"]["recall_at_k"], 3)
    axes[0].annotate(f"+{dp*100:.1f} pts vs B3", xy=(1.3, FR["heldout"]["adaptive"]["precision_at_k"] + 0.09), xytext=(1.15, 1.08), fontsize=8.5, ha="center", color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9))
    axes[1].annotate(f"+{dr*100:.1f} pts vs B3", xy=(2.3, FR["timeqa"]["adaptive"]["recall_at_k"] + 0.09), xytext=(2.0, 0.85), fontsize=8.5, ha="center", color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.9))
    axes[0].set_ylabel("Score"); axes[1].legend(loc="upper right", frameon=False, fontsize=8.8, ncol=2)
    fig.suptitle("Retrieval quality of the retrieving systems (top-5)", fontsize=12.5, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "fig06_retrieval_metrics.png")

def fig07():
    names = {"sentence_selector_off": "Temporal sentence selector", "entity_cut_off": "Entity / surname cut", "window_off": "Adaptive window", "sufficiency_off": "Sufficiency loop"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    def panel(ax, ds, keys, metric, title):
        ys = np.arange(len(keys))[::-1]
        for y, k in zip(ys, keys):
            v = SIG[ds][f"adaptive vs {k}"][metric]; d, lo, hi, p = v["diff_a_minus_b"], v["ci95_low"], v["ci95_high"], v["mcnemar_p"]
            col = NAVY if lo > 0 else SLATE
            ax.barh(y, d * 100, color=col, height=0.55, alpha=0.9)
            ax.errorbar(d * 100, y, xerr=[[max(0, (d - lo) * 100)], [max(0, (hi - d) * 100)]], fmt="none", ecolor="#111827", capsize=4, lw=1)
            ps = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
            ax.text(max(hi * 100, d * 100) + 0.8, y, f"{d*100:+.1f} pts  [{lo*100:+.1f}, {hi*100:+.1f}]  p = {ps}", va="center", fontsize=8.3)
        ax.set_yticks(ys); ax.set_yticklabels([names[k] for k in keys], fontsize=9.5)
        ax.axvline(0, color="#111827", lw=0.8); ax.xaxis.grid(True, color="#E2E8F0"); ax.set_axisbelow(True); clean(ax)
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY); ax.set_xlabel("Contribution of component (Adaptive − ablated), percentage points")
    panel(axes[0], "heldout", ["entity_cut_off"], "contains", "Held-out (n=193), Contains & EM"); axes[0].set_xlim(-2, 26); axes[0].set_ylim(-0.8, 0.8)
    panel(axes[1], "timeqa", ["sentence_selector_off", "entity_cut_off", "window_off", "sufficiency_off"], "contains", "TimeQA (n=500), Contains"); axes[1].set_xlim(-3, 36)
    fig.suptitle("Component ablations with paired bootstrap 95% CIs and exact McNemar p-values", fontsize=12.5, fontweight="bold", color=NAVY)
    fig.text(0.5, -0.02, "Positive values mean removing the component hurts. Router-off is omitted: it changes no answer outside the templated trivia skips. Navy = CI excludes zero; slate = CI includes zero.", ha="center", fontsize=8.5, style="italic", color=SLATE)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94)); save(fig, "fig07_ablation_impact.png")

def fig08():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]; sy = ["B2_standard_rag", "B3_fixed_temporal", "adaptive", "adaptive_fast"]; x = np.arange(len(DS)); w = 0.2
    for i, s in enumerate(sy):
        mean = [FR[d][s]["latency_s"] for d in DS]; med = [FR[d][s]["latency_s_median"] for d in DS]
        bars = ax.bar(x + (i - 1.5) * w, mean, w, color=SYSCOL[s], label=SYSLAB[s] + " (mean)", edgecolor="white")
        ax.scatter(x + (i - 1.5) * w, med, marker="D", s=22, color="white", edgecolor="#111827", zorder=3, label="median" if i == 0 else None)
        for b, v in zip(bars, mean): ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=7.3, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([DSLAB[d] for d in DS]); ax.set_ylabel("Seconds per query (CPU)"); ax.set_ylim(0, 11.5)
    ax.yaxis.grid(True, color="#E2E8F0"); ax.set_axisbelow(True); clean(ax); ax.legend(frameon=False, fontsize=8.3, loc="upper left", ncol=2)
    ax.set_title("(a) Mean (bars) and median (white diamonds) latency", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(0.99, -0.16, "Adaptive-Fast latencies come from the later Adaptive-only ablation run (not a matched four-system run).", transform=ax.transAxes, ha="right", fontsize=7.8, style="italic", color=SLATE)
    ax = axes[1]; b, a = F["hang_fix"]["janet_frame_before_s"], F["hang_fix"]["janet_frame_after_s"]
    ax.bar([0, 1], [b, a], color=[CRIMSON, TEAL], width=0.55); ax.set_yscale("log"); ax.set_ylim(1, 2e5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Before fix\n(unbounded loop)", "After fix\n(≤3 retrieves, 60 s, dateutil cap)"], fontsize=8.8)
    ax.text(0, b * 1.4, f"{b:,.1f} s", ha="center", fontsize=9.5, fontweight="bold"); ax.text(1, a * 1.4, f"{a:.3f} s", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(0.5, 1000, f"≈ {b/a:,.0f}× faster", ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_ylabel("Wall-clock seconds (log scale)"); ax.yaxis.grid(True, which="major", color="#E2E8F0"); ax.set_axisbelow(True); clean(ax)
    ax.set_title("(b) Hang probe: \u201cJanet Frame received which award in 1990?\u201d", fontsize=10.5, fontweight="bold", color=NAVY)
    fig.tight_layout(); save(fig, "fig08_latency_and_hangfix.png")

def fig09():
    R = F["router_large_100k"]; FU = F["router_large_funnel"]
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.05], hspace=0.22, wspace=0.32)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 150); ax.set_ylim(2, 25); ax.axis("off")
    steps = [("Load 10 real QA sources", "nq_open · squad_v2 · triviaqa\ntimeqa · situatedqa · sciq\ncommonsense_qa · qasc · arc\nopenbookqa", "slate"),
             ("Rule labelling\n(ref. year 2026)", f"{FU['labeled_total']:,} labelled", "navy"),
             ("De-duplicate", f"{FU['after_dedup']:,} kept\n({FU['dedup_dropped']:,} dropped)", "teal"),
             ("Leak filter\n6,117 Q · 226 ent. · 335 tpl.", f"{FU['after_leak_filter']:,} kept", "crimson"),
             ("Class balance\n(all current kept)", f"{R['total']:,} rows\nsynthetic {R['synthetic_rows']} · leaks 0", "amber"),
             ("DeBERTa-v3-base fine-tune\nKaggle T4 · fp32", "5 epochs · lr 2e-5 · bs 16\nckpt / 500 steps (in progress)", "green")]
    for i, (t, sub, c) in enumerate(steps):
        x = 1 + i * 25; box(ax, x, 8, 22, 15, f"{t}\n\n{sub}", c, fs=8.6)
        if i < len(steps) - 1: arrow(ax, (x + 22, 15.5), (x + 25, 15.5))
    ax.text(75, 4, "Leak-free router corpus pipeline (scripts/build_large_router_dataset.py → data/router_large/stats.json → Kaggle notebook atrag-router-large-100k-training)", ha="center", fontsize=9, style="italic", color=SLATE)
    ax.set_title("(a) Construction and training pipeline", fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax1 = fig.add_subplot(gs[1, 0]); sp = ["train", "val", "test"]; cls = ["current", "historical", "timeless"]; cc = [CRIMSON, NAVY, TEAL]; bottom = np.zeros(3)
    for c, col in zip(cls, cc):
        v = np.array([R["by_split_and_class"][s][c] for s in sp]); ax1.bar(sp, v, bottom=bottom, color=col, label=c, width=0.6); bottom += v
    for i, s in enumerate(sp): ax1.text(i, bottom[i] + 1200, f"{R['by_split'][s]:,}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 92000); ax1.set_ylabel("Rows"); ax1.legend(frameon=False, fontsize=8.5); ax1.set_title("(b) Split × temporal class", fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v/1000)}k")); clean(ax1)
    ax2 = fig.add_subplot(gs[1, 1]); vals = [R["by_temporal_class"][c] for c in cls]
    wedges, _ = ax2.pie(vals, colors=cc, startangle=90, wedgeprops=dict(width=0.38, edgecolor="white"))
    ax2.text(0, 0, f"{R['total']:,}\nrows", ha="center", va="center", fontsize=10, fontweight="bold")
    ax2.legend(wedges, [f"{c}: {v:,}" for c, v in zip(cls, vals)], loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8.6)
    ax2.set_title("(c) Temporal class balance", fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax3 = fig.add_subplot(gs[1, 2]); src = sorted(R["by_source"].items(), key=lambda kv: kv[1])
    ax3.barh([k for k, _ in src], [v for _, v in src], color=SLATE)
    for i, (_, v) in enumerate(src): ax3.text(v + 300, i, f"{v:,}", va="center", fontsize=8)
    ax3.set_xlim(0, 33000); ax3.set_title("(d) Rows by source", fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax3.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v/1000)}k")); clean(ax3); ax3.tick_params(axis="y", labelsize=8.5)
    fig.suptitle("100,000-row leak-free learned-router corpus and resumable training pipeline", fontsize=13, fontweight="bold", color=NAVY, y=0.995)
    save(fig, "fig09_learned_router_100k.png")

if __name__ == "__main__":
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09): f()
