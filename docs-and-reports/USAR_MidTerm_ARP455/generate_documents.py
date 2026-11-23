"""Generate USAR Mid-Term Progress Report (DOCX) and Presentation (PPTX)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from pptx import Presentation
from pptx.dml.color import RGBColor as PptRGB
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches as PptInches, Pt as PptPt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def set_run_font(run, name="Times New Roman", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_para(doc, text, *, size=12, bold=False, italic=False, align="justify", space_after=8, space_before=0, first_line=True):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    elif align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line and align == "justify":
        pf.first_line_indent = Cm(0.75)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, size=14, bold=True, color=RGBColor(0x1A, 0x36, 0x5D))
    else:
        set_run_font(run, size=12, bold=True, color=RGBColor(0x2B, 0x4C, 0x7E))
    return p


def add_bullets(doc, items, size=12):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p.clear()
        run = p.add_run(item)
        set_run_font(run, size=size)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run_font(run, size=10, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        shading = hdr[i]._tePr if False else None
        from docx.oxml import OxmlElement

        tc = hdr[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1A365D")
        shd.set(qn("w:val"), "clear")
        tcPr.append(shd)
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = ""
            p = cells[c_idx].paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=10)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = w
    doc.add_paragraph()
    return table


def add_figure(doc, path: Path, caption: str, width_inches=6.2):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width_inches))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    cap.paragraph_format.first_line_indent = Cm(0)
    run = cap.add_run(caption)
    set_run_font(run, size=10, italic=True)


def build_progress_report() -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    # ---- Cover ----
    for line, size, bold in [
        ("Guru Gobind Singh Indraprastha University", 14, True),
        ("University School of Automation & Robotics (USAR)", 12, True),
        ("East Delhi Campus, Surajmal Vihar, Delhi", 11, False),
    ]:
        add_para(doc, line, size=size, bold=bold, align="center", first_line=False, space_after=2)

    add_para(doc, "", first_line=False, space_after=6)
    add_para(
        doc,
        "1st Mid-Term Evaluation — Minor Project (ARP-455)",
        size=12,
        bold=True,
        align="center",
        first_line=False,
        space_after=4,
    )
    add_para(doc, "Progress Report", size=16, bold=True, align="center", first_line=False, space_after=10)

    add_para(
        doc,
        "When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering",
        size=13,
        bold=True,
        align="center",
        first_line=False,
        space_after=6,
    )
    add_para(
        doc,
        "(Adaptive Temporal RAG)",
        size=12,
        italic=True,
        align="center",
        first_line=False,
        space_after=14,
    )

    add_para(doc, "Submitted By:", size=11, bold=True, align="center", first_line=False, space_after=2)
    add_para(doc, "Rishi Kumar", size=12, bold=True, align="center", first_line=False, space_after=1)
    add_para(doc, "B.Tech. (AI & ML), Batch B2", size=11, align="center", first_line=False, space_after=1)
    add_para(doc, "Enrolment No.: 61219051624", size=11, align="center", first_line=False, space_after=12)

    add_para(doc, "Under the Guidance of", size=11, italic=True, align="center", first_line=False, space_after=2)
    add_para(doc, "Mr. Anuj Kumar", size=12, bold=True, align="center", first_line=False, space_after=1)
    add_para(doc, "Assistant Professor, USAR", size=11, align="center", first_line=False, space_after=16)

    add_para(doc, "Academic Session: 2026–27  |  Evaluation Date: 8 October 2026", size=10, align="center", first_line=False, space_after=20)

    # Supervisor sign-off block
    add_para(doc, "Supervisor Certification", size=12, bold=True, align="center", first_line=False, space_after=6)
    add_para(
        doc,
        "This is to certify that the progress reported herein for the Minor Project (ARP-455) titled "
        "“Adaptive Temporal RAG” has been carried out under my supervision and reflects the work "
        "completed till the 1st Mid-Term Evaluation.",
        size=11,
        align="justify",
        first_line=False,
        space_after=16,
    )
    add_para(doc, "Signature of Project Supervisor: ____________________", size=11, align="left", first_line=False, space_after=4)
    add_para(doc, "Name: Mr. Anuj Kumar", size=11, align="left", first_line=False, space_after=2)
    add_para(doc, "Date: ____________________", size=11, align="left", first_line=False, space_after=8)

    doc.add_page_break()

    # ---- TOC ----
    add_heading_custom(doc, "Table of Contents", level=1)
    toc_items = [
        "1. Title of the Project",
        "2. Introduction",
        "3. Literature Survey",
        "4. Problem Statement",
        "5. Motivation",
        "6. Proposed Design Flow (Architectures & Diagrams)",
        "7. Proposed Methodology",
        "8. Results (Progress to Date)",
        "9. GANTT / PERT Chart",
        "10. Hardware and Software Requirements",
        "11. References",
    ]
    for t in toc_items:
        add_para(doc, t, align="left", first_line=False, space_after=4)

    doc.add_page_break()

    # ---- 1 Title ----
    add_heading_custom(doc, "1. Title of the Project", level=1)
    add_para(
        doc,
        "When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering "
        "(Adaptive Temporal RAG).",
        first_line=False,
    )
    add_para(
        doc,
        "Short title used in implementation and demo: Adaptive Temporal RAG — a framework that decides whether retrieval "
        "is needed and which temporal ranking policy to apply for time-sensitive question answering.",
        first_line=False,
    )

    # ---- 2 Introduction ----
    add_heading_custom(doc, "2. Introduction", level=1)
    add_para(
        doc,
        "Large Language Models (LLMs) perform well on many language tasks, but their parameters are frozen after training. "
        "They cannot natively track facts that change after the training cut-off and may produce outdated or fabricated "
        "answers for time-sensitive questions.",
    )
    add_para(
        doc,
        "Retrieval-Augmented Generation (RAG) mitigates this by fetching external documents at query time and supplying "
        "them as context to the generator. Conventional RAG ranks passages almost exclusively by semantic similarity. "
        "That is insufficient when the correct evidence depends on time: “current” facts need recent documents; "
        "historical questions need evidence near a target year; timeless questions may need no retrieval at all.",
    )
    add_para(
        doc,
        "This minor project designs, implements, and evaluates Adaptive Temporal RAG: a DistilBERT multitask query "
        "analyzer predicts retrieval necessity, temporal class (current / historical / timeless), and confidence; a "
        "policy module then selects no-retrieval, recency ranking, historical windowing, or semantic-only fallback. "
        "The system is compared against LLM-only, Standard RAG, and Fixed Temporal RAG baselines on a curated dated corpus.",
    )

    # ---- 3 Literature ----
    add_heading_custom(doc, "3. Literature Survey", level=1)
    add_heading_custom(doc, "3.1 Foundational Retrieval-Augmented Generation", level=2)
    add_para(
        doc,
        "Lewis et al. (2020) introduced RAG by combining dense retrieval with sequence-to-sequence generation, improving "
        "factual accuracy on knowledge-intensive tasks. Dense Passage Retrieval (Karpukhin et al., 2020) showed that "
        "learned embeddings often outperform sparse lexical retrieval. Fusion-in-Decoder (Izacard & Grave, 2021) improved "
        "how multiple passages are fused during generation. These systems form the semantic-similarity baseline of this "
        "project (Standard RAG).",
    )
    add_heading_custom(doc, "3.2 Temporal and Freshness-Aware Retrieval", level=2)
    add_para(
        doc,
        "FreshLLM / FreshQA-style evaluations (Vu et al., 2023) show that LLMs and standard RAG frequently fail on "
        "fast-changing facts unless retrieval favours recent, high-quality sources. Temporal RAG variants incorporate "
        "document timestamps into filtering or re-ranking relative to the time referenced by the query. Such methods "
        "motivate the Fixed Temporal RAG baseline in this work, but typically apply one temporal rule uniformly to every query.",
    )
    add_heading_custom(doc, "3.3 Adaptive / Selective Retrieval", level=2)
    add_para(
        doc,
        "Self-RAG (Asai et al., 2023) learns when to retrieve and how to critique evidence. Adaptive-RAG (Jeong et al., 2024) "
        "routes queries by complexity to no-retrieval, single-step, or multi-step pipelines. FLARE (Jiang et al., 2023) "
        "triggers retrieval when generation confidence drops. These methods address whether to retrieve, but not jointly "
        "with temporal intent (present vs past vs timeless).",
    )
    add_heading_custom(doc, "3.4 Research Gap", level=2)
    add_para(
        doc,
        "Prior adaptive systems decide if retrieval is needed; prior temporal systems decide how to weight time after "
        "retrieval is already triggered. Few systems jointly estimate retrieval necessity, temporal category, and "
        "decision confidence in one query-conditioned policy. Adaptive Temporal RAG targets exactly this gap.",
    )

    # ---- 4 Problem ----
    add_heading_custom(doc, "4. Problem Statement", level=1)
    add_para(
        doc,
        "Conventional RAG retrieves primarily by semantic similarity and assumes similar retrieval behaviour for all "
        "queries. For time-sensitive information needs this causes: (i) outdated answers when older but similar documents "
        "outrank recent ones; (ii) wrong historical answers when the most recent similar document describes a different "
        "period; (iii) unnecessary retrieval for timeless/trivial queries; and (iv) avoidable latency and cost.",
    )
    add_para(
        doc,
        "Problem addressed: How can a RAG system automatically detect whether retrieval is necessary and what temporal "
        "intent a query carries, then select a query-specific retrieval and ranking strategy that improves temporal "
        "correctness and reduces unnecessary retrieval without degrading answer quality?",
        first_line=False,
    )

    # ---- 5 Motivation ----
    add_heading_custom(doc, "5. Motivation", level=1)
    add_para(
        doc,
        "Practical motivation. Assistants are routinely asked current-event and office-holder questions; silently wrong "
        "answers erode trust. Retrieval is not free—skipping it when unnecessary reduces latency and cost. Enterprise "
        "corpora (news, filings, policies) are timestamped, yet metadata is rarely used beyond crude filters.",
    )
    add_para(
        doc,
        "Research motivation. Temporal reasoning remains a known weakness of LLMs and RAG stacks. Combining retrieval-"
        "necessity detection with temporal-intent detection under a confidence-aware policy is a focused, evaluable "
        "contribution suitable for a semester minor project, with clear baselines and metrics.",
    )

    # ---- 6 Design Flow / Architectures ----
    add_heading_custom(doc, "6. Proposed Design Flow", level=1)
    add_para(
        doc,
        "This section presents the design flow through two architectures: (A) Temporal RAG — the fixed temporal baseline "
        "used for comparison; and (B) Adaptive Temporal RAG — the proposed system. Both share the same dated corpus, "
        "MiniLM embeddings, FAISS index, and generator interface so differences are attributable to policy design.",
        first_line=False,
    )

    add_heading_custom(doc, "6.1 Architecture A — Temporal RAG (Fixed Temporal Baseline)", level=2)
    add_para(
        doc,
        "Temporal RAG always retrieves. The query is embedded with Sentence-Transformers MiniLM; FAISS returns a wide "
        "candidate set. A fixed rule then applies: if a year is detected in the query (or time hint), candidates are "
        "filtered to a ±2-year window and re-ranked for historical proximity; otherwise candidates receive a recency "
        "boost. Final ranking uses α·semantic + (1−α)·temporal (default α = 0.6). Top-k passages are passed to the generator. "
        "There is no query analyzer and no option to skip retrieval.",
    )
    add_figure(
        doc,
        FIG / "architecture_temporal_rag.jpg",
        "Figure 1. Temporal RAG architecture (always retrieve + fixed temporal policy).",
    )
    add_para(doc, "Design flow summary (Temporal RAG):", first_line=False, bold=True, space_after=4)
    # bold=True not in add_para signature properly - I used wrong kw. Fix by bullets only.
    add_bullets(
        doc,
        [
            "Input query → MiniLM embedding → FAISS candidate retrieval (always).",
            "If year present → historical window filter (±2 years) → historical temporal re-rank.",
            "Else → recency-based temporal re-rank.",
            "Score fusion → top-k evidence → LLM / extractive generator → answer.",
        ],
    )

    add_heading_custom(doc, "6.2 Architecture B — Adaptive Temporal RAG (Proposed System)", level=2)
    add_para(
        doc,
        "Adaptive Temporal RAG inserts a DistilBERT multitask query analyzer before retrieval. The analyzer predicts: "
        "(1) retrieval necessity (0/1); (2) temporal class — current, historical, or timeless; (3) confidence. A policy "
        "module then selects: no retrieval (parametric answer); current → recency re-rank; historical → year-window filter "
        "then historical re-rank; timeless or low confidence → safer semantic-only ranking. When retrieving, MiniLM + FAISS "
        "produce candidates (≈50), which are re-ranked to top-k (default 5) before generation.",
    )
    add_figure(
        doc,
        FIG / "architecture_adaptive_temporal_rag.jpg",
        "Figure 2. Adaptive Temporal RAG architecture (conditional retrieval + class-specific policy).",
    )
    add_para(doc, "Design flow summary (Adaptive Temporal RAG):", first_line=False, space_after=4)
    add_bullets(
        doc,
        [
            "Query → DistilBERT analyzer → necessity, temporal class, confidence.",
            "If needs_retrieval = 0 → generate without documents.",
            "If confidence < 0.55 → FAISS candidates → semantic-only re-rank → generate.",
            "If class = current → candidates → recency re-rank → generate.",
            "If class = historical → year window (±2) → historical re-rank → generate.",
            "If class = timeless → semantic-only re-rank → generate.",
        ],
    )

    add_heading_custom(doc, "6.3 Comparative Design View", level=2)
    add_table(
        doc,
        ["Aspect", "Temporal RAG", "Adaptive Temporal RAG"],
        [
            ["Retrieval", "Always", "Conditional (analyzer)"],
            ["Temporal policy", "Fixed rule (year / recency)", "Class-specific + confidence fallback"],
            ["Query analyzer", "No", "Yes (DistilBERT multitask)"],
            ["Skip retrieval", "No", "Yes (necessity = 0)"],
            ["Ranking formula", "α·sem + (1−α)·temp", "Same when temporal modes apply"],
            ["Role in project", "Baseline B3", "Proposed system"],
        ],
    )

    add_heading_custom(doc, "6.4 High-Level Data Flow", level=2)
    add_para(
        doc,
        "Data stores: (i) corpus.jsonl — dated passages with doc_id, text, timestamp, entity/topic metadata; "
        "(ii) queries.jsonl — labelled queries with needs_retrieval, temporal_class, time_hint, gold answers/docs, and "
        "train/val/test splits; (iii) artifacts/ — FAISS index, embeddings, trained analyzer weights, evaluation outputs. "
        "Online path: query → (optional) analyzer → retrieve/rank → generate → answer + evidence. Offline path: build "
        "dataset → build index → train analyzer → evaluate four systems → Streamlit demo.",
        first_line=False,
    )

    # ---- 7 Methodology ----
    add_heading_custom(doc, "7. Proposed Methodology", level=1)
    add_heading_custom(doc, "7.1 Dataset", level=2)
    add_para(
        doc,
        "A synthetic educational corpus of approximately 800 short dated passages covers politics, business leadership, "
        "sports, awards, laws, and timeless encyclopedia facts. High-churn entities appear at multiple timestamps so that "
        "historical vs current ranking differs. About 1050 labelled queries include needs_retrieval, temporal_class, "
        "optional time_hint, gold_answer, and gold_doc_ids, split 70/15/15 with train paraphrases. Approximate balance: "
        "~35% current / ~30% historical / ~30% timeless; ~25% no-retrieval.",
    )
    add_heading_custom(doc, "7.2 Indexing and Baselines", level=2)
    add_para(
        doc,
        "Documents are embedded with sentence-transformers/all-MiniLM-L6-v2 and indexed in FAISS IndexFlatIP. Four systems "
        "share the same index and generator interface: B1 LLM-only (never retrieve); B2 Standard RAG (always semantic "
        "top-k); B3 Fixed Temporal RAG (architecture in §6.1); Proposed Adaptive Temporal RAG (architecture in §6.2).",
    )
    add_heading_custom(doc, "7.3 Query Analyzer Training", level=2)
    add_para(
        doc,
        "A DistilBERT encoder with three heads (necessity, temporal class, confidence) is fine-tuned on the labelled "
        "train split. Loss combines cross-entropy for necessity and temporal class with a confidence term. Best validation "
        "checkpoint (necessity F1 and temporal accuracy) is saved to artifacts for inference in the adaptive pipeline and demo.",
    )
    add_heading_custom(doc, "7.4 Temporal Scoring", level=2)
    add_para(
        doc,
        "For current mode, newer documents receive higher temporal affinity (age mapped over ~10 years). For historical "
        "mode, affinity peaks near the query year. Semantic and temporal scores are fused with α = 0.6. A fixed reference "
        "date (2025-10-01) keeps evaluation reproducible.",
    )
    add_heading_custom(doc, "7.5 Generation and Evaluation", level=2)
    add_para(
        doc,
        "The generator facade supports extractive stub (fast offline eval), local HF LLMs, or API LLMs. Metrics include "
        "answer contains / EM / token-F1, temporal accuracy (top-1 evidence date heuristic), precision@k / recall@k, "
        "latency, and retrieval rate. A Streamlit app exposes system toggle, α, and confidence threshold for demonstration.",
    )

    # ---- 8 Results ----
    add_heading_custom(doc, "8. Results (if any)", level=1)
    add_para(
        doc,
        "Progress to the mid-term evaluation includes a complete implementation of dataset construction, FAISS indexing, "
        "all four pipelines, DistilBERT analyzer training path, offline evaluation scripts, result charts, and a Streamlit "
        "demo. Measured results on the held-out test split (75 queries) with the extractive generator backend are summarized below.",
        first_line=False,
    )
    add_table(
        doc,
        ["System", "Contains", "Temporal Acc.", "Retrieval Rate", "Latency (s)"],
        [
            ["B1 LLM-only", "0.027", "0.307", "0.00", "~0.00"],
            ["B2 Standard RAG", "0.427", "0.573", "1.00", "~0.022"],
            ["B3 Temporal RAG (Fixed)", "0.773", "0.867", "1.00", "~0.020"],
            ["Adaptive Temporal RAG", "0.600", "0.840", "0.68", "~0.098"],
        ],
    )
    add_para(
        doc,
        "Observations. Adaptive Temporal RAG approaches Fixed Temporal RAG on temporal accuracy while retrieving for only "
        "~68% of queries (skipping timeless / no-retrieval items). Under the extractive stub, answer contains trails B3 "
        "because skipped-retrieval definition-style questions have no passages to extract from; stronger HF/API generators "
        "are planned for fairer answer-quality comparison. Both temporal systems clearly outperform Standard RAG on "
        "temporal accuracy. Work remaining: stronger generator experiments, deeper error analysis, and final report polish.",
        first_line=False,
    )
    add_figure(
        doc,
        FIG / "results_charts.png",
        "Figure 3. Comparative evaluation charts across B1, B2, B3, and Adaptive Temporal RAG.",
        width_inches=5.8,
    )

    # ---- 9 Gantt ----
    add_heading_custom(doc, "9. GANTT / PERT Chart", level=1)
    add_para(
        doc,
        "The project follows a twelve-week plan from literature review through evaluation and reporting. Mid-term "
        "evaluation (Week of 8 Oct 2026) corresponds to completion of baselines, Temporal RAG, Adaptive integration, "
        "initial experiments, and demo readiness.",
        first_line=False,
    )
    add_table(
        doc,
        ["Phase", "Weeks", "Status (Mid-Term)", "Key Output"],
        [
            ["1. Literature & problem definition", "W1", "Completed", "Problem statement, research gap"],
            ["2. Dataset preparation", "W2–W3", "Completed", "corpus.jsonl, queries.jsonl"],
            ["3. Baselines (B1, B2)", "W4", "Completed", "LLM-only & Standard RAG"],
            ["4. Temporal RAG (Fixed / B3)", "W5", "Completed", "Year window + recency ranking"],
            ["5. Adaptive query analyzer", "W6–W7", "Completed", "DistilBERT multitask model"],
            ["6. Adaptive system integration", "W8", "Completed", "End-to-end Adaptive pipeline"],
            ["7. Experiments & evaluation", "W9–W10", "In progress", "Metrics, charts, error analysis"],
            ["8. Demo, report & viva prep", "W11–W12", "Ongoing", "Streamlit demo, final report"],
        ],
    )
    add_para(doc, "Simplified Gantt (active weeks marked with ■):", first_line=False, space_after=4)
    add_table(
        doc,
        ["Phase", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12"],
        [
            ["Literature", "■", "", "", "", "", "", "", "", "", "", "", ""],
            ["Dataset", "", "■", "■", "", "", "", "", "", "", "", "", ""],
            ["Baselines B1/B2", "", "", "", "■", "", "", "", "", "", "", "", ""],
            ["Temporal RAG", "", "", "", "", "■", "", "", "", "", "", "", ""],
            ["Analyzer", "", "", "", "", "", "■", "■", "", "", "", "", ""],
            ["Adaptive integrate", "", "", "", "", "", "", "", "■", "", "", "", ""],
            ["Experiments", "", "", "", "", "", "", "", "", "■", "■", "", ""],
            ["Report / Demo", "", "", "", "", "", "", "", "", "", "", "■", "■"],
        ],
    )

    # ---- 10 H/W S/W ----
    add_heading_custom(doc, "10. Hardware and Software Requirements", level=1)
    add_heading_custom(doc, "10.1 Software Requirements", level=2)
    add_table(
        doc,
        ["Component", "Choice used / planned"],
        [
            ["Language", "Python 3.10+"],
            ["Embeddings", "sentence-transformers/all-MiniLM-L6-v2"],
            ["Vector index", "FAISS (IndexFlatIP)"],
            ["Query analyzer", "DistilBERT (distilbert-base-uncased), PyTorch"],
            ["Generators", "Extractive stub; optional HF (e.g. Phi-3) / API LLM"],
            ["Demo UI", "Streamlit"],
            ["Experiment logging", "Pandas / CSV, Matplotlib charts"],
            ["Notebooks", "Google Colab (GPU T4) for longer training"],
            ["Version control", "Git"],
            ["OS", "Windows / Linux / macOS"],
        ],
    )
    add_heading_custom(doc, "10.2 Hardware Requirements", level=2)
    add_table(
        doc,
        ["Resource", "Minimum", "Recommended"],
        [
            ["RAM", "8 GB", "16 GB+"],
            ["CPU", "Quad-core (i5 / Ryzen 5)", "Higher clock / more cores"],
            ["GPU", "Not required for extractive / API path", "8 GB+ VRAM (Colab T4 / RTX) for local LLM & faster training"],
            ["Storage", "10 GB free", "25 GB+ (models, embeddings, logs)"],
            ["Internet", "Required for model downloads / API", "Stable broadband"],
        ],
    )

    # ---- 11 References ----
    add_heading_custom(doc, "11. References", level=1)
    refs = [
        "Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.",
        "Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.",
        "Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain QA. EACL.",
        "Vu, T., et al. (2023). FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.",
        "Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.",
        "Jeong, S., et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.",
        "Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE).",
        "Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.",
        "Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE TBD (FAISS).",
    ]
    for i, r in enumerate(refs, 1):
        add_para(doc, f"[{i}] {r}", align="left", first_line=False, space_after=4)

    add_para(doc, "", first_line=False, space_after=20)
    add_para(
        doc,
        "— End of Progress Report —",
        size=11,
        italic=True,
        align="center",
        first_line=False,
    )

    out = ROOT / "ARP455_Progress_Report_Adaptive_Temporal_RAG.docx"
    doc.save(out)
    return out


def _slide_title(slide, text):
    if slide.shapes.title:
        slide.shapes.title.text = text
        for p in slide.shapes.title.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = PptPt(28)
                r.font.color.rgb = PptRGB(0x1A, 0x36, 0x5D)


def _add_bullets(tf, items, size=18):
    tf.clear()
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.size = PptPt(size)
        p.font.color.rgb = PptRGB(0x1A, 0x1A, 0x1A)


def build_presentation() -> Path:
    prs = Presentation()
    prs.slide_width = PptInches(13.333)
    prs.slide_height = PptInches(7.5)

    # 1 Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Adaptive Temporal RAG"
    sub = slide.placeholders[1]
    sub.text = (
        "When and What to Retrieve for Time-Aware QA\n"
        "Minor Project (ARP-455) — 1st Mid-Term Evaluation\n\n"
        "Rishi Kumar  |  61219051624  |  B.Tech AI&ML (B2)\n"
        "Guide: Mr. Anuj Kumar, Assistant Professor, USAR\n"
        "GGSIPU — University School of Automation & Robotics\n"
        "8 October 2026"
    )

    # 2 Introduction / Motivation
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "1. Introduction & Motivation")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "LLMs are frozen after training → outdated / hallucinated answers for changing facts",
            "Standard RAG retrieves by semantic similarity only — ignores time",
            "Need: decide WHEN to retrieve and WHAT temporal policy to use",
            "Motivation: trust for current facts, correct historical evidence, lower cost when retrieval is unnecessary",
        ],
    )

    # 3 Problem & objectives
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "2. Problem Statement & Objectives")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "Problem: one-size-fits-all RAG fails for current / historical / timeless queries",
            "Objective 1: Detect retrieval necessity + temporal class + confidence",
            "Objective 2: Apply class-specific ranking (recency / year-window / semantic)",
            "Objective 3: Compare vs LLM-only, Standard RAG, Fixed Temporal RAG",
            "Objective 4: Demo + measurable mid-term results on a dated corpus",
        ],
    )

    # 4 Literature gap
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "3. Literature Gap")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "RAG / DPR / FiD — strong semantic retrieval baselines",
            "FreshLLM & temporal RAG — improve freshness, but often fixed policy for all queries",
            "Self-RAG / Adaptive-RAG / FLARE — decide whether to retrieve, not temporal intent",
            "Gap: jointly model necessity + temporal class + confidence in one policy",
        ],
    )

    # 5 Temporal RAG architecture
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    _slide_title(slide, "4. Architecture A — Temporal RAG (Baseline)")
    pic_path = FIG / "architecture_temporal_rag.jpg"
    if pic_path.exists():
        slide.shapes.add_picture(str(pic_path), PptInches(0.6), PptInches(1.4), width=PptInches(12.1))

    # 6 Adaptive architecture
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _slide_title(slide, "5. Architecture B — Adaptive Temporal RAG (Proposed)")
    pic_path = FIG / "architecture_adaptive_temporal_rag.jpg"
    if pic_path.exists():
        slide.shapes.add_picture(str(pic_path), PptInches(0.6), PptInches(1.4), width=PptInches(12.1))

    # 7 Methodology / stack
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "6. Methodology & Tech Stack")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "Dataset: ~800 dated passages, ~1050 labelled queries (70/15/15)",
            "Index: MiniLM embeddings + FAISS IndexFlatIP",
            "Analyzer: DistilBERT multitask (necessity / temporal / confidence)",
            "Ranking: final = α·semantic + (1−α)·temporal (α = 0.6)",
            "Systems: B1 LLM-only · B2 Standard RAG · B3 Temporal RAG · Adaptive",
            "Demo: Streamlit  |  Training path: local / Colab GPU",
        ],
    )

    # 8 Results
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "7. Results to Date (Test Split, n=75)")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "B1 LLM-only — Contains 0.03 | Temporal Acc. 0.31 | Retrieve 0%",
            "B2 Standard RAG — Contains 0.43 | Temporal Acc. 0.57 | Retrieve 100%",
            "B3 Temporal RAG — Contains 0.77 | Temporal Acc. 0.87 | Retrieve 100%",
            "Adaptive — Contains 0.60 | Temporal Acc. 0.84 | Retrieve 68%",
            "Key: Adaptive ≈ B3 on temporal accuracy while retrieving less often",
        ],
        size=17,
    )

    # 9 Progress / Gantt
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "8. Progress vs Plan (Mid-Term)")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "Completed: literature, dataset, B1–B3, Adaptive pipeline, analyzer, Streamlit demo",
            "Completed: offline evaluation scripts + result charts",
            "In progress: stronger generator experiments, deeper error analysis",
            "Upcoming (W11–W12): final report polish, viva prep, optional FastAPI deployment",
            "Supervisor sign-off included in Progress Report (ARP-455 requirement)",
        ],
    )

    # 10 Conclusion / next steps
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _slide_title(slide, "9. Conclusion & Next Steps")
    _add_bullets(
        slide.placeholders[1].text_frame,
        [
            "Delivered a working Adaptive Temporal RAG with clear Temporal RAG baseline",
            "Shows that joint necessity + temporal policy improves time-aware retrieval behaviour",
            "Next: HF/API generators, learned temporal scorers, larger FreshQA-style data",
            "Thank you — Questions welcome",
        ],
    )

    out = ROOT / "ARP455_MidTerm_Presentation_Adaptive_Temporal_RAG.pptx"
    prs.save(out)
    return out


def main():
    report = build_progress_report()
    ppt = build_presentation()
    print("REPORT:", report)
    print("PPT:", ppt)


if __name__ == "__main__":
    main()
