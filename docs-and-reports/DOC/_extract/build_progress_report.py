"""Rebuild ARP-455 mid-term progress report: denser content, one section per page."""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = Path(r"c:\Users\Assessment\Documents\cursor\DOC\ARP455_Progress_Report_Adaptive_Temporal_RAG.docx")
BACKUP = Path(r"c:\Users\Assessment\Documents\cursor\DOC\ARP455_Progress_Report_Adaptive_Temporal_RAG_BACKUP.docx")
MEDIA = Path(r"c:\Users\Assessment\Documents\cursor\DOC\_extract\docx_unzip\word\media")
LOGO = Path(r"c:\Users\Assessment\Documents\cursor\DOC\_extract\ggsipu_logo.jpg")


def set_run_font(run, size=11, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def add_para(doc, text, *, size=11, bold=False, italic=False, align="left", space_after=6, space_before=0):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = 1.15
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_num(doc, text, level=1):
    size = 14 if level == 1 else 12
    return add_para(doc, text, size=size, bold=True, space_before=10, space_after=8)


def add_section_start(doc, text):
    """Major numbered section heading that always begins on a new page."""
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    set_run_font(run, size=14, bold=True)
    return p


def add_bullets(doc, items, *, size=11):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.clear()
        run = p.add_run(item)
        set_run_font(run, size=size)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15


def set_cell_text(cell, text, *, bold=False, size=10):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)


def shade_header_row(row):
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "D9E2F3")
        shd.set(qn("w:val"), "clear")
        tcPr.append(shd)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for j, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[j], h, bold=True, size=10)
    shade_header_row(table.rows[0])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell_text(table.rows[i + 1].cells[j], str(val), size=10)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)
    doc.add_paragraph()
    return table


def try_add_image(doc, path: Path, width=5.8, caption=None):
    if path and path.exists():
        doc.add_picture(str(path), width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            add_para(doc, caption, size=10, italic=True, align="center", space_after=10)
        return True
    return False


def main():
    if OUT.exists():
        # Always keep one rolling backup of whatever is currently on disk.
        shutil.copy2(OUT, BACKUP)

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # ========================= COVER (Arsh-style: logo then centered title block) =========================
    # Match Arsh first page: university logo centered at top, then ARP line, title, students, supervisor, school block.
    add_para(doc, "", space_after=6)
    if LOGO.exists():
        doc.add_picture(str(LOGO), width=Inches(1.65))
        logo_p = doc.paragraphs[-1]
        logo_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        logo_p.paragraph_format.space_after = Pt(14)
        logo_p.paragraph_format.space_before = Pt(6)
    else:
        add_para(doc, "[University Logo]", size=10, italic=True, align="center", space_after=10)

    add_para(doc, "Minor Project (ARP 455)", size=18, bold=True, align="center", space_after=16)
    add_para(
        doc,
        "When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering",
        size=14,
        bold=True,
        align="center",
        space_after=8,
    )
    add_para(doc, "(Adaptive Temporal RAG)", size=12, italic=True, align="center", space_after=28)

    add_para(doc, "Submitted by:", size=12, bold=True, align="center", space_after=6)
    add_para(doc, "Shashank Kumar (08619051623) AIML-B2", size=12, align="center", space_after=2)
    add_para(doc, "Daksh Agarwal (10819051623) AIML-B2", size=12, align="center", space_after=22)

    add_para(doc, "Under the Supervision of", size=12, bold=True, align="center", space_after=6)
    add_para(doc, "Dr. Annu Priya", size=12, align="center", space_after=2)
    add_para(doc, "Assistant Professor, USAR", size=11, italic=True, align="center", space_after=28)

    add_para(doc, "University School of Automation and Robotics,", size=12, align="center", space_after=2)
    add_para(doc, "Guru Gobind Singh Indraprastha University", size=12, align="center", space_after=2)
    add_para(doc, "Surajmal Vihar, New Delhi - 110092", size=11, align="center", space_after=18)
    add_para(
        doc,
        "1st Mid-Term Evaluation  |  Academic Session 2026-27  |  Evaluation Date: 8 October 2026",
        size=11,
        italic=True,
        align="center",
        space_after=8,
    )

    # ========================= 1. INTRODUCTION =========================
    add_section_start(doc, "1. Introduction")
    add_para(
        doc,
        "Large Language Models (LLMs) have become the default backend for conversational assistants, search "
        "summarization, and enterprise question answering. Despite strong fluency, their parametric knowledge is "
        "frozen at training time. After the cut-off date, the model cannot observe the world; it can only remix "
        "what was already stored in weights. For time-sensitive questions - Who is the current Prime Minister?, "
        "Who was President in 2018?, What changed in the company policy last quarter? - this limitation appears "
        "as outdated answers, confident hallucinations, or silent failure modes that are hard for end users to detect.",
        align="justify",
    )
    add_para(
        doc,
        "Retrieval-Augmented Generation (RAG) addresses staleness by fetching external passages at query time and "
        "conditioning generation on that evidence. In the classic dense-RAG pattern, a query is embedded, similar "
        "documents are retrieved from a vector index, and the top-k passages are concatenated into a prompt. This "
        "pipeline improves factuality on knowledge-intensive benchmarks, but it still treats almost every query the "
        "same way: always retrieve, always rank by semantic similarity, always spend the latency and cost of retrieval.",
        align="justify",
    )
    add_para(
        doc,
        "Semantic similarity alone is a poor proxy for temporal correctness. A passage about the same entity can be "
        "highly similar in wording yet describe the wrong time period. Current-state questions need recent evidence; "
        "historical questions need evidence near a target year or exact date; timeless definitional or arithmetic "
        "questions may need no corpus evidence at all. Applying one fixed retrieval policy to all of these cases "
        "produces three recurring failure types: (i) recency errors, where an older but similar document outranks a "
        "newer correct one; (ii) historical mismatch, where the newest document is preferred even though the user "
        "asked about a past period; and (iii) over-retrieval, where unnecessary documents add noise, latency, and cost.",
        align="justify",
    )
    add_para(
        doc,
        "This minor project proposes Adaptive Temporal RAG: a complete system that first analyzes the query, then "
        "selects a retrieval and ranking policy conditioned on that analysis. A multitask transformer analyzer predicts "
        "three signals - retrieval necessity, temporal class (current / historical / timeless), and confidence. A "
        "deterministic policy router maps those signals to no-retrieval, recency ranking, adaptive historical windowing, "
        "or semantic-only fallback. On the Adaptive path only, an evidence-sufficiency feedback loop judges whether the "
        "retrieved set is strong enough and, if not, applies repairs (widen window, expand candidates, raise top-k, "
        "relax to semantic, boost recency) before generation.",
        align="justify",
    )
    add_para(
        doc,
        "The implementation is evaluated against three baselines that share the same corpus, index, and generator "
        "interface: B1 LLM-only (never retrieve), B2 Standard RAG (always semantic), and B3 Fixed Temporal RAG "
        "(always retrieve with a rigid temporal rule). Delivery artifacts include a Google Colab end-to-end notebook "
        "(dataset build, indexing, analyzer training, batch evaluation, interactive demo, artifact download), a local "
        "Streamlit demo with comparative analysis, and automated smoke tests covering adaptive windows, sufficiency "
        "repairs, and multi-provider API failover (Groq -> OpenRouter -> OpenCode).",
        align="justify",
    )
    add_para(
        doc,
        "Scope note for mid-term: the corpus is a synthetic educational collection designed so that historical and "
        "current rankings diverge for the same entities. Results reported here are therefore project-internal bake-offs "
        "and Colab pilot training logs, not claims of state-of-the-art performance on public leaderboards such as FreshQA. "
        "The scientific contribution at this stage is the joint policy design, the adaptive window + sufficiency machinery, "
        "and a reproducible comparison protocol suitable for viva demonstration.",
        align="justify",
    )

    # ========================= 2. LITERATURE =========================
    add_section_start(doc, "2. Literature Survey")
    add_para(
        doc,
        "The literature relevant to this project spans four threads: foundational RAG, temporal/freshness-aware "
        "retrieval, adaptive or selective retrieval, and the remaining research gap that Adaptive Temporal RAG targets. "
        "Landmark contributions are summarized below.",
        align="justify",
    )

    add_heading_num(doc, "2.1 Foundational Retrieval-Augmented Generation", level=2)
    add_para(
        doc,
        "Lewis et al. (2020) introduced RAG by coupling a dense retriever with a sequence-to-sequence generator, "
        "showing large gains on knowledge-intensive NLP tasks where parametric memory alone is insufficient. Dense "
        "Passage Retrieval (Karpukhin et al., 2020) demonstrated that dual-encoder embeddings trained with in-batch "
        "negatives often outperform classical sparse lexical retrieval for open-domain QA. Fusion-in-Decoder "
        "(Izacard & Grave, 2021) improved how multiple retrieved passages are consumed during generation by encoding "
        "passages independently and fusing them in the decoder. These works establish the Standard RAG baseline used "
        "in this project: always retrieve, rank by semantic similarity, generate from top-k evidence.",
        align="justify",
    )

    add_heading_num(doc, "2.2 Temporal and Freshness-Aware Retrieval", level=2)
    add_para(
        doc,
        "FreshLLM / FreshQA-style evaluations (Vu et al., 2023) show that both closed-book LLMs and naive RAG stacks "
        "fail frequently on fast-changing facts unless retrieval prefers recent, trustworthy sources. Subsequent temporal "
        "RAG variants incorporate document timestamps into candidate filtering, re-ranking, or prompt construction so that "
        "evidence is aligned with the time referenced by the query. Such methods motivate our Fixed Temporal RAG baseline "
        "(B3). However, most temporal systems still assume retrieval is always on and typically apply one temporal rule "
        "(for example a fixed +/- N year window or a global recency boost) to every query, regardless of whether the "
        "question is current, historical, or timeless.",
        align="justify",
    )

    add_heading_num(doc, "2.3 Adaptive / Selective Retrieval", level=2)
    add_para(
        doc,
        "A parallel line of work asks when to retrieve rather than how to time-rank. Self-RAG (Asai et al., 2023) trains "
        "models to decide whether to retrieve and to critique generated evidence through reflection tokens. Adaptive-RAG "
        "(Jeong et al., 2024) routes questions by complexity to no-retrieval, single-step, or multi-step retrieval "
        "pipelines. FLARE (Jiang et al., 2023) triggers retrieval dynamically when generation confidence drops. These "
        "systems reduce unnecessary retrieval and can improve efficiency, but they do not jointly model temporal intent "
        "(present versus past versus timeless) as a first-class routing signal.",
        align="justify",
    )

    add_heading_num(doc, "2.4 Embedding, Reranking, and Practical Stack Choices", level=2)
    add_para(
        doc,
        "Sentence-BERT style bi-encoders (Reimers & Gurevych, 2019) and stronger open embedding models such as BGE "
        "(Xiao et al., 2023) make dense indexes practical at coursework scale. FAISS (Johnson et al., 2019) provides "
        "efficient similarity search. Cross-encoders (for example ms-marco MiniLM) improve precision by jointly scoring "
        "query-passage pairs after a cheap first-stage recall step. DistilBERT (Sanh et al., 2019) and DeBERTa "
        "(He et al., 2021) offer compact or stronger encoders for the multitask analyzer used in this project. Our "
        "implementation uses BGE-small + FAISS + cross-encoder rerank for retrieval, and DistilBERT (T4 pilot) / DeBERTa "
        "(A100 large stage) for query analysis.",
        align="justify",
    )

    add_heading_num(doc, "2.5 Research Gap", level=2)
    add_bullets(
        doc,
        [
            "Adaptive systems decide if retrieval is needed, but usually ignore temporal class.",
            "Temporal systems decide how to weight time, but usually retrieve for every query with one fixed rule.",
            "Few systems jointly estimate necessity + temporal category + confidence, then repair weak evidence with a feedback loop.",
            "Gap addressed here: a query-conditioned Adaptive Temporal RAG policy with adaptive historical windows and evidence-sufficiency repairs, evaluated against B1/B2/B3 under a shared protocol.",
        ],
    )

    # ========================= 3. PROBLEM =========================
    add_section_start(doc, "3. Problem Statement")
    add_para(
        doc,
        "In conventional RAG, the retrieval subsystem is largely query-agnostic after embedding. The same top-k semantic "
        "search is executed whether the user asks for a definition, a current office-holder, or a fact from a specific "
        "year. Document timestamps, if present, are often unused or applied only through brittle hand-written filters. "
        "As a result, time-sensitive information needs expose systematic errors even when the corpus contains the correct "
        "evidence.",
        align="justify",
    )
    add_para(doc, "Concrete failure modes motivating this project:", bold=True)
    add_bullets(
        doc,
        [
            "Outdated current answers: an older biography or news snippet about the same entity outranks a newer correct passage because lexical/semantic overlap is high.",
            "Wrong-period historical answers: the newest document is preferred even though the user asked about 2018 or an exact past date.",
            "Unnecessary retrieval: timeless encyclopedia questions and trivial closed-book items still pay full retrieval latency and may inject distracting context.",
            "Rigid temporal filters: a fixed +/- 2 year window can be too narrow for vague historical cues and too wide for exact-date questions, discarding good evidence or admitting noise.",
            "No recovery after weak retrieval: if the first candidate set is incomplete, many pipelines generate anyway instead of widening search or relaxing constraints.",
        ],
    )
    add_para(
        doc,
        "Formal problem addressed: Given a natural-language question q and a timestamped corpus C, automatically infer "
        "whether external evidence is required and what temporal intent q carries, then select a query-specific retrieval "
        "and ranking strategy (including adaptive date windows and evidence repair) that improves temporal correctness and "
        "reduces unnecessary retrieval without degrading answer quality relative to strong fixed-policy baselines.",
        align="justify",
    )
    add_para(
        doc,
        "Success criteria for the minor project are operational rather than leaderboard-chasing: (1) a working Adaptive "
        "pipeline with measurable routing signals; (2) fair comparison against B1/B2/B3 on the same test split; "
        "(3) demonstrable demos in Colab and Streamlit; and (4) honest reporting of both bake-off metrics and Colab pilot "
        "training limitations.",
        align="justify",
    )

    # ========================= 4. MOTIVATION =========================
    add_section_start(doc, "4. Motivation")
    add_para(
        doc,
        "Practical motivation. Production assistants are constantly asked current-event and office-holder questions. A "
        "single silently wrong answer can erode user trust more than an explicit refusal. At the same time, retrieval is "
        "not free: every call adds embedding cost, index latency, reranker compute, and longer prompts for the generator. "
        "Skipping retrieval when it is unnecessary therefore improves both reliability and efficiency. Enterprise corpora "
        "- policies, filings, newsrooms, knowledge bases - almost always carry timestamps, yet many deployed RAG stacks "
        "still ignore that metadata beyond crude metadata filters.",
        align="justify",
    )
    add_para(
        doc,
        "Research motivation. Temporal reasoning remains a known weakness of LLMs and of RAG systems that optimize only "
        "for semantic overlap. The literature already contains strong pieces of the puzzle (selective retrieval on one "
        "side; temporal re-ranking on the other), but a compact system that jointly models whether to retrieve and which "
        "temporal policy to apply is still underexplored at coursework scale. Adding an evidence-sufficiency feedback loop "
        "makes the Adaptive path more than a static router: the system can detect weak evidence and spend extra retrieval "
        "budget only when needed.",
        align="justify",
    )
    add_para(
        doc,
        "Pedagogical / project motivation. Adaptive Temporal RAG is well-scoped for ARP-455: it has a clear problem "
        "statement, implementable components (dataset, index, analyzer, four pipelines, metrics, demos), and a transparent "
        "evaluation story that can be shown in a viva. The mid-term goal is not to claim public SOTA; it is to ship a "
        "complete adaptive system, measure it against honest baselines, and identify what the next A100-scale training "
        "run must improve (especially temporal-class learning).",
        align="justify",
    )
    add_para(doc, "Design principles guiding the implementation:", bold=True)
    add_bullets(
        doc,
        [
            "Share one corpus/index/generator across B1-B3 and Adaptive so differences are attributable to policy.",
            "Prefer learned routing (analyzer) with deterministic safety fallbacks (low-confidence semantic path, trivial no-retrieval).",
            "Keep B3 as a strong fixed temporal baseline; make Adaptive strictly more flexible (adaptive windows + sufficiency).",
            "Report extractive bake-off numbers and Colab pilot training numbers separately, without mixing claims.",
            "Package reproducibility: Colab E2E notebook + artifact zip + local Streamlit reload path.",
        ],
    )

    # ========================= 5. DESIGN FLOW =========================
    add_section_start(doc, "5. Proposed Design Flow")
    add_para(
        doc,
        "This section presents the design flow through two architectures: (A) Temporal RAG - the fixed temporal baseline "
        "used for comparison (B3); and (B) Adaptive Temporal RAG - the proposed system. Both architectures consume the "
        "same dated corpus, the same dense index, and the same generator facade. The only intentional differences are "
        "whether a query analyzer is present, whether retrieval can be skipped, how historical windows are chosen, and "
        "whether weak evidence can trigger repair rounds.",
        align="justify",
    )

    add_heading_num(doc, "5.1 System Architecture and Process Flowchart", level=2)
    add_para(doc, "Architecture A - Temporal RAG (Fixed Temporal Baseline / B3)", size=11, bold=True, space_before=4)
    add_para(
        doc,
        "Temporal RAG always retrieves. The query is embedded with BGE-small; FAISS returns a wide candidate set "
        "(candidate_k). An optional cross-encoder re-ranks those candidates by joint query-passage relevance. A fixed "
        "temporal rule then applies: if a year or date is detected in the query (or provided as a time hint), candidates "
        "are filtered to a rigid +/- 2-year window and re-ranked for historical proximity to that target; otherwise "
        "candidates receive a recency boost favoring newer timestamps. Final ranking uses the fusion formula "
        "final_score = alpha * semantic_score + (1 - alpha) * temporal_score with default alpha approximately 0.55-0.6. "
        "Top-k passages are passed to the generator. There is no query analyzer, no confidence gate, and no option to "
        "skip retrieval.",
        align="justify",
    )
    try_add_image(
        doc,
        MEDIA / "image1.jpg",
        width=5.6,
        caption="Figure 1. Temporal RAG architecture (always retrieve + fixed temporal policy).",
    )
    add_para(doc, "Design flow summary (Temporal RAG / B3):", bold=True)
    add_bullets(
        doc,
        [
            "Input query -> BGE embedding -> FAISS candidate retrieval (always on).",
            "Cross-encoder rerank over FAISS candidates.",
            "If year/date present -> historical window filter (+/- 2 years) -> historical temporal re-rank.",
            "Else -> recency-based temporal re-rank.",
            "Score fusion -> top-k evidence -> generator (extractive or API LLM) -> answer.",
        ],
    )

    add_para(doc, "Architecture B - Adaptive Temporal RAG (Proposed System)", size=11, bold=True, space_before=8)
    add_para(
        doc,
        "Adaptive Temporal RAG inserts a multitask query analyzer before retrieval. The analyzer (DistilBERT on the T4 "
        "pilot; DeBERTa on the planned A100 large stage) reads the query once and emits: (1) retrieval necessity; "
        "(2) temporal class - current, historical, or timeless; (3) confidence. A deterministic policy router then selects "
        "behavior. Trivial closed-book items with needs_retrieval = 0 skip the index entirely. Otherwise the router chooses "
        "current (recency re-rank), historical (adaptive date window + historical re-rank), or semantic-only fallback "
        "(used for low confidence, timeless soft retrieval, or safer defaults). After ranking, an evidence-sufficiency "
        "judge scores coverage, retrieval strength, and temporal fit. If the score is below threshold, repairs are applied "
        "and retrieval is retried for up to two rounds; a weak answer can trigger one additional post-answer retry. "
        "Generation uses extractive offline evaluation or multi-provider API failover: Groq -> OpenRouter -> OpenCode.",
        align="justify",
    )
    try_add_image(
        doc,
        MEDIA / "image2.jpg",
        width=5.6,
        caption="Figure 2. Adaptive Temporal RAG architecture (conditional retrieval + class-specific policy).",
    )
    add_para(doc, "Design flow summary (Adaptive Temporal RAG):", bold=True)
    add_bullets(
        doc,
        [
            "Query -> multitask analyzer -> necessity, temporal class, confidence.",
            "If trivial and needs_retrieval = 0 -> generate without documents (true no-retrieval).",
            "Else policy routes to current / historical_adaptive_window / semantic fallback.",
            "Shared retriever subgraph: BGE + FAISS + cross-encoder -> mode-specific temporal fusion.",
            "Evidence sufficiency loop (Adaptive only): judge -> optional repairs -> re-retrieve (<= 2 rounds).",
            "Generator -> answer + analyzer fields + retrieval/sufficiency meta for demo/eval.",
        ],
    )

    add_para(doc, "Comparative design view:", bold=True, space_before=6)
    add_table(
        doc,
        ["Aspect", "Temporal RAG (B3)", "Adaptive Temporal RAG"],
        [
            ["Retrieval", "Always", "Conditional (analyzer + trivial skip)"],
            ["Temporal policy", "Fixed +/-2y / recency", "Class-specific + adaptive historical window"],
            ["Query analyzer", "No", "Yes (DistilBERT / DeBERTa multitask)"],
            ["Evidence repair", "No", "Yes (sufficiency feedback loop)"],
            ["Skip retrieval", "No", "Yes (trivial / no-retrieval path)"],
            ["Ranking formula", "alpha*sem + (1-alpha)*temp", "Same when temporal modes apply"],
            ["Demo visibility", "Answer + docs", "Answer + analyzer + policy + sufficiency meta"],
            ["Role in project", "Baseline B3", "Proposed system"],
        ],
        col_widths=[1.5, 2.2, 2.6],
    )

    add_heading_num(doc, "5.2 Online and Offline Data Flow", level=2)
    add_para(
        doc,
        "Data stores: (i) data/corpus.jsonl - dated passages with doc_id, text, timestamp, and entity/topic metadata; "
        "(ii) data/queries.jsonl - labelled queries with needs_retrieval, temporal_class, time_hint, gold answers/docs, "
        "and train/val/test splits; (iii) artifacts/ - FAISS index, embeddings, trained analyzer weights, evaluation "
        "outputs, and comparative charts. Online path: user query -> (optional) analyzer -> retrieve/rank -> sufficiency "
        "(Adaptive) -> generate -> answer + evidence. Offline path: build dataset -> build index -> train analyzer -> "
        "evaluate four systems -> plot comparative analysis -> demo. Inference never reads gold labels; labels are used "
        "only for analyzer training and metric computation.",
        align="justify",
    )

    # ========================= 6. METHODOLOGY =========================
    add_section_start(doc, "6. Proposed Methodology")
    add_heading_num(doc, "6.1 Project Objectives", level=2)
    add_bullets(
        doc,
        [
            "Routing objective: Fine-tune a multitask analyzer that jointly predicts retrieval necessity, temporal class, and confidence from query text alone.",
            "Policy objective: Implement Adaptive Temporal RAG with class-specific ranking and an adaptive historical date window, while keeping B3 as a fixed +/-2 year baseline.",
            "Evidence-quality objective: Add an evidence-sufficiency feedback loop that repairs weak retrieval before answering.",
            "Comparison objective: Evaluate Adaptive against B1 (LLM-only), B2 (Standard RAG), and B3 (Fixed Temporal RAG) using Contains, EM, token-F1, temporal accuracy, P@k/R@k, latency, and retrieval rate.",
            "Delivery objective: Ship notebooks/00_colab_e2e.ipynb (dataset -> index -> train -> eval -> interactive demo -> artifact download) and app/streamlit_app.py with a comparative analysis page.",
            "Generation objective: Support extractive offline evaluation and multi-provider API generation with automatic failover (Groq -> OpenRouter -> OpenCode).",
            "Honesty objective: Separate local bake-off metrics from Colab pilot training logs; clearly state synthetic-corpus limits.",
        ],
    )

    add_heading_num(doc, "6.2 Six-Stage Technical Pipeline", level=2)
    add_para(
        doc,
        "1. Dataset selection and labeling. A synthetic educational corpus of approximately 800 short dated passages "
        "covers politics, business leadership, sports, awards, laws, and timeless encyclopedia facts. High-churn entities "
        "appear at multiple timestamps so that historical versus current ranking can be distinguished experimentally. "
        "Labelled queries include needs_retrieval, temporal_class, optional time_hint, gold_answer, and gold_doc_ids. "
        "Approximate class balance is ~35% current / ~30% historical / ~30% timeless, with ~25% no-retrieval items. "
        "Local full builds target about 1050 queries with a 70/15/15 split; the Colab T4 pilot can target larger "
        "TARGET_QUERIES (for example 3000) and multiply training rows with paraphrase augmentation (AUGMENT_FACTOR) so "
        "that GPU time is meaningful without inventing new facts.",
        align="justify",
    )
    add_para(
        doc,
        "2. Indexing and dense retrieval. Documents are embedded with BAAI/bge-small-en-v1.5 using a retrieval instruction "
        "prefix and indexed in FAISS IndexFlatIP over L2-normalized vectors (inner product approximates cosine similarity). "
        "At inference, FAISS returns up to candidate_k neighbors (often ~80). A cross-encoder "
        "(cross-encoder/ms-marco-MiniLM-L-6-v2) then re-ranks those candidates and keeps a shorter list before temporal "
        "fusion. B2, B3, and Adaptive share this first-stage stack so later differences come from temporal policy and routing.",
        align="justify",
    )
    add_para(
        doc,
        "3. Query analyzer training. A DistilBERT (T4 pilot) or DeBERTa (A100 large) encoder is fine-tuned with three heads: "
        "binary necessity, three-way temporal class, and a confidence head. Training uses labelled train queries, optional "
        "paraphrase augmentation, and early stopping / best-checkpoint selection on validation necessity F1 and temporal "
        "accuracy. The checkpoint is stored as artifacts/analyzer.pt and loaded by Adaptive inference, Streamlit, and the "
        "Colab demo widgets. There is no separate learned has_time_span head; year/date hints for ranking are parsed from "
        "query text (and optional dataset time_hint) by deterministic temporal utilities.",
        align="justify",
    )
    add_para(
        doc,
        "4. Temporal scoring and adaptive windows. Current mode maps document age to a recency affinity (favoring newer "
        "timestamps relative to a fixed evaluation reference date for reproducibility). Historical mode peaks affinity near "
        "the inferred target year/date. B3 keeps a rigid +/-2-year window. Adaptive calls infer_adaptive_window(): exact "
        "dates start tight (about +/-0.35y, expand up to ~1.5y), year mentions start around +/-1y (up to ~4y), decade cues "
        "start wider, and vague historical cues start around +/-3y (up to ~8y). The window expands until enough in-window "
        "documents are found or the max half-width is reached. Semantic and temporal scores are fused with alpha ~ 0.55-0.6.",
        align="justify",
    )
    add_para(
        doc,
        "5. Evidence sufficiency loop and generation. After ranking, assess_evidence_sufficiency() scores query-term "
        "coverage (weight 0.35), top retrieval score (0.25), mean retrieval score (0.15), temporal fit (0.15), and lexical "
        "overlap (0.10). If the aggregate score is below ~0.55, the Adaptive pipeline applies repairs and re-retrieves for "
        "up to two rounds: expand_candidates, raise_topk, widen_window, relax_to_semantic, and/or boost_recency. A weak "
        "generated answer can trigger one more retry. The generator facade supports an extractive stub for fast offline "
        "metrics and API backends with provider failover for higher-quality answers in demos.",
        align="justify",
    )
    add_para(
        doc,
        "6. Comparative benchmarking and demos. scripts/evaluate.py runs B1/B2/B3/Adaptive on the held-out test split and "
        "logs per-example outputs. scripts/plot_results.py and the comparative analysis module produce overall tables, "
        "by-class breakdowns, deltas versus baselines, efficiency charts (quality versus retrieval rate), and markdown/"
        "JSON summaries for the report. Streamlit exposes system toggles and a Comparative analysis page. The Colab "
        "notebook runs the same stages interactively and downloads an artifact zip that can overwrite local artifacts/ "
        "and data/ for offline reuse without retraining.",
        align="justify",
    )

    add_heading_num(doc, "6.3 Evaluation Protocol", level=2)
    add_para(
        doc,
        "All four systems are scored on the same test queries. Primary metrics: Contains (gold answer string appears in "
        "prediction), Exact Match, token-level F1, temporal accuracy (heuristic based on top-1 evidence date versus query "
        "temporal intent), Precision@k / Recall@k when gold document IDs exist, mean latency, and retrieval rate "
        "(fraction of queries that hit the index). Because extractive generation quotes from retrieved text, skipped "
        "retrieval on timeless items can lower Contains even when routing is correct; this is discussed explicitly in "
        "Results rather than hidden.",
        align="justify",
    )

    # ========================= 7. RESULTS =========================
    add_section_start(doc, "7. Results (Work Completed up to Mid-Term Evaluation)")
    add_para(
        doc,
        "In accordance with the 1st Mid-Term Evaluation schedule, milestones have been completed through dataset "
        "construction, indexing, all four pipelines, Adaptive enhancements (adaptive window + sufficiency + API failover), "
        "Colab T4 pilot training, local comparative bake-off, Streamlit/Colab demos, and smoke testing. Two complementary "
        "quantitative views are reported: (A) system-level bake-off metrics on 75 held-out test queries with the extractive "
        "generator; and (B) Colab T4 analyzer-training logs plus post-training sanity queries from the GPU pilot session.",
        align="justify",
    )

    add_heading_num(doc, "7.1 Summary of Completed Milestones (Weeks 1 to Mid-Term)", level=2)
    add_bullets(
        doc,
        [
            "Weeks 1-2 (Literature and problem definition): Formalized the joint necessity + temporal-intent gap; defined B1/B2/B3 vs Adaptive protocol, metrics, and viva narrative.",
            "Weeks 2-3 (Dataset): Built corpus.jsonl and queries.jsonl with multi-timestamp entities, temporal labels, gold answers/docs, and stratified splits; Colab path supports larger TARGET_QUERIES and train paraphrase augmentation.",
            "Weeks 4-5 (Baselines and Fixed Temporal RAG): Implemented B1 LLM-only, B2 Standard RAG, and B3 Fixed Temporal RAG (+/-2 year / recency) on a shared FAISS index and generator interface.",
            "Weeks 6-7 (Adaptive analyzer and retrieval upgrade): Implemented multitask DistilBERT analyzer heads; confidence-aware policy router; upgraded embeddings to BGE-small and added cross-encoder rerank.",
            "Weeks 7-8 (Adaptive enhancements and delivery): Adaptive historical windows; evidence-sufficiency feedback loop; Groq->OpenRouter->OpenCode failover; Colab E2E notebook; Streamlit comparative analysis; e2e smoke tests (adaptive window, sufficiency repairs, failover routing).",
        ],
    )

    add_heading_num(doc, "7.2 Quantitative Proof of Work", level=2)
    add_para(
        doc,
        "A. Comparative bake-off on the held-out test split (75 queries; extractive generator). Metrics were computed by "
        "scripts/evaluate.py over the same questions for all four systems. These numbers are real measured outputs on the "
        "project's synthetic educational corpus; they are not public-leaderboard scores.",
        align="justify",
    )
    add_table(
        doc,
        ["System", "Contains", "Temporal Acc.", "Retrieval Rate", "Role"],
        [
            ["B1 LLM-only", "0.027", "0.307", "0%", "Never retrieves"],
            ["B2 Standard RAG", "0.427", "0.573", "100%", "Always semantic"],
            ["B3 Fixed Temporal RAG", "0.773", "0.867", "100%", "Always + fixed window"],
            ["Adaptive Temporal RAG", "0.600", "0.840", "68%", "Proposed conditional policy"],
        ],
        col_widths=[2.1, 1.0, 1.2, 1.2, 1.5],
    )
    add_para(
        doc,
        "Headline deltas for Adaptive versus Standard RAG: about +17 percentage points Contains and +27 points Temporal "
        "Accuracy, while retrieving on only 68% of queries (~32% less retrieval than B2/B3). Versus Fixed Temporal RAG, "
        "Adaptive nearly matches temporal accuracy (0.840 vs 0.867). The Contains gap (0.600 vs 0.773) is interpreted in "
        "Section 7.3 as largely an extractive-evaluation artifact on timeless / skipped-retrieval items rather than a "
        "collapse of temporal ranking on current/historical questions.",
        align="justify",
    )

    add_para(
        doc,
        "B. Colab T4 pilot - analyzer training (authoritative artifact: artifacts/colab_t4_pilot_run.json). This run "
        "validates that the end-to-end Colab recipe executes on GPU before the planned A100 large stage.",
        align="justify",
        space_before=4,
    )
    add_table(
        doc,
        ["Field", "Value"],
        [
            ["Stage", "t4_pilot"],
            ["Train rows (before augment)", "2850"],
            ["Validation queries", "75"],
            ["Augment factor", "8 (-> ~22,800 train examples)"],
            ["Epochs / batch size", "12 / 8"],
            ["Wall-clock training time", "32.5 minutes"],
            ["Best validation score", "0.4762"],
            ["Final train loss (epoch 12)", "1.6975"],
            ["Val temporal accuracy (range)", "~0.28-0.33 (near chance for 3 classes)"],
            ["Val necessity F1 (range)", "~0.37-0.63 (unstable across epochs)"],
        ],
        col_widths=[2.7, 3.8],
    )
    add_para(doc, "Selected epoch log (from the same pilot):", bold=True)
    add_table(
        doc,
        ["Epoch", "Train loss", "Val nec F1", "Val temp acc"],
        [
            ["1", "1.7962", "0.366", "0.333"],
            ["2", "1.7365", "0.619", "0.333"],
            ["6", "1.7090", "0.628", "0.320"],
            ["9", "1.7001", "0.619", "0.280"],
            ["12", "1.6975", "0.432", "0.320"],
        ],
        col_widths=[1.2, 1.4, 1.4, 1.5],
    )
    add_para(doc, "Post-training sanity queries from the same Colab run:", bold=True)
    add_table(
        doc,
        ["Query", "Predicted class", "Needs retrieval", "Confidence"],
        [
            ["Who is the current Prime Minister of India?", "timeless", "0", "0.88"],
            ["Who was the President of the United States in 2018?", "timeless", "1", "0.41"],
            ["Define photosynthesis.", "timeless", "0", "0.88"],
            ["What is 2 + 2?", "timeless", "0", "0.88"],
        ],
        col_widths=[3.0, 1.3, 1.3, 1.1],
    )

    add_para(
        doc,
        "C. Engineering verification (smoke tests). Automated e2e checks covering dataset/index/analyzer load, B1-B3/"
        "Adaptive answers, adaptive window metadata (year and exact_date cases), sufficiency-loop repairs (observed "
        "multi-round widen_window behavior on a hard query), Colab notebook cell contracts, and API failover logic "
        "(dead provider then live Groq) passed. At the time of testing, only Groq was configured in the local .env; "
        "OpenRouter and OpenCode keys remain optional for full three-provider live routing.",
        align="justify",
    )

    try_add_image(
        doc,
        MEDIA / "image3.png",
        width=5.6,
        caption="Figure 3. Comparative evaluation charts across B1, B2, B3, and Adaptive Temporal RAG (local bake-off).",
    )

    add_heading_num(doc, "7.3 Key Observations", level=2)
    add_bullets(
        doc,
        [
            "Adaptive is clearly better than Standard RAG on both answer Contains and temporal accuracy while retrieving less often - the intended efficiency/quality trade-off.",
            "Adaptive nearly matches Fixed Temporal on temporal accuracy; B3 remains a strong always-on temporal baseline that Adaptive must continue to close on answer Contains after stronger generation (API/HF) is used for fair timeless scoring.",
            "Under extractive evaluation, skipped or soft-routed timeless questions hurt Adaptive Contains because there is little passage text to quote. This does not mean the router is wrong to avoid retrieval; it means the metric + generator pair under-rewards correct no-retrieval decisions.",
            "The Colab T4 pilot proved pipeline completeness, but the analyzer did not yet learn reliable temporal classes (val temp acc ~ chance; sanity queries collapsed to timeless). Paraphrase-heavy augmentation alone is insufficient; the next A100 large run should emphasize cleaner class signal, stronger encoder, and full four-system Colab eval tables.",
            "Adaptive historical windows and sufficiency repairs are implemented and exercised. They are the main differentiators versus B3 beyond conditional retrieval.",
            "Remaining error patterns to attack next: ambiguous current questions when the corpus snapshot lags the real world; historical questions without an explicit year; extractive truncation of multi-token names; and analyzer confusion between timeless definitions and current-entity questions.",
        ],
    )

    add_heading_num(doc, "7.4 Work Remaining After Mid-Term", level=2)
    add_bullets(
        doc,
        [
            "Run A100 large stage (larger dataset/augmentation, stronger analyzer) and regenerate full B1/B2/B3/Adaptive test metrics with API generation enabled.",
            "Deeper by-class error analysis and qualitative case studies for viva slides.",
            "Final report polish, architecture PNG refresh if needed, and defense rehearsal using Streamlit + Colab demo.",
        ],
    )

    # ========================= 8. GANTT =========================
    add_section_start(doc, "8. GANTT Chart")
    add_para(
        doc,
        "The project follows a twelve-week plan from literature review through evaluation and reporting. Mid-term "
        "evaluation (week of 8 October 2026) corresponds to completion of baselines, Fixed Temporal RAG, Adaptive "
        "integration (including adaptive windows and sufficiency), Colab/Streamlit demos, and initial quantitative "
        "bake-off results. The large A100 training/eval pass and final viva polish remain in the post-mid-term window.",
        align="justify",
    )
    add_para(
        doc,
        "Progress status snapshot: Phase 1 (setup/literature) 100% complete; Phase 2 (data/index/baselines) 100% "
        "complete; Phase 3 (Adaptive modeling + enhancements) substantially complete; Phase 4 (full experiments) active / "
        "in progress; Phase 5 (final reporting and defense) ongoing.",
        align="justify",
    )
    add_table(
        doc,
        ["Phase", "Weeks", "Status (Mid-Term)", "Key Output"],
        [
            ["1. Literature and problem definition", "W1", "Completed", "Problem statement, research gap"],
            ["2. Dataset preparation", "W2-W3", "Completed", "corpus.jsonl, queries.jsonl"],
            ["3. Baselines (B1, B2)", "W4", "Completed", "LLM-only and Standard RAG"],
            ["4. Temporal RAG (Fixed / B3)", "W5", "Completed", "Fixed window + recency ranking"],
            ["5. Adaptive query analyzer", "W6-W7", "Completed", "Multitask DistilBERT / DeBERTa path"],
            ["6. Adaptive system integration", "W7-W8", "Completed", "Adaptive window + sufficiency + Colab E2E"],
            ["7. Experiments and evaluation", "W9-W10", "In progress", "Bake-off done; A100 large run next"],
            ["8. Demo, report and viva prep", "W11-W12", "Ongoing", "Streamlit/Colab demo, final report"],
        ],
        col_widths=[2.5, 0.9, 1.3, 2.0],
    )
    add_para(doc, "Simplified Gantt (active weeks marked with #):", bold=True)
    add_table(
        doc,
        ["Phase", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12"],
        [
            ["Literature", "#", "", "", "", "", "", "", "", "", "", "", ""],
            ["Dataset", "", "#", "#", "", "", "", "", "", "", "", "", ""],
            ["Baselines B1/B2", "", "", "", "#", "", "", "", "", "", "", "", ""],
            ["Temporal RAG", "", "", "", "", "#", "", "", "", "", "", "", ""],
            ["Analyzer", "", "", "", "", "", "#", "#", "", "", "", "", ""],
            ["Adaptive integrate", "", "", "", "", "", "", "#", "#", "", "", "", ""],
            ["Experiments", "", "", "", "", "", "", "", "", "#", "#", "", ""],
            ["Report / Demo", "", "", "", "", "", "", "", "", "", "", "#", "#"],
        ],
    )
    add_para(
        doc,
        "Risk/mitigation note used in planning: if the T4 pilot finishes too quickly, increase AUGMENT_FACTOR or epochs; "
        "if it overruns the 2-3 hour teaching budget, reduce augmentation or freeze lower encoder layers. The recorded "
        "pilot finished in 32.5 minutes, which confirmed correctness but also showed that longer/cleaner training is still "
        "needed for temporal-class quality on the large stage.",
        align="justify",
    )

    # ========================= 9. REQUIREMENTS =========================
    add_section_start(doc, "9. Hardware and Software Requirements")
    add_heading_num(doc, "9.1 Hardware", level=2)
    add_para(
        doc,
        "The project is intentionally runnable on a laptop for extractive evaluation and API-backed demos, while heavier "
        "analyzer training is offloaded to Google Colab GPUs.",
        align="justify",
    )
    add_bullets(
        doc,
        [
            "General: one standard desktop or laptop computer with a multi-core processor (Intel i5 / AMD Ryzen 5 class or better).",
            "Memory: minimum 8 GB RAM; 16 GB or more recommended when loading embedding models, FAISS indexes, and Streamlit together.",
            "GPU: not required for the extractive path or API generation path. Recommended for training: NVIDIA GPU via Google Colab T4 for the pilot stage and A100 for the planned large stage (~6 hour budget).",
            "Storage: minimum 10 GB free disk; 25 GB or more recommended for Hugging Face model caches, embeddings.npy, FAISS index files, and experiment logs.",
            "Network: stable broadband for model downloads and optional Groq/OpenRouter/OpenCode API calls.",
        ],
    )
    add_table(
        doc,
        ["Resource", "Minimum", "Recommended"],
        [
            ["RAM", "8 GB", "16 GB+"],
            ["CPU", "Quad-core", "Higher clock / more cores"],
            ["GPU", "None (extractive/API)", "Colab T4 pilot / A100 large"],
            ["Storage", "10 GB free", "25 GB+"],
            ["Internet", "Required for downloads/API", "Stable broadband"],
        ],
        col_widths=[1.5, 2.2, 2.6],
    )

    add_heading_num(doc, "9.2 Software", level=2)
    add_bullets(
        doc,
        [
            "Operating system: Windows 10/11, Linux, or macOS.",
            "Language: Python 3.10 or above.",
            "Deep learning / NLP: PyTorch; Hugging Face Transformers; sentence-transformers.",
            "Retrieval: FAISS (IndexFlatIP); BAAI/bge-small-en-v1.5 embeddings; cross-encoder/ms-marco-MiniLM-L-6-v2 reranker.",
            "Analyzer: distilbert-base-uncased (T4 pilot) / DeBERTa-v3 family (A100 large stage).",
            "Data and plotting: NumPy, Pandas, Matplotlib.",
            "Demo and notebooks: Streamlit; Google Colab; Jupyter / VS Code / Cursor.",
            "Optional generators: Groq, OpenRouter, OpenCode API keys (failover order configurable); extractive stub for offline metrics.",
            "Experiment artifacts: results.csv, results_summary/comparative JSON/Markdown, charts under artifacts/.",
            "Version control: Git.",
        ],
    )
    add_para(
        doc,
        "Primary runnable entry points: notebooks/00_colab_e2e.ipynb for the GPU end-to-end path; "
        "app/streamlit_app.py for the local interactive demo and comparative analysis page; "
        "scripts/e2e_smoke_test.py for regression checks before a viva run.",
        align="justify",
    )

    # ========================= 10. REFERENCES =========================
    add_section_start(doc, "10. References")
    add_para(
        doc,
        "The following references underpin the literature survey, baseline choices, and implementation stack used in "
        "this mid-term progress report.",
        align="justify",
    )
    refs = [
        "Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems (NeurIPS).",
        "Karpukhin, V., Oguz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. Proceedings of EMNLP.",
        "Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. Proceedings of EACL.",
        "Vu, T., Iyyer, M., Wang, X., et al. (2023). FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.",
        "Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.",
        "Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.",
        "Jiang, Z., Xu, F. F., Gao, L., et al. (2023). Active Retrieval Augmented Generation (FLARE).",
        "Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.",
        "He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. International Conference on Learning Representations (ICLR).",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP.",
        "Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE models).",
        "Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data (FAISS).",
        "Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv preprint (cross-encoder reranking paradigm).",
        "Guru Gobind Singh Indraprastha University / USAR. Minor Project (ARP-455) guidelines and mid-term evaluation schedule, Academic Session 2026-27.",
    ]
    for i, ref in enumerate(refs, 1):
        add_para(doc, f"{i}. {ref}", size=10, space_after=5, align="justify")

    add_para(doc, "- End of Progress Report -", italic=True, align="center", space_before=18)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Backup {BACKUP}")


if __name__ == "__main__":
    main()
