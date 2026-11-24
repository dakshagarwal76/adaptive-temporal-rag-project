<!-- COVER -->
GURU GOBIND SINGH INDRAPRASTHA UNIVERSITY
UNIVERSITY SCHOOL OF AUTOMATION AND ROBOTICS
EAST DELHI CAMPUS, SURAJMAL VIHAR, DELHI - 110032
Minor Project Report
on
When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering (Adaptive Temporal RAG)
Submitted in partial fulfilment of B.Tech. (7th Semester)
In
AI-ML
Name: Daksh Agarwal
Enrollment Number: 10819051623
Under the supervision of
Dr. Annu Priya
Assistant Professor, USAR
November 2026

# CERTIFICATE

This is to certify that the Minor Project Report entitled “When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering (Adaptive Temporal RAG)” is a record of bona fide work carried out by Daksh Agarwal, Enrollment Number 10819051623, a student of B.Tech. (7th Semester), Artificial Intelligence and Machine Learning, University School of Automation and Robotics, Guru Gobind Singh Indraprastha University, East Delhi Campus, in partial fulfilment of the requirements of Minor Project (ARP 455).

The work was completed under my supervision during the seventh semester and is submitted for evaluation on 25 November 2026. The report describes the frozen Checkpoint B pipeline for time-aware question answering, and the measurements it quotes are those saved at commit 44938a7, including negative findings. The matter embodied in this report has not, to the best of my knowledge, been submitted for the award of any other degree or diploma.

Signature: _______________________________

Name: Dr. Annu Priya

Designation: Assistant Professor, University School of Automation and Robotics

Date: _______________________________

# DECLARATION

I hereby declare that the Report entitled (When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering) is an authentic record of work completed as requirements of Minor Project (ARP 455) under the supervision of Dr. Annu Priya, Assistant Professor, University School of Automation and Robotics, GGSIPU.

(Signature of student) ____

(Name of Student) Daksh Agarwal

(Enrollment Number) 10819051623

Date: ____

(Signature of Supervisor)

(Name of Supervisor) Dr. Annu Priya

Date: ____

# ACKNOWLEDGEMENT

The author is grateful to Dr. Annu Priya, Assistant Professor, University School of Automation and Robotics, for supervision of this minor project (ARP 455). Her guidance on problem formulation, on the discipline of a single frozen configuration, and on reporting a negative result with the same care as a positive one shaped both the experiments and this report. Checkpoint B in `configs/final.yaml`, the shared extractive reader and answer projection, and the decision to leave the sufficiency-loop outcome visible beside the full system follow that standard.

The author thanks Prof. Arvinder Kaur, Dean, University School of Automation and Robotics, and the faculty and staff of the School at the East Delhi Campus, Surajmal Vihar, for the academic setting of the seventh-semester project and for the formal requirements that bring the work to evaluation on 25 November 2026. The frozen measurements were obtained on a laptop central processing unit, which made a bounded per-question budget a practical constraint as well as a scientific one.

Project partner Shashank Kumar (Enrollment Number 08619051623) collaborated throughout the design of Adaptive Temporal RAG. Joint work on the pipeline, on the reading of the frozen tables, and on the companion paper draft is acknowledged with thanks. The October pilot, the TimeQA transfer, and the Checkpoint B freeze of 23 November 2026 were shared technical work. Responsibility for the wording of this report, and for copying numbers only from the verified record at commit 44938a7, rests with the author.

The author thanks his family for their patience during the October pilot, the November evaluation runs, and the final days in which this report was written for the 25 November 2026 evaluation.

Daksh Agarwal (10819051623)

# ABBREVIATIONS AND NOMENCLATURE

**Table 0.1:** Abbreviations used in this report, in alphabetical order.

| Abbreviation | Meaning |
|---|---|
| AMP | Automatic mixed precision; the October pilot left an AMP binary-cross-entropy path after a crash. |
| API | Application programming interface; optional generators exist, and the frozen reader is extractive. |
| ARP | Course identifier ARP 455, the USAR minor project. |
| B1 | No-retrieval baseline: empty document list, hardcoded trivia replies (`LLMOnlyPipeline`). |
| B2 | Standard dense retrieval: always retrieve, no date window, no sufficiency loop (`StandardRAGPipeline`). |
| B3 | Fixed ±2 year baseline when a year is found (`FixedTemporalRAGPipeline`). |
| B4 | Dense union BM25 with exponential year decay; no cross-encoder (`TempRALMDecayPipeline`). |
| BCE | Binary cross-entropy, the loss family implicated in the October AMP crash. |
| BGE | Dense encoder `BAAI/bge-small-en-v1.5` (Xiao et al., 2024). |
| BM25 | Okapi BM25 lexical ranker (`BM25Index`); used only by B4 in the frozen comparison. |
| CE | Cross-encoder score from `cross-encoder/ms-marco-MiniLM-L-6-v2`; also the symbol for cross-entropy in the router loss. |
| CI | Paired bootstrap percentile interval, 10,000 resamples, seed 0. |
| CPU | Central processing unit; every frozen evaluation ran on a laptop CPU. |
| DFD | Data-flow diagram (Chapter 5). |
| DPR | Dense passage retrieval (Karpukhin et al., 2020). |
| E-R | Entity–relationship diagram of documents, queries, evidence, and evaluation records. |
| ECE | Expected calibration error; implemented for the large router, and unreported because that run has no metrics. |
| EM | Exact match after SQuAD-style normalisation (`exact_match`). |
| F1 | Token-level F1, the harmonic mean of token precision and token recall. |
| FAISS | Similarity search; the frozen index is exact `IndexFlatIP` (Johnson, Douze, & Jégou, 2019). |
| FiD | Fusion-in-Decoder (Izacard & Grave, 2021); cited for context and not run here. |
| fp32 | 32-bit floating point, the precision specified for the queued DeBERTa router run. |
| GBM | Gradient boosting machine (Friedman, 2001); experimental learned ranker only. |
| GGSIPU | Guru Gobind Singh Indraprastha University. |
| GPU | Graphics processing unit; unused for the frozen evaluations. |
| IR | Information retrieval. |
| JSON | JavaScript Object Notation, used for results, significance, and router statistics. |
| JSONL | JSON Lines, the row format for corpus documents and queries. |
| LLM | Large language model; the frozen reader does not call one. |
| McNemar | Exact two-sided test on discordant answer pairs. |
| MiniLM | Distilled transformer (Wang et al., 2020) used as the MS MARCO cross-encoder. |
| NLP | Natural language processing. |
| P@5 | Precision at 5 against stored gold document identifiers. |
| QA | Question answering. |
| R@5 | Recall at 5 against stored gold document identifiers. |
| RAG | Retrieval-augmented generation (Lewis et al., 2020). |
| SQuAD | Stanford Question Answering Dataset; normalisation follows SQuAD (Rajpurkar, Jia, & Liang, 2018). |
| T4 | NVIDIA T4 used in the October Colab pilot and specified for the Kaggle router run. |
| UI | User interface: Streamlit, and the Next.js application at https://adaptive-temporal-rag.vercel.app. |
| USAR | University School of Automation and Robotics, GGSIPU. |
| YAML | Configuration language of `configs/final.yaml`. |

**Table 0.2:** Principal symbols, following the project paper. Where one letter is reused, both roles are stated.

| Symbol | Meaning |
|---|---|
| $q$ | The input question. |
| $\mathcal{D}$ | The corpus, a set of dated documents. |
| $d_i$ | The $i$-th document (passage or chunk). |
| $\tau_i$ | Index timestamp of $d_i$. On the templated corpus this is the fact date; on TimeQA it is the first date found, or 1 January 1970 if none is found. |
| $t$ | Target date implied by the question. |
| $\delta(d,t)$ | Distance from $t$ to $d$: zero inside a stated interval, otherwise the minimum absolute deviation to an endpoint or a point date. |
| $h$ | Half-width of the historical window, in years. |
| $h_{\max}$ | Maximum half-width for the cue that produced the window. |
| $w$ | Window width in years used to scale the historical temporal score. |
| $\Delta y$ | Absolute year gap. B4 multiplies its blend by $\exp(-\lvert\Delta y\rvert/2)$. |
| $\gamma$ | Policy confidence: half the untrained sigmoid plus half the larger maximum class probability, when blending is on. |
| $\alpha$ | Weight on the min–max semantic score in the default gate. Frozen value $0.6$. |
| $s_i$ | Blended first-stage score: $0.35$ times min–max FAISS plus $0.65$ times the raw cross-encoder score. |
| $\tilde{s}_i$ | Min–max normalised semantic score inside the surviving pool. |
| $g^{\mathrm{cur}}_i$ | Current-mode recency of document $i$ relative to the reference date. |
| $g^{\mathrm{hist}}_i$ | Historical temporal score of document $i$, decreasing in $\delta(d_i,t)$. |
| $b_i$ | Lexical boost $\min(0.25, 0.05H)$, where $H$ is the count of remaining query tokens found in the document. |
| $S$ | Sufficiency score. Evidence is accepted during retrieval when the subject check passes and $S \ge 0.55$. |
| $C$ | Fraction of content query tokens found in the retrieved text. |
| $T$ | In the sufficiency score, temporal fit in $[0,1]$. In the router, the softmax temperature. The two uses are distinct. |
| $k$ | Documents passed to the reader. Frozen value $k = 5$. |
| $\mathcal{W}(q)$ | Cue-dependent historical window for question $q$. |
| $\lambda$ | Weight on the temporal cross-entropy. Frozen value $1.75$. The confidence head is outside this loss. |
| $p^{\mathrm{nec}}$ | Softmax distribution of the two-way necessity head. |
| $p^{\mathrm{tmp}}$ | Softmax distribution of the three-way temporal head (current, historical, timeless). |
| $\hat{y}$ | Answer string returned by the system. |

# ABSTRACT

Adaptive Temporal RAG asks when retrieval should run and which dated evidence should be read. Checkpoint B (`configs/final.yaml`) embeds text with `BAAI/bge-small-en-v1.5` and searches it by exact FAISS `IndexFlatIP`. The cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks 80 candidates and keeps 40. The blend is 0.35 times the min–max dense score plus 0.65 times the raw cross-encoder score, and five documents reach a shared extractive reader. BM25 is used only by baseline B4. The analyser `distilbert-base-uncased` has a necessity head and a temporal head. Its confidence head lies outside the loss; the October 2026 pilot observed a near-constant 0.75. Retrieval is skipped only when necessity is 0 and `is_trivial_query` matches a fixed trivia list. The reference date is 1 October 2025. Every frozen evaluation ran on a laptop CPU.

On 193 questions about six entities, Adaptive exact match is 0.762 and the fixed ±2 year filter (B3) is 0.684. The difference is +0.078, the 95% bootstrap interval is [+0.041, +0.119], and exact McNemar $p = 6.10\mathrm{e}{-05}$. Containment differs by +0.052 ($p = 0.002$). Versus B4, dense retrieval union BM25 with exponential year decay and no cross-encoder, the exact-match difference is +0.404. On 77 templated questions containment is tied at 0.987, and exact match differs by one question ($p = 1.000$). On 500 TimeQA questions containment is 0.296, compared with 0.260 for standard retrieval ($p = 0.006$), 0.270 for the fixed window ($p = 0.117$; the interval includes zero), and 0.234 for B4 ($p = 5.39\mathrm{e}{-04}$). TimeQA containment scores a selected sentence; exact match extracts a span and stays near zero. Stored timestamps are often a birth year or 1970-01-01, so B3 is a weak control and temporal accuracy is undefined. Removing the sentence selector lowers TimeQA containment by 0.128 ($p = 2.89\mathrm{e}{-08}$). The sufficiency loop leaves answer scores unchanged. On the held-out file, Adaptive-Fast mean latency is 0.663 s where full Adaptive is 2.584 s, on a later Adaptive-only run. The probe “Janet Frame received which award in 1990?” fell from 20,974.8 s to 8.184 s. A 100,000-row leak-free router corpus is prepared for `microsoft/deberta-v3-base`. That Kaggle T4 run is in progress and has no accuracy, macro-F1, or calibration figure.

Keywords: adaptive retrieval; temporal question answering; retrieval-augmented generation; date-window ranking; cross-encoder reranking; evidence sufficiency; exact match; TimeQA.

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Evolution of Retrieval-Augmented Generation

Open-domain question answering asks for a short answer drawn from a large body of text. Neural readers first assumed that a relevant passage had already been chosen. The selection problem was then attacked with sparse lexical rankers and, later, with dense dual encoders. Karpukhin et al. (2020) showed that a question encoder and a passage encoder, compared by inner product, can replace a sparse first stage. The passages still have to be read. Lewis et al. (2020) join the two stages: a dense index proposes passages, and a generator is conditioned on them. That pattern is retrieval-augmented generation. The index can be edited when a fact changes, which is attractive precisely when the fact is an office, a statute, or a title that holds only on an interval.

Later work changed how much of the stack is trained and how many passages the reader sees. Izacard et al. (2023) train the retriever and the reader together in Atlas, so neighbour search is no longer a frozen module under an unrelated decoder. Izacard and Grave (2021) fuse many retrieved passages inside the decoder. Neither design is the system frozen here. The frozen reader is extractive and offline. It scores sentences, shortens some of them by pattern, and, on TimeQA, extracts a span before exact match. The choice is deliberate. A comparison of retrieval policies is difficult to read if each policy also brings its own answer stringer. This project holds the reader fixed and varies the decision to retrieve, the time window, and the sufficiency loop.

A parallel line asks whether retrieval should happen at all. Asai et al. (2024) learn tokens that decide to retrieve and tokens that critique a draft. Jeong et al. (2024) route a question to no retrieval, one retrieval, or several retrievals according to complexity. Jiang et al. (2023) retrieve again during generation when the draft looks uncertain. Those systems treat retrieval as a budget that follows the question. They do not, in the accounts used here, score whether a passage date matches the time the question asks about. The project title splits that budget into two questions. “When” asks whether the index should be called. “What” asks which dates, which subject, and which sentence should survive. The implementation answers the second question with a window, a distance over intervals, a subject cut, and a date-aware sentence selector. The first question, in the frozen code, is a whitelist guarded by a classifier.

The corpus side of the problem is older than retrieval-augmented generation. Li and Croft (2003) put document age into a language model so that a recency-seeking query would not depend only on a hand-tuned boost. Kanhabua and Nørvåg (2011) compared mixtures of textual similarity and temporal similarity. Question-answering benchmarks then made the same point with explicit answers: a question can be about “now”, about a named year, or about a fact that does not depend on time (Jia et al., 2018; Chen, Wang, & Wang, 2021; Zhang & Choi, 2021). This project sits between those traditions. It keeps a modern dense index and a cross-encoder, and it places an explicit temporal policy on top, with the constants written down and the reader shared.

## 1.2 The Temporal Staleness and Multi-Period Retrieval Challenge

Three questions, all answered by the frozen Streamlit demonstration on 24 November 2026, show the taxonomy used throughout the report.

“Who is the current Prime Minister of India?” has no meaning until a reference day is fixed. A semantically similar passage about an earlier prime minister is a true sentence and the wrong answer. The demonstration marks the question current, with confidence 0.746 and current-class probability 0.999. The policy name is `current`. The returned string is Narendra Modi. Sufficiency is 0.905 and latency is 0.621 s. The recency score uses 1 October 2025, the day the templated examination was built against. A deployment clock and this examination clock are different objects. The project does not treat the demonstration as a live news evaluation.

“Who was the President of the United States in 2018?” meets several true passages about the same office. Embedding neighbourhood is a weak guide to the year. The demonstration marks the question historical, with `needs_retrieval` 1, necessity probability 0.999, and confidence 0.751. The policy is `historical_adaptive_window`. The cue is a bare year, so the target date is the midpoint 2018-06-15, the initial half-width is 1 year, the maximum is 4 years, and the run expands the window zero times. The top evidence passage is dated 2018-04-16, score 1.046, document `doc_e8b1e7947d`, and the sentence names Donald Trump. Sufficiency is 0.98, above the threshold 0.55, and latency is 2.629 s. The answer string is Donald Trump. This is the behaviour the historical branch is built to produce when the index date is the fact date.

“What is 2 + 2?” does not depend on the corpus. Calling the index spends time and can attach an irrelevant passage to a stable fact. The demonstration sets `needs_retrieval` to 0, temporal class timeless, confidence 0.75, and policy `no_retrieval`. The answer is 4, no evidence is retrieved, and latency is 0.065 s. The skip fires because the necessity class is 0 and the string contains a marker listed in `is_trivial_query` in `src/generate.py`. A necessity prediction of 0 on any other question still retrieves. There is no parametric memory underneath the skip. The closed-book control is the same extractive reader with an empty document list, plus a few hardcoded replies. A skip can be scored only as “the index was not called.”

Real corpora add failure modes that these three questions only suggest. A passage may state a range rather than a point. A biography may open with a birth year, and a chunk with no parsed date may be stored as 1 January 1970. A fixed window on that stored timestamp then drops the paragraph that names the asked year. A repair loop without a cap can spend hours on one question, as the Janet Frame probe did before the guardrails. The measurements that follow show that the components aimed at these modes do not all succeed. The held-out exact-match gain over a fixed window is stable. Templated containment is a tie. The TimeQA comparison with that window is unstable. The sufficiency loop does not buy answers.

## 1.3 Motivation

### 1.3.1 Practical

Office holders, chief executives, competition winners, awards, and statutes change, and a passage that was correct in one year remains in the index afterwards. A system that always returns the nearest passage will sometimes recite the wrong holder. A system that always retrieves, including for arithmetic that the reader resolves by a shortcut, pays the cross-encoder on every question. On the evaluation machine, a Dell Latitude 5420 with an Intel Core i7-1165G7 and 32 GB of memory, that cost is visible: held-out Adaptive mean latency is 2.584 s, and TimeQA Adaptive mean latency is 8.736 s, both on CPU. The practical motive is a policy that can name the time it is aiming at, can refuse a small trivia list, and can stop. The stop has two parts. One is a sufficiency score. Turning its repair loop off does not lower exact match or containment, and the later Adaptive-Fast means are lower: 0.663 s on the held-out file and 3.184 s on TimeQA. The other is a hard budget of three retrieves and 60 seconds, which brought the probe “Janet Frame received which award in 1990?” from 20,974.8 s to 8.184 s. The budget is required even when the score does not improve the answer.

A reader outside the development loop should be able to inspect the decision. The Streamlit application exposes the frozen checkpoint, the analyser label, the policy name, the window, the sufficiency breakdown, and the dated evidence. A hosted Next.js interface at https://adaptive-temporal-rag.vercel.app offers Ask and Results tabs over an 803-document dated corpus, using a lexical retrieval port and the extractive reader. Those interfaces illustrate the policy. They are not a second evaluation, and the hosted confidence figures are not the DistilBERT $\gamma$ of the frozen tables.

### 1.3.2 Research

The research motive is a comparison that does not confound retrieval with answer formatting. An audit on 17 November 2026 found that an earlier pass applied `project_answer` only on the adaptive path, so exact-match gaps mixed a shorter string with a different ranking. Checkpoint B, frozen on 23 November 2026 in commit 0db77f6, shares the reader, the projection, the sentence selector, and the entity or surname cut across B1, B2, B3, and Adaptive. Commit 44938a7 on 24 November 2026 adds B4 and the paired tests. What remains free to vary is the retrieval decision, the time window, and the sufficiency loop. B4 is an additional decay ranker and does not run the cross-encoder, so its latency is not matched to B2 or B3. Under that protocol the project can say, with a $p$-value, where an adaptive window is ahead of a fixed ±2 year filter and where it is not. It can also say that $\exp(-\lvert\Delta y\rvert/2)$ on a dense-plus-BM25 pool does not replace either window on these files. The sufficiency loop, introduced to decide when context is enough, does not raise the saved answer metrics. Adaptive-Fast is reported beside Adaptive so that this negative result stays visible.

The same motive limits claims about published systems. TempRALM re-ranks with a temporal score and can mask text dated after the query time (Gade & Jetcheva, 2024). B4 takes the decay idea only. Adaptive-RAG is the closest system in name and is not reimplemented (Jeong et al., 2024). Chen, Wang, and Wang (2021) report Fusion-in-Decoder exact match of 60.5 on the easy TimeQA test and 46.8 on the hard test, with human exact match 87.0 on the hard test. The 500-question file here is a subset of chunks from `hugosousa/TimeQA`, 5,931 chunks in all, and is not that leaderboard. Exact match on it stays between 0.000 and 0.010 for every frozen system.

### 1.3.3 Pedagogical

The pedagogical motive is to carry a retrieval stack from a pilot to a frozen comparison inside one semester, and to write down where the measurement stops. The stack includes a dense index, a cross-encoder, a small classifier with three heads, a rule-based date tagger, a subject filter, an extractive reader, a paired test, and a leakage filter for a larger router corpus. Each piece is a standard tool with a local rule: FAISS inner product (Johnson, Douze, & Jégou, 2019), a MiniLM cross-encoder in the MS MARCO setting (Wang et al., 2020; Nguyen et al., 2016; Nogueira & Cho, 2019), DistilBERT (Sanh et al., 2019; Devlin et al., 2019), and SQuAD normalisation (Rajpurkar, Jia, & Liang, 2018). The pedagogical value is in the assembly and in the audit. Validation necessity accuracy and temporal accuracy of the templated analyser are both 1.0. The label on that split is often the word “current” or a written year, so the figure says the classifier fits those cues. It does not say the classifier reads time in Wikipedia prose. No real-text test of the frozen router is reported. A 100,000-row corpus was built so that a stronger encoder could be trained without copying the evaluation questions. At the time of writing that training has no test metrics. Leaving the cell blank is part of the exercise.

## 1.4 Scope and Overview of Adaptive Temporal RAG

The system under evaluation is `checkpoint_b_final`. Documents are embedded with `BAAI/bge-small-en-v1.5`. Search is exact `IndexFlatIP`. Evaluation asks for 80 candidates, min–max normalises their FAISS scores inside the pool, leaves the cross-encoder score raw, and blends the two at 0.35 / 0.65. The top 40 blended values are kept, and temporal policy then selects $k = 5$ documents. The blend terms do not share a scale, and the weights were not chosen on a development split. The analyser is `MultitaskQueryAnalyzer` over `distilbert-base-uncased`, with `necessity_head`, `temporal_head`, and `confidence_head`. Training minimises necessity cross-entropy plus $\lambda = 1.75$ times temporal cross-entropy. The confidence head is absent from that loss. `resolve_adaptive_routing` in `src/adaptive_policy.py` skips retrieval only for the trivia whitelist. Otherwise it selects a current, historical, or semantic mode, and it falls back to semantic mode when $\gamma < 0.55$.

Dates are read by `extract_dates`, `parse_temporal`, and `extract_intervals` in `src/temporal_tagger.py`. A bare year is placed on 15 June. Intervals cover year–year ranges, “from … to …”, “until” or “through”, two-digit seasons, and “for $n$ years beginning in $y$” with $n \le 40$. A span longer than 45 years is discarded, which is aimed at birth–death pairs and will also discard a long reign. Distance $\delta(d, t)$ is zero when the asked date falls inside a stated range. Historical half-widths depend on the cue: 0.35 to 1.5 years for an exact calendar date, 1.5 to 3.5 for early, mid, or late in a year, 1.0 to 4.0 for a bare year, 5.0 to 8.0 for a decade, and 3.0 to 8.0 when the question is historical and names no year. While no survivor lies inside the window, the half-width grows by $h \leftarrow \min(h_{\max},\, 1.5h + 0.25)$. Current questions use a recency score against 1 October 2025 with denominator 3650, about ten years, and then sort a lexical tie by timestamp. Sufficiency $S$ mixes coverage, the top score, the mean score, temporal fit, and the lexical boost, and accepts evidence at $S \ge 0.55$ when the subject check passes. At most two repair rounds run. A budget of 60 seconds stops the loop and keeps the best evidence so far.

The frozen comparison covers three test files and six system rows. The templated corpus has 803 passages and 3,000 questions: 2,849 train, 74 validation, and 77 test. The held-out rebuild targets 1,200 questions and places every question that names one of six entities into a test split of 193. The names are Academy Award for Best Picture, CEO of Apple, CEO of Twitter/X, GDPR, President of Brazil, and UEFA Champions League winners. The TimeQA subset has 5,931 chunks and 900 questions, 400 train and 500 test. The rows are B1 (no retrieval), B2 (standard dense retrieval), B3 (fixed ±2 year filter), B4 (dense union BM25 with exponential decay), Adaptive (router, window, and sufficiency loop), and Adaptive-Fast (the saved sufficiency-off ablation, with latency from that later Adaptive-only run). A learned path, with gradient boosting and `deepset/tinyroberta-squad2`, remains behind the flag `learned_experimental` and is not a row in the frozen tables. The experimental router `microsoft/deberta-v3-base` is being trained on the 100,000-row leak-free set. This report states the data construction and the training specification. It states no quality metric for that model.

The scope has a short documented history. On 7–8 October 2026 a Colab T4 pilot locked B1, B2, B3, and Adaptive, with DistilBERT multitask heads. An automatic-mixed-precision binary-cross-entropy crash was removed by moving to a logits-safe loss. The same pilot found that the confidence head was outside the loss and emitted a near-constant 0.75. The first mid-term evaluation was on 8 October 2026. From 25 October to 17 November 2026 the pipeline was transferred onto TimeQA: surname-aware pooling, richer interval parsing, date-aware sentence selection, candidate caching, and a fix for SQuAD-reader abstention under the installed transformers version. The 17 November audit is the projection finding already noted. Commit 0db77f6 on 23 November froze Checkpoint B, including the shared reader, a maximum of three retrieves, and the 60-second timeout. Commit 44938a7 on 24 November added B4 and the paired tests. A router dataset of 50,001 rows was expanded to the 100,000-row set in `data/router_large/stats.json`. The report covers that frozen system and the unfinished router. It does not cover a live clock, a reimplementation of Adaptive-RAG, or a reproduction of published TempRALM.

## 1.5 Summary of Contributions

The contributions are stated at the strength the frozen tables support.

1. One Checkpoint B pipeline, loaded from `configs/final.yaml`, that separates a retrieval decision, a cue-dependent window, and a bounded sufficiency loop, while B1–B3 and Adaptive share `extractive_answer`, `project_answer`, the sentence selector, and `filter_subject`.
2. A date model in which stored timestamps and in-text intervals both enter $\delta(d, t)$, with hand-set half-widths, a 15 June midpoint, a 45-year lifespan discard, and explicit handling of ranges, seasons, and “before”.
3. A held-out result under that shared reader: exact match 0.762 versus 0.684 for B3, difference +0.078, interval [+0.041, +0.119], McNemar $p = 6.10\mathrm{e}{-05}$. Containment differs by +0.052, $p = 0.002$.
4. A null reading of the other two files, reported at the same rank: templated containment tied at 0.987 and exact match one question apart ($p = 1.000$); TimeQA containment versus B3 equal to +0.026 with interval [−0.004, +0.056] and $p = 0.117$.
5. A decay baseline, B4, showing that a smooth year penalty without a cutoff and without the cross-encoder stays behind the windowed systems on the templated and held-out exact-match scores (held-out difference +0.404, $p = 3.06\mathrm{e}{-15}$) and behind Adaptive on TimeQA containment (+0.062, $p = 5.39\mathrm{e}{-04}$). B4 latency is a different stack.
6. An ablation account. Removing the sentence selector lowers TimeQA containment by 0.128 ($p = 2.89\mathrm{e}{-08}$). Removing the entity cut lowers held-out exact match by 0.062 ($p = 4.88\mathrm{e}{-04}$). Removing the sufficiency loop does not lower answer scores, and Adaptive-Fast records the lower latency on its own later run.
7. An execution bound: the Janet Frame probe fell from 20,974.8 s to 8.184 s.
8. A 100,000-row router corpus with residual leakage 0 against 6,117 forbidden questions, 226 forbidden entities, and 335 forbidden templates, and with synthetic fraction 0.0. Training of DeBERTa-v3-base on that corpus is in progress and is not a measured contribution.
9. Two demonstration interfaces, which expose the policy on the three question types and do not replace the frozen tables.

The project does not claim a new state of the art on TimeQA, does not claim that the adaptive window beats a fixed window on every file, and does not claim a learned skip beyond the trivia whitelist.

## 1.6 Organisation of the Report

The report follows the minor-project structure expected for ARP 455 (Guru Gobind Singh Indraprastha University, 2026). Chapter 2 surveys retrieval-augmented generation, temporal retrieval and question answering, selective retrieval, dense and cross-encoder ranking, and the evaluation issues of calibration and leakage. It closes with a comparative matrix and with gaps tied to components of the frozen system. Chapter 3 states the limits of a query-agnostic pipeline and of a single fixed window, tabulates temporal failure modes, writes the distance and the answer metrics as they are computed, and records the anti-leakage rules. Chapter 4 describes the modules in `src/`, `scripts/`, and `app/`. Chapter 5 records the experimental design, the hardware and software, the data-flow and entity-relationship views, and the algorithms. Chapter 6 presents the snapshots, the master table, the ablations, and the paired tests. Chapter 7 summarises findings, limitations, and the unfinished DeBERTa router, for which no accuracy is yet available. Chapter 8 lists the references in APA 7th edition.

# CHAPTER 2: LITERATURE SURVEY

## 2.1 Foundational Retrieval-Augmented Generation

Lewis et al. (2020) formulate retrieval-augmented generation for knowledge-intensive tasks. A retriever proposes passages from a non-parametric index, and a parametric generator produces the output conditioned on those passages. The division matters for facts that change: the index can be rebuilt while the generator stays fixed. The usual pipeline that follows has one retrieval step, a fixed number of passages, and no variable for the time the question is about. A question about the present and a question about 2016 enter the same neighbour search. This project keeps the non-parametric index and the conditioning idea, and it adds the time variable outside the generator. Because the frozen generator is extractive, conditioning means that the reader may use only the documents the policy passes to it, or an empty list when the whitelist skip fires.

Karpukhin et al. (2020) replace a sparse first stage with dense passage retrieval. Question and passage encoders are trained so that inner product ranks a relevant passage above irrelevant ones. The frozen system is in that family only in a narrow sense. Passages are embedded once with a published checkpoint, the question is embedded at query time, and FAISS computes an exact inner product. No passage encoder is fine-tuned on the 803 templated passages or on the TimeQA chunks. Dense retrieval therefore supplies a semantic shortlist. It does not, by itself, know that 2018 and 2020 are different answers to the question of who held an office.

Izacard et al. (2023) take the opposite fork with Atlas. Retriever and reader are trained together, including in few-shot regimes, so a change in retrieval changes the reader. That design can absorb some errors of a hand-built ranker, and it makes an ablation harder to attribute. The present project freezes the reader, builds the ranker from published models plus hand-set gates, and keeps the experimental learned ranker out of the main table. The cost is that a bad constant stays bad. The blend weights, $\lambda$, $\alpha$, the window table, and the 0.55 threshold were set by hand. The benefit is that a difference between Adaptive and B3 can be read as a difference of window and loop, once the reader, the projection, and the entity cut are shared.

Izacard and Grave (2021) move the bottleneck to the reader. Fusion-in-Decoder encodes each retrieved passage with the question and lets the decoder attend across those encodings. Chen, Wang, and Wang (2021) evaluate that style of reader on TimeQA and report 60.5 exact match on the easy test and 46.8 on the hard test, with a human hard-test figure of 87.0. Those three numbers belong to that published evaluation. They are cited so that the 0.008 exact match of the frozen extractive reader on a 500-question chunk subset is not mistaken for a comparison with Fusion-in-Decoder. The local lesson is narrower. How the passages are read dominates exact match on time-sensitive prose, which is why the sentence selector is ablated on its own and why its removal moves TimeQA containment by 0.128. The gap this project works on is the missing time argument between the proposal of passages and the reader.

## 2.2 Temporal and Freshness-Aware Information Retrieval and Question Answering

Before retrieval-augmented generation, information retrieval already treated time as part of relevance. Li and Croft (2003) incorporate document age into a language model so that queries which seek recent documents do not depend only on a manually chosen boost. Their setting is recency: a document is more or less suitable because it is newer. Kanhabua and Nørvåg (2011) compare rankings that mix textual similarity with temporal similarity, on a news archive whose judgements are themselves scoped in time. The question they pose is the one this project inherits in heuristic form. What time is relevant, and how should that time combine with text match? The adaptive window is a descendant of that question. It is not an implementation of those mixture models, and this project does not evaluate on their collections.

The local descendant of that mixture is a gate, written in Chapter 3, and it is not one formula on every file. Tagged questions, untagged prose, and current questions take different branches. Temporal question answering then made the same issue visible at the level of answers. Jia et al. (2018) introduce TempQuestions, evaluated against a knowledge graph in which time can be stored as a qualifier. A prose corpus stores time as strings inside sentences and as one index timestamp that may not be the fact date. The tagger in `src/temporal_tagger.py` tries to recover a little of that structure: point dates, intervals, seasons, and a small set of relative phrases. It does not ground questions in a knowledge graph.

Chen, Wang, and Wang (2021) construct TimeQA by aligning time-evolving Wikidata facts with Wikipedia. Exact match is their metric, and the Fusion-in-Decoder figures above locate the difficulty of the published task. This project loads a subset from `hugosousa/TimeQA`: 5,931 chunks and 900 questions, 500 of them test. Whether an easy/hard field was present was not recorded, so the report claims neither an easy nor a hard split. Chunks are pieces of the sampled page, an easier target than open Wikipedia. Adaptive containment on that file is 0.296. The paired gain over B3 ($p = 0.117$) is unstable, the gain over B2 ($p = 0.006$) and over B4 ($p = 5.39\mathrm{e}{-04}$) is not, and recall at 5 is 0.488 against 0.392 for B3. A birth year or 1 January 1970 fails a ±2 test even when the sentence names the asked year, which is why B3 is a weak control and why the sentence selector matters on this file.

Zhang and Choi (2021) show with SituatedQA that one question string can have different answers once a time or another extra-linguistic context is supplied. The router’s three-way head is a coarse version of the time half of that idea: current, historical, or timeless. It does not represent place, and it does not predict an answer under a counterfactual date. SituatedQA enters this project as a source in the 100,000-row router corpus, 6,699 rows in the saved set, under a written labelling policy. It is not one of the three frozen question-answering tests.

Two further benchmarks study adaptation as text arrives over time. Liška et al. (2022) introduce StreamingQA. Kasai et al. (2023) introduce RealTime QA, which asks for the answer at the time of the question against a moving evidence stream. Both assume a clock that advances and a corpus that grows. The frozen evaluation uses a fixed corpus and the fixed reference day 1 October 2025. Current-question behaviour is measured on the templated examination built against that day. FreshQA, which includes fast-changing facts and false premises (Vu et al., 2023), was not run. Dhingra et al. (2022) take a closed-book route: a language model conditioned on a timestamp recalls a changing fact differently in different years. That result shows temporal knowledge can sit in parameters. The skip path here cannot be read as the same result. There is no timestamp-conditioned language model underneath `no_retrieval`. The timeless control is an empty document list plus hardcoded trivia.

Gade and Jetcheva (2024) bring temporality into retrieval-augmented models directly. TempRALM re-ranks retrieved passages with a temporal score and can mask passages dated after the query time. Masking is a hard constraint against the future of the question. Re-ranking is a soft preference for temporal agreement. B4 takes only the decay idea: the union of dense and BM25 candidates, an equal min–max blend, and a factor $\exp(-\lvert\Delta y\rvert/2)$, with no hard cutoff, no sufficiency loop, and no cross-encoder. The published TempRALM system is not in the tables. On these files the decay ranker sits near B2 (templated exact match 0.662 beside 0.649) and behind both windows on the held-out file (exact match 0.358 beside 0.762 and 0.684). A smooth penalty that never drops a candidate does not reproduce a year band. B4’s lower latency is partly the missing cross-encoder, so the speed is not a matched comparison.

The temporal literature therefore supplies the problem, the benchmarks, and two mechanisms this project simplifies: a mixture of text and time, and a decay or mask around the query time. The local mechanisms are a window on a parsed date, a recency score for “current”, and a sentence scorer that prefers date agreement. The simplifications are measured later, including where they fail.

## 2.3 Adaptive, Selective and Self-Reflective Retrieval

Always retrieving is a policy. It is the policy of B2 and of B3. On a test set where almost every question is about a dated fact, it is also nearly optimal as a binary decision. Adaptive retrieval rate is 1.000 on the held-out file and 1.000 on TimeQA. The only frozen file on which the router skips is the templated test, where the rate is 0.870, that is 67 of 77 questions. The other ten are the whitelist. Mean latency on that file does not fall below B3: 1.690 s for Adaptive and 1.555 s for B3. The literature on selective retrieval remains the right literature for the “when” half of the title, because the project asked the question even though the skip, as implemented, answers it only for trivia.

Asai et al. (2024) propose Self-RAG. The model learns reflection tokens that decide whether to retrieve, whether a passage is relevant and supportive, and whether a draft is useful. A skip in that framework can mean that the parameters already carry the answer, or that the passage does not support the sentence. A skip in the frozen pipeline means only that `needs_retrieval` is 0 and `is_trivial_query` returned true. Markers include “2 + 2”, “H2O”, “days are in a week”, “spell ”, and a few similar items. Self-RAG is not reimplemented. The shared phrase “retrieve or not” should not be read as a comparison of quality. The engineering reason for the whitelist is the shared reader: a reflection-token generator would have broken the comparison the freeze was built to protect.

Jeong et al. (2024) propose Adaptive-RAG, which classifies question complexity and chooses among no retrieval, a single retrieval, and iterative multi-step retrieval. The routing label is complexity, not time. A question about 2018 and a question about the current office-holder may have the same complexity and opposite temporal needs. Adaptive-RAG is the closest system in name. It is not a temporal ranker, and this project does not run it. The sufficiency loop is the local feature that most resembles iterative retrieval. If $S < 0.55$, the loop may enlarge the candidate pool, raise $k$, widen the window, or fall back toward semantic ranking. The resemblance ends at the measurement. On TimeQA, sufficiency off has containment 0.298 and the full system has 0.296, with exact match unchanged at 0.008 and McNemar $p = 1.000$. On the templated and held-out files the answer metrics match. The loop adds a slow tail. Held-out mean latency is 2.584 s and the median is 1.715 s; the gap is the loop on a minority of questions. That tail is not a complexity schedule, and it does not buy the answers such a schedule would be asked to justify.

Jiang et al. (2023) study active retrieval during generation. Low-confidence fragments of a draft trigger a further retrieval, after which generation continues. The decision is token-level and sits inside decoding. The sufficiency loop repairs a retrieval-level score before the extractive reader finishes, and it is capped at three retrieves. Both ideas say “retrieve again if unsure”, and they are unsure about different objects. Active retrieval of that kind does not consult $\delta(d, t)$. The local loop can widen $\mathcal{W}(q)$, which is a temporal repair, and the evaluation finds that this repair leaves the scored answers unchanged. That negative result belongs in the literature comparison. An active-retrieval idea, once instantiated as this score and these repairs, is not supported on the three frozen tests.

The selective-retrieval literature and this project meet at one implemented behaviour. `MultitaskQueryAnalyzer` produces a three-way temporal label and a binary necessity label, and the confidence head is left untrained. Learned critique, a complexity schedule, token-level active retrieval, and a skip that fires because a model knows the fact are all unimplemented. The 100,000-row corpus is an attempt to give the temporal and necessity labels a broader distribution than the 17,094 templated rows. Until the DeBERTa run finishes, that attempt is a dataset with an empty metric column.

## 2.4 Dense Embeddings, Lexical-Semantic Fusion and Cross-Encoder Reranking

The first-stage encoder is `BAAI/bge-small-en-v1.5` from C-Pack (Xiao et al., 2024). It is small enough that an exact index of 803 passages, or of 5,931 TimeQA chunks, fits the laptop evaluation. Queries use the prefix the model requires. The index type is FAISS `IndexFlatIP`, an exact inner-product scan (Johnson, Douze, & Jégou, 2019). At this scale an approximate index would add a recall error unrelated to time. A miss at rank 80 is a miss of the embedding. The depth 80 is a constant, not a tuned operating point.

A bi-encoder score is cheap and, on short office-holder passages, often ties documents that share an office name. Cross-encoders score the question and the passage together. Nogueira and Cho (2019) showed that a BERT cross-encoder is a strong passage reranker. The model used here is the MiniLM distillation (Wang et al., 2020) released as `cross-encoder/ms-marco-MiniLM-L-6-v2`, in the line of MS MARCO (Nguyen et al., 2016) and of sentence-level transformer embeddings (Reimers & Gurevych, 2019). `CrossEncoderReranker` scores the 80 candidates. The frozen blend is

$$
s_i = 0.35\,\tilde{s}^{\mathrm{f}}_i + 0.65\,s^{\mathrm{ce}}_i .
$$

The FAISS term is min–max normalised to $[0,1]$ inside the pool. The cross-encoder term is left raw. The coefficients 0.35 and 0.65 are therefore not the weights of a mixture of comparable quantities. A large positive cross-encoder score dominates the unit-interval dense score. The reported runs neither normalise the cross-encoder term nor choose the weights on a development split. The top 40 values of $s_i$ enter the temporal policy. B2, B3, and Adaptive use this blend. B4 does not. Any accuracy gap between B4 and B3 mixes a different temporal formula with a missing reranker. Templated B4 mean latency of 0.031 s and held-out mean latency of 0.040 s are real measurements of that lighter stack.

Lexical match remains, in two different forms that must not be collapsed. `BM25Index` is the lexical arm of B4 only. Inside B2, B3, and Adaptive the lexical signal is a boost. After a stop list, if $H$ remaining query tokens occur in the document, $b = \min(0.25,\, 0.05H)$. The stop list includes months, years, and words such as “between”, “held”, and “team”, because those tokens were matching the wrong passage. The boost is added inside the temporal gate and is an input to sufficiency. It is not BM25. Calling Adaptive a hybrid dense-plus-BM25 retriever would mis-state the frozen system.

The analyser beside this retriever is DistilBERT (Sanh et al., 2019), a distilled BERT (Devlin et al., 2019), identifier `distilbert-base-uncased`. Distillation matters here as the reason a small encoder was stable on the October T4 pilot. The heads are linear maps from the classification-token state: two necessity logits, three temporal logits, and one confidence logit. The experimental router replaces the encoder with DeBERTa-v3-base (He, Liu, Gao, & Chen, 2021), trained in fp32. Disentangled attention is the architectural reason to try a stronger encoder on prose less templated than the 17,094-row set. No accuracy, macro-F1, or expected calibration error from that run is available, and none is inferred. A smaller experimental ranker, used only on the learned path, is a gradient boosting machine (Friedman, 2001): 80 trees of depth 3, fit on 357 rows of which 143 are positive. The project paper records that a hand-built temporal prior, not the trees, carries most of the weight on that path, and that the path is excluded from the frozen tables. Its reader is `deepset/tinyroberta-squad2`. The main numbers belong to published bi-encoders and cross-encoders, plus rules.

## 2.5 Calibration, Leakage Control and Evaluation Methodology in Temporal QA

Rajpurkar, Jia, and Liang (2018) define SQuAD 2.0 so that a reader must extract a span and must also abstain when the passage does not contain an answer. The frozen metrics inherit the normalisation implemented as `normalize_text`: lower case, punctuation removed, whitespace collapsed. Exact match is 1 when the normalised strings are equal. Token F1 is the harmonic mean of token precision and token recall, taking the best gold string when several are listed. Containment is local to this project:

$$
\mathrm{Cont}(\hat{y}, y) =
\begin{cases}
1 & \text{if } y \subseteq \hat{y} \text{ or } \hat{y} \subseteq y,\\
0 & \text{otherwise.}
\end{cases}
$$

A short prediction inside a long gold string scores 1, and so does a sentence of up to 400 characters that merely contains the gold span. On TimeQA the heuristic answers are sentences, so containment mostly asks whether the selected sentence contains the gold text. Exact match is reported beside it for that reason. Unanswerable questions, or empty gold strings, replace the three scores by abstention accuracy. Abstention is an empty string or a short refusal such as “insufficient evidence”. The November transfer included a fix so that the experimental SQuAD reader’s abstention behaviour remained coherent under transformers 5.20.2. The frozen tables use the extractive reader.

Calibration enters because the policy trusts $\gamma$. The training loss is necessity cross-entropy plus $\lambda = 1.75$ times temporal cross-entropy, with class weights. The confidence logit is computed and then ignored. Nothing trains its weights. At inference, class logits are divided by a temperature before the softmax, and the blended scalar is half the sigmoid of the untrained logit plus half the larger maximum class probability. If $\gamma < 0.55$, the policy retrieves in semantic mode. An untrained head can therefore change which ranker runs. The project did not count how often $\gamma$ falls below 0.55 on the test sets. The pilot’s near-constant 0.75, and the demonstration values 0.751, 0.746, and 0.75, are consistent with a head of little input-dependent range. Expected calibration error is implemented on the large-router path and is not a number in this report.

Leakage control for the router corpus is fully measured. `LeakageIndex` drops questions that collide with the evaluation material: 6,117 forbidden questions, 226 entities, and 335 templates. The funnel labelled 313,314 rows, dropped 2,337 duplicates, then dropped 10,647 template collisions, 1,227 entity collisions, and 87 exact collisions, leaving 299,016. Class targets then selected 100,000 rows: 1,729 current, 49,135 historical, and 49,136 timeless, with synthetic fraction 0.0. The split is 79,982 train, 10,077 validation, and 9,941 test, seed 42. Residual leaks are 0. That statement is about the router file. The question-answering analyser was trained on the main templated file, 17,094 rows after six augmentations, batch size 8, with 7 of 15 configured epochs present in the log. The main file contains the six held-out names. Train and validation of the rebuilt held-out file do not. The split holds the names out of a question file while the router that runs at test time has already seen them. Almost every other component is a rule and was not trained on question labels.

Comparisons are paired. An exact two-sided McNemar test uses discordant questions. A 95% interval is a paired bootstrap percentile interval of Adaptive minus the other system, 10,000 resamples, seed 0. Questions with an empty gold-identifier list are omitted from precision and recall. Temporal accuracy looks only at the top timestamp: within three years of 1 October 2025 for a current question, and within two years of a year written in the question for a historical question. It does not read $\hat{y}$. It is undefined on TimeQA. For B3 the metric largely restates the filter, because every survivor of a non-empty ±2 band lies inside the band the metric checks. A temporal accuracy of 1.000 for B3, and the same figure for Adaptive, on the templated and held-out files is not answer quality.

Two further choices affect every later number. Constants were set by hand, and some TimeQA rules were chosen while looking at development failures, so there is no untouched development split for those choices. All frozen runs are on CPU. Latency that crosses the four-system run and the later ablation run is labelled as two clocks. Adaptive-Fast means come from the ablation run. The protocol is built to make answer metrics comparable, and to keep latency claims inside the run that produced them.

## 2.6 Comparative Literature Review Matrix

**Table 2.1:** Qualitative placement of fourteen related works. Cells describe mechanisms. They do not import numerical results. The TimeQA exact-match figures discussed above are cited only from Chen, Wang, and Wang (2021).

| Work (Year) | Venue | Core Mechanism | Temporal Awareness | Adaptive Routing | Calibration/Repair | Limitation w.r.t. this project |
|---|---|---|---|---|---|---|
| Lewis et al. (2020) | NeurIPS | Generator conditioned on retrieved passages | No time variable in the base pipeline | Retrieve as the default step | None in the base model | A nearest passage can be about another year |
| Karpukhin et al. (2020) | EMNLP | Dual-encoder dense passage retrieval | None | First stage only | None | This project does not fine-tune a dual encoder |
| Izacard et al. (2023) | JMLR | Joint training of retriever and reader (Atlas) | None | Few-shot retrieval | Training absorbs retriever error | The frozen reader is shared and is not trained with the index |
| Izacard and Grave (2021) | EACL | Fusion-in-Decoder over many passages | None | Fixed retrieval set | None | The extractive reader is not that decoder |
| Asai et al. (2024) | ICLR | Reflection tokens for retrieve and critique | None | Learned retrieve-or-skip and critique | Self-critique of drafts | The local skip is a trivia whitelist |
| Jeong et al. (2024) | NAACL | Route by question complexity | None | No retrieval, one retrieval, or several | Iterative retrieval | Not a temporal ranker, and not reimplemented |
| Jiang et al. (2023) | EMNLP | Active retrieval during generation | None | Retrieve when a span looks uncertain | Mid-generation repair | The local loop repairs a retrieval score and leaves answers unchanged |
| Li and Croft (2003) | CIKM | Document age inside a language model | Recency for time-seeking queries | None | None | Recency is not an arbitrary historical window |
| Kanhabua and Nørvåg (2011) | SIGIR | Mixture of text similarity and temporal similarity | Time-scoped rankings | None | None | The local window is a heuristic descendant, not those mixtures |
| Chen, Wang, and Wang (2021) | NeurIPS D&B | Time-evolving facts aligned to Wikipedia | Answers that change with time | None | None | The 500-question chunk subset is not their leaderboard |
| Zhang and Choi (2021) | EMNLP | Extra-linguistic context changes the answer | One string, different answers at different times | None | None | Router-corpus source, not a frozen test |
| Dhingra et al. (2022) | TACL | Language model conditioned on a timestamp | Closed-book recall depends on the year | None | None | The no-retrieval path has no such model |
| Vu et al. (2023) | arXiv | Search augmentation for fast-changing facts | Freshness, including false premises | Search when the evidence must be current | None | FreshQA was not run; the reference day is frozen |
| Gade and Jetcheva (2024) | arXiv | Temporal re-rank and masking of future text | Query-time temporal score | Re-rank rather than skip | Masking as a hard repair | B4 copies a decay formula only and omits the cross-encoder |

Temporal awareness and adaptive routing rarely appear in the same row. Where time is central, the system is a ranker, a benchmark, or a closed-book model. Where routing is central, the label is complexity, reflection, or token uncertainty, and the passage date is unscored. The frozen pipeline puts a small mark in both columns, and it leaves the calibration column weak on the evidence: the confidence head is untrained, and the repair loop does not improve the saved metrics.

## 2.7 Identified Research Gaps

Each gap below is tied to a component. Naming the component does not mean the gap is closed. Several components address a gap only in part, and the evaluation says so.

1. Standard retrieval-augmented generation has no argument for the time the question denotes (Lewis et al., 2020; Karpukhin et al., 2020). The components aimed at this gap are the three-way temporal head, $\mathcal{W}(q)$, and $\delta(d, t)$. On the held-out file the adaptive window is ahead of a fixed band. On the templated file, once the reader is shared, containment is tied. The gap is reduced on one file and remains open as a general claim.

2. Adaptive routers choose a budget from complexity or from self-critique, not from a date (Asai et al., 2024; Jeong et al., 2024; Jiang et al., 2023). `resolve_adaptive_routing` fires a skip on 10 of 77 templated questions and on none of the other tests. Learned abstention remains open. The 100,000-row corpus prepares a later measurement and is not that measurement.

3. Recency models and a single fixed band do not follow the specificity of the cue (Li & Croft, 2003). `infer_adaptive_window` uses the half-width table of Chapter 3. The widths were set by hand. Replacing the adaptive window by ±2 years leaves templated and held-out exact match unchanged and lowers TimeQA containment to 0.272, with paired $p = 0.134$. Cue-dependent width is a plausible mechanism whose answer-level effect, alone, is not a significant paired result.

4. A single stored timestamp is a poor proxy for the dates inside a document. TimeQA chunks often carry a birth year or 1970-01-01. `extract_intervals` and the sentence selector are the components aimed at this gap. B3, which filters on the stored year, is a weak control on that corpus. Temporal accuracy is left undefined there so that the metric cannot treat the index date as the fact date.

5. Iterative retrieval without a bound, and an unbounded fuzzy date parse, can dominate wall-clock time. `max_sufficiency_rounds = 2`, `query_timeout_s = 60`, and the long-passage cap in the tagger address the operational gap: the Janet Frame probe fell from 20,974.8 s to 8.184 s. They do not address answer quality. The sufficiency ablation is a negative result.

6. A confidence scalar that the policy trusts ought to be trained and checked. The frozen head is outside the loss. Expected calibration error is defined in the large-router code and has no reported value. DeBERTa-v3-base training on the leak-free corpus is the planned attack on this gap and, at the date of this report, is unfinished.

7. Answer-string post-processing can impersonate a retrieval gain. The shared call to `project_answer` and `extract_factoid_span`, after the 17 November audit, is what makes the frozen exact-match table a comparison of retrieval. Earlier comparisons that projected only Adaptive answers are not used.

8. Router training data can repeat evaluation questions by string, by entity, or by template. `LeakageIndex` leaves residual leaks at 0 on the 100,000-row file. On the question-answering side the six entity names are absent from the rebuilt train and validation questions and present in the analyser training file. The ranker is mostly rules, which were not fit to those labels. The held-out split is not described as a pure test of an unseen-entity neural retriever.

These eight gaps organise the problem statement and the later module descriptions. A component introduced against a gap is also the component whose ablation is read when the gap stays open.

# CHAPTER 3: PROBLEM STATEMENT

## 3.1 Limitations of Query-Agnostic and Fixed-Window RAG

A query-agnostic retriever scores a question against a document and ignores the time the question denotes. In the frozen stack that retriever is B2: always retrieve, rank by the dense-plus-cross-encoder blend, and pass $k = 5$ documents to the shared reader. The sentence selector is shared, so the reader may still prefer a sentence whose date agrees with the question, but the pool itself was not cut by a window. On the templated test this costs a large exact-match gap. B2 exact match is 0.649 and containment is 0.727; Adaptive is at 0.948 and 0.987. On the held-out file the gap is larger. B2 exact match is 0.332 and containment is 0.409; Adaptive is at 0.762 on both. Precision at 5 moves with the same pattern: 0.411 versus 0.668 on the templated test, and 0.346 versus 0.682 on the held-out test. Many passages concern the same entity in different years, so the semantic neighbourhood is full of true sentences about the wrong year. A query-agnostic pool can hand the sentence selector a list that does not contain the right year.

A fixed window is a strong correction when the stored timestamp is the fact date. B3 always retrieves. If a year is found, it keeps candidates whose stored year lies within ±2, and if that set is empty it keeps the unfiltered pool. It then calls the shared ranker without the time-first flag, and it runs no sufficiency loop. On the templated test B3 containment equals Adaptive at 0.987, that is 76 of 77 questions, and exact match is 0.935 against 0.948. The paired tests give $p = 1.000$ for both. Precision at 5 is 0.608 for B3 and 0.668 for Adaptive, so the pools differ, and the answer stage does not turn that difference into a stable exact-match gain. A ±2 year band is enough for this generator, which writes a fact date into $\tau_i$ and usually asks with a bare year. The fixed window is less adequate when the cue is finer or coarser than ±2, and when $\tau_i$ is the wrong date. An exact calendar date does not need a four-year band. A decade cue is a different request. A chunk whose index date is a birth year is harmed by ±2. On TimeQA, B3 recall at 5 is 0.392, B2 reaches 0.468, and Adaptive reaches 0.488. Containment is 0.270 for B3 and 0.296 for Adaptive, and the paired interval includes zero. The fixed window is the right baseline and, on noisy timestamps, a weak one.

B4 shows a third limitation. A smooth penalty $\exp(-\lvert\Delta y\rvert/2)$ on an equal dense-plus-BM25 blend, with no hard drop and no cross-encoder, does not recover the windowed answers. Templated exact match is 0.662. Held-out exact match is 0.358. TimeQA containment is 0.234, below B2. A penalty that never discards still spends the reader on distant years, and the missing cross-encoder removes a semantic filter the other retrieving systems have. Adding time in an arbitrary functional form does not solve the problem. The form has to be able to discard, and the discard has to be applied to a date that means the fact.

Two further limitations sit beside the ranker. B2 and B3 call the index on every question, including trivia whose answers are hardcoded. On the 77-question file the retrieval rate is 1 for those baselines and 0.870 for Adaptive, and the skip does not make Adaptive faster, because ten skipped questions do not offset the cross-encoder on the rest. Separately, a repair loop that may retrieve again, and a fuzzy date parser on a long passage, will eventually meet a question that keeps them busy. Before the cap, the Janet Frame probe took 20,974.8 s. The problem therefore includes a resource constraint. A policy that is slightly more accurate and occasionally unbounded is not an acceptable solution on this machine, and it is not a fair neighbour in a latency table.

## 3.2 Taxonomy of Temporal Failure Modes with Concrete Examples

The modes below are those the frozen system was built to recognise. The last column names the responsible component. Later chapters record whether the component succeeds. The sufficiency row is included because an unbounded repair is itself a failure, even though the repair does not improve answer metrics.

**Table 3.1:** Temporal failure modes, an example, the mechanism, and the component responsible for that mechanism.

| Failure mode | Example | Root cause | Component that addresses it |
|---|---|---|---|
| Recency error | “Who is the current Prime Minister of India?” | Semantic neighbours include earlier holders. “Current” needs a reference day. | $g^{\mathrm{cur}}$ with reference date 1 October 2025, then a latest-timestamp sort inside a lexical tie. Policy `current`. |
| Historical mismatch | “Who was the President of the United States in 2018?” | The same office has many true passages. Inner product does not read the year. | `infer_adaptive_window`: bare-year half-width 1.0 to 4.0, target 15 June, together with $\delta(d, t)$. |
| Over-retrieval | “What is 2 + 2?” | Always-retrieve calls the index for a stable fact. | `resolve_adaptive_routing` with `is_trivial_query`. The skip needs necessity 0 and a whitelist hit. B1 is the empty-context control. |
| Entity drift | A pool mixing several offices or people who share content words | Overlap on “president” or “award” keeps the wrong subject. | `filter_subject`. Templated documents with an entity field are cut to overlap within 0.01 of the best. Empty TimeQA fields use a surname rule. |
| Date-format and interval error | “2004–05”, “from 2014 to 2019”, “before March 2016”, “(born 1934)” | A year token is not an interval, and a birth year is not the asked fact. | `parse_temporal` and `extract_intervals`: seasons, from–to, until/through, $n \le 40$, a 45-year discard, and stripping of “(born …)”. |
| Timestamp noise | A TimeQA chunk stored as a birth year or as 1970-01-01 | A ±2 filter on $\tau$ drops the paragraph that names the asked year in the text. | $\delta$ consults in-text intervals and point dates as well as a usable timestamp. Temporal accuracy is undefined. B3 is a weak control. |
| Unbounded repair | “Janet Frame received which award in 1990?” | Repeated retrieval, or fuzzy parsing of a long passage, fails to terminate usefully. | At most three retrieves, a 60-second timeout, and the long-passage cap. The probe fell from 20,974.8 s to 8.184 s. |

The rows co-occur. The wrong person and the wrong year are often both in the shortlist. The entity cut runs on the historical branch before the window test, and in the frozen comparison it also runs for B2 and B3, so a precision gap is not an exclusive property of an Adaptive-only entity field. On TimeQA the field is empty. A document is kept when lexical overlap is at least 0.05 and the text contains a surname, unless overlap is within 0.01 of the best document. The surname is the last capitalised token of the first person-like span, ignoring months and question words, and it must be at least three characters. The rule is English-specific and was added while TimeQA failures were being inspected. Removing it drops held-out exact match from 0.762 to 0.699 and TimeQA containment from 0.296 to 0.252. The failure mode is real on these files. The rule was also shaped by development inspection, which Chapter 6 treats as a threat to validity.

The interval rules have a sharp edge chosen on purpose. A span longer than 45 years is discarded so that a birth–death pair does not contain every asked date in a long life. The same rule discards a long reign or a long career. “Before month year” becomes the interval from three years earlier through that month. `parse_temporal` handles “last year” and “since”, while a separate target is not given to every relative phrase a ranker might imagine. A decade cue is recognised when the tagger emits one. There is no general ordinal logic. The patterns written in the tagger are covered. The patterns that were not written remain open.

One templated miss is a property of $\mathcal{D}$. A test question asks who was president of Brazil in 2016, and the corpus has no passage for the holder in that year. Every ranker that stays inside 2016 misses it. A perfect window cannot retrieve a document that was never written. The hole is one reason templated containment is 0.987 rather than 1.

## 3.3 Formal Problem Formulation

The corpus is a finite set of dated documents

$$
\mathcal{D} = \{(d_i, \tau_i)\}.
$$

The timestamp $\tau_i$ is the value stored in the index. It is not the only date in $d_i$. On the templated corpus it is the fact date. On the TimeQA subset it is the first date the tagger finds, or 1 January 1970 when it finds none. A question $q$ may name a target date $t$. The system returns a string $\hat{y}$ and a bit recording whether retrieval ran.

The temporal class and the necessity bit are

$$
c \in \{\text{timeless}, \text{current}, \text{historical}\}, \qquad r \in \{0, 1\}.
$$

They are predictions of the analyser in the adaptive system and fields of the question item in the data. Retrieval is omitted only when $r = 0$ and `is_trivial_query`($q$) is true. In every other case the index is called, including a necessity prediction of 0 on a non-trivia question and the case $\gamma < 0.55$, which selects semantic mode. The historical window $\mathcal{W}(q)$ is the band of half-width $h$ about $t$. The half-width grows by

$$
h \leftarrow \min(h_{\max},\; 1.5h + 0.25)
$$

until a survivor satisfies $\delta(d, t) \le 365.25\, h$, or until $h$ reaches $h_{\max}$. An exact calendar date starts at 0.35 and stops at 1.5. Early, mid, or late in a year starts at 1.5 and stops at 3.5. A bare year starts at 1.0 and stops at 4.0. A decade starts at 5.0 and stops at 8.0. A historical question with no year starts at 3.0 and stops at 8.0. Current questions use, with reference date $r_{\mathrm{ref}}$ equal to 1 October 2025,

$$
g^{\mathrm{cur}}_i = \max\left(0,\; 1 - \frac{\max(r_{\mathrm{ref}}-\tau_i,\, 0)}{3650}\right).
$$

Let $I(d)$ be the intervals in a passage and $U(d)$ the point dates, including a usable timestamp. The distance is exactly

$$
\delta(d, t) =
\begin{cases}
0 & \text{if } t \in [u,v] \text{ for some } [u,v]\in I(d),\\[0.25em]
\min\limits_{x \in \partial I(d)\cup U(d)} |x-t| & \text{otherwise,}
\end{cases}
$$

where $\partial I(d)$ is the set of endpoints. Distance zero means the asked date falls inside a stated range. The historical score decreases with that distance,

$$
g^{\mathrm{hist}}_i
= \max\left(0,\;
1 - \frac{\delta(d_i, t)}{\max(90,\; 365.25\,w)}
\right),
$$

with $w$ the window width in years. The default gate, with lexical boost $b_i$ and $\alpha = 0.6$, is

$$
f_i = \alpha \tilde{s}_i + (1-\alpha)\,g_i\tilde{s}_i + b_i .
$$

For every system except B4 the pool is scored by

$$
s_i = 0.35\,\tilde{s}^{\mathrm{f}}_i + 0.65\,s^{\mathrm{ce}}_i
$$

from 80 candidates, of which 40 are kept and of which $k = 5$ are read. B4 replaces the blend by an equal min–max mixture of a dense score and a BM25 score, multiplied by $\exp(-\lvert\Delta y\rvert/2)$. Sufficiency, accepted at $0.55$ when the subject check passes, is

$$
\begin{aligned}
S =\;& 0.35\,C + 0.25\,\min(1, s_{(1)}) + 0.15\,\overline{s}\\
&+ 0.15\,T + 0.10\,\min(1, b_{(1)}/0.25).
\end{aligned}
$$

These formulae define the policy. They are not the output of a fit. The coefficients are constants.

Answer strings are normalised as in SQuAD. Exact match is 1 when the normalised prediction equals the normalised gold. Token F1 is the harmonic mean $2PR/(P+R)$ on token multisets, taking the best gold when several are listed. Containment is the bidirectional inclusion of Chapter 2. Precision at 5 and recall at 5 use stored gold document identifiers. On the templated data those identifiers are up to three passages of the named entity, preferring those nearest the asked date. On TimeQA a gold identifier is a chunk that contains an answer string, at most eight, and questions with an empty gold list are omitted from the average.

The evaluation objective, which this problem statement adopts, is to maximise answer correctness under a fairness constraint and a latency bound:

$$
\begin{aligned}
\text{maximise } & \mathrm{EM}(\hat{y}, y),\ \text{and secondarily containment and token F1}, \\
\text{subject to } & \text{one extractive reader, } \texttt{project\_answer},\ \text{one sentence selector, and one entity cut}, \\
& \text{at most three retrieves and 60 seconds per question}, \\
& \text{the skip rule stated above}.
\end{aligned}
$$

The maximisation is not a numerical search over the hand-set constants. It is the criterion by which B1, B2, B3, B4, Adaptive, and Adaptive-Fast are compared on three fixed tests, with paired intervals. Secondary reports are P@5, R@5, retrieval rate, and mean and median latency, each inside the run that produced it. Temporal accuracy is reported only where $\tau$ is a fact date, and even there it is not a substitute for exact match. A policy that raises exact match by changing the reader, or by running past the timeout, lies outside the feasible set.

The router’s training objective is narrower. It minimises the weighted sum of necessity and temporal cross-entropies, with $\lambda = 1.75$, and it does not include exact match. The confidence head is outside that loss. A high class probability on the word “current” can coexist with an untrained $\gamma$. The analyser is a labelling front-end. The quantity the evaluation claims to care about is the answer, under the constraints just listed.

## 3.4 Operational and Research Objectives

The objectives are numbered so that later chapters can refer to them. Where a measurement fails an objective, the result is attached in the same item.

1. Run B1, B2, B3, B4, and Adaptive from `checkpoint_b_final`, and recover Adaptive-Fast from the sufficiency-off ablation.
2. Skip retrieval only when necessity is 0 and `is_trivial_query` matches; otherwise retrieve. The measured skip is 10 of 77 templated questions, and the retrieval rate is 1 on the other two tests.
3. Choose $\mathcal{W}(q)$ from the cue table and the growth rule, and score documents with $\delta$, $g^{\mathrm{cur}}$, or $g^{\mathrm{hist}}$ as the class requires.
4. Keep the answer stage fair: one extractive `Generator`, one `project_answer`, one sentence selector, and one `filter_subject` for B1, B2, B3, and Adaptive.
5. Bound every question by three retrieves and 60 seconds, and bound fuzzy date parsing on long passages. The acceptance check is the Janet Frame probe at 8.184 s, against 20,974.8 s before the guard.
6. Measure exact match, token F1, containment, P@5, R@5, temporal accuracy where it is defined, retrieval rate, and mean and median latency on the full tests of 77, 193, and 500 questions.
7. Compare systems with exact McNemar tests and 10,000-resample bootstrap intervals, seed 0, and report intervals that include zero with the same prominence as intervals that exclude it. This item is what requires the TimeQA-versus-B3 containment result ($p = 0.117$) to be described as unstable.
8. Ablate the router, the adaptive window, the sufficiency loop, the entity cut, and the sentence selector, one at a time, on all three tests, and publish a null sufficiency result if that is what the rows show.
9. Build a leak-free router corpus of 100,000 rows, synthetic fraction 0, residual leaks 0, as training material for `microsoft/deberta-v3-base`. Training may be started. Accuracy, macro-F1, and ECE are not reported before a finished run.
10. Expose the frozen policy in a local demonstration and a hosted interface, for a current question, a historical question, and a timeless question. The interfaces are operational. They are not an extra test set.

Objectives 2 and 8 deserve a special reading. The title asks when to retrieve, and objective 2 is implemented. Objective 8 shows that, once trivia items are set aside, turning the router off does not change exact match, because the retrieval rate was already 1. The scientific objective is to measure the router where it skips, and to avoid crediting it where it does not.

## 3.5 Fair-Evaluation Protocol and Anti-Leakage Constraints

The protocol has three layers: a shared answer stage, a frozen file of switches, and two different ways of holding data out.

The shared answer stage is what makes exact match comparable. `scripts/evaluate.py` builds one extractive `Generator` and reads `enable_sentence_selector`, `enable_entity_cut`, and `enable_answer_projection` from `configs/final.yaml`. In the frozen file all three are true. B1 differs from Adaptive by receiving no documents, not by shortening its string with a different function. B2 and B3 differ by the window and by the absence of the sufficiency loop. The 17 November 2026 audit is why this layer exists. Before it, only Adaptive applied `src/answer_projection.py`. Those earlier files are not the tables used here. On TimeQA, `span_extract_timeqa` is true as well: exact match and token F1 use `extract_factoid_span` on the selected sentence, while containment still uses the sentence. The split is uniform. It is why exact match can sit at 0.008 while containment sits at 0.296 without either figure being an aggregator error.

The switches that may differ are in the same YAML file. `always_retrieve` is false for Adaptive, so the router is on. `enable_adaptive_window` and `enable_sufficiency_loop` are true. `candidate_k` is 80, `ce_rerank_keep` is 40, `top_k` is 5, `alpha` is 0.6, both thresholds are 0.55, `max_sufficiency_rounds` is 2, and `query_timeout_s` is 60. An ablation flips one flag. It does not rebuild the index or change the reader. Latency from an ablation run is reported as Adaptive-Fast or as an ablation mean. It is not substituted for the four-system mean. B4 is `TempRALMDecayPipeline`, a different first stage, so the fairness claim for B4 is weaker and is restated whenever B4 is discussed: the same reader and projection, a different retrieval stack, and no cross-encoder.

The held-out question file rebuilds the templated generator with a target of 1,200 questions. Every fifth entity in a sorted list is withheld from that file’s train and validation splits, and every question naming one of those entities is placed in test. The six names are Academy Award for Best Picture, CEO of Apple, CEO of Twitter/X, GDPR, President of Brazil, and UEFA Champions League winners. Test size is 193 because those questions are dumped into test, not because a 15% sample was drawn. This construction stops the rule-based ranker from being scored only on entities that the same file used while patterns were drafted. It does not retrain the analyser. The analyser remains the one fit on the main file, which contains the six names. Validation necessity and temporal accuracy of 1.0 say that the cues in the templated validation split are fitted. They do not say that the six names were unseen to the encoder. The components that move held-out exact match in the ablations are the entity cut and the windowed retrieval, not a freshly trained dual encoder.

The router corpus uses a stricter filter because it will train a labeller. The forbidden material comprises 6,117 questions, 226 entities, and 335 templates, including 6,075 official TimeQA test questions added to the forbidden list. After deduplication and the three leak drops, residual leaks on the 100,000 selected rows are 0, and synthetic rows selected are 0. The labelling policy is part of the constraint. An explicit year maps to historical when the latest year is before reference year 2026 minus one, and to current otherwise. Explicit current cues, present-tense changing offices, original temporally dependent SituatedQA questions, and fresh or realtime sources map to current. TimeQA and other temporal sets default to historical. General sources without a time cue, including Natural Questions, TriviaQA, SQuAD v2, ARC, CommonsenseQA, OpenBookQA, SciQ, and QASC, map to timeless with `needs_retrieval` 0. Current and historical labels use `needs_retrieval` 1. Only 1,729 current rows are available, so current is the scarce class, while historical and timeless are downsampled to 49,135 and 49,136. A model trained on this file would see far fewer current questions than timeless or historical ones. That imbalance is a property of the sources after the rules, and it will matter when metrics exist. They do not exist yet. The Kaggle specification, fp32 on one device, five epochs, patience 2, learning rate $2\times 10^{-5}$, batch size 16, and maximum length 128, is recorded so the run can be finished. It is not a placeholder accuracy.

Two smaller constraints keep the metrics from leaking a false conclusion. Temporal accuracy is not computed on TimeQA. Containment is not presented as exact match. The missing 2016 Brazil passage is treated as a hole in $\mathcal{D}$, so a miss on that question is not charged to the window as if the document had been ranked away. Under those constraints the rest of the report asks for better exact match where the index date means the fact, an honest interval where it does not, a shared reader throughout, and a hard stop at three retrieves or 60 seconds.
