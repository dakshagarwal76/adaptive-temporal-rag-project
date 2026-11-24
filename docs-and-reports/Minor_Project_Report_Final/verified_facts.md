# Verified Fact Sheet — Adaptive Temporal RAG (commit 44938a7)

All numbers below are copied from the repository at commit `44938a7` (`artifacts/final_results.json`, `artifacts/significance.json`, `artifacts/eval_timings.json`, `configs/final.yaml`, `docs/paper/adaptive_temporal_rag.tex`) and `data/router_large/stats.json`. Do not use any number that is not on this sheet or in those files.

## Team / metadata
- **student**: Daksh Agarwal
- **enrollment**: 10819051623
- **partner**: Shashank Kumar (08619051623)
- **batch**: AIML-B2
- **programme**: B.Tech. 7th Semester, AI-ML
- **supervisor**: Dr. Annu Priya, Assistant Professor, USAR
- **institution**: University School of Automation and Robotics, GGSIPU East Delhi Campus, Surajmal Vihar, Delhi - 110032
- **course**: Minor Project (ARP 455)
- **title**: When and What to Retrieve: Adaptive Temporal Retrieval-Augmented Generation for Time-Aware Question Answering (Adaptive Temporal RAG)
- **evaluation**: 25 November 2026, 10:00 AM IST

## Frozen configuration (`configs/final.yaml`)
```yaml
# Frozen Checkpoint B system. evaluate.py loads this file.
# Differences between B1/B2/B3/Adaptive: retrieval decision, time window, sufficiency loop.
name: checkpoint_b_final
generator: extractive
top_k: 5
candidate_k: 80
ce_rerank_keep: 40
alpha: 0.6
conf_threshold: 0.55
sufficiency_threshold: 0.55
max_sufficiency_rounds: 2   # initial retrieve + up to 2 repairs = 3 total
query_timeout_s: 60.0
enable_ce_rerank: true
enable_sufficiency_loop: true
enable_adaptive_window: true
enable_entity_cut: true
enable_sentence_selector: true
enable_answer_projection: true
always_retrieve: false      # router on
span_extract_timeqa: true
learned_experimental: true  # not in the main four-system table
```

## Models
- **embedder**: BAAI/bge-small-en-v1.5
- **faiss**: IndexFlatIP (exact)
- **cross_encoder**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **frozen_analyzer**: distilbert-base-uncased (multitask: necessity 2-way, temporal 3-way, confidence head untrained; lambda=1.75; blended gamma threshold 0.55)
- **learned_path_reader**: deepset/tinyroberta-squad2 (experimental, not in tables)
- **large_router**: microsoft/deberta-v3-base (100k leak-free data; Kaggle T4 training in progress, no metrics yet)

## Method constants (paper Sec. 4)
- **ce_blend**: s_i = 0.35*minmax(faiss) + 0.65*ce (raw)
- **candidates**: 80
- **ce_keep**: 40
- **top_k**: 5
- **alpha**: 0.6
- **window_growth**: h <- min(h_max, 1.5h + 0.25)
- **window_table_years**: {'exact calendar date': [0.35, 1.5], 'early/mid/late in a year': [1.5, 3.5], 'bare year': [1.0, 4.0], 'decade': [5.0, 8.0], 'historical, no year': [3.0, 8.0]}
- **current_recency**: g_cur = max(0, 1 - max(r - tau,0)/3650), r = 1 Oct 2025
- **sufficiency**: S = 0.35C + 0.25 min(1,s1) + 0.15 mean_s + 0.15T + 0.10 min(1,b1/0.25); accept if S>=0.55
- **lexical_boost**: b = min(0.25, 0.05H)
- **b4_decay**: equal blend dense+BM25 (minmax) x exp(-|dyear|/2)
- **guardrails**: max_sufficiency_rounds=2 (3 retrieves total), query_timeout_s=60, dateutil long-passage cap
- **skip_rule**: retrieval skipped only when necessity=0 AND query matches fixed trivia whitelist

## Datasets
- **synthetic_main**: 803 passages, 3,000 questions: 2,849 train / 74 val / 77 test
- **heldout**: 1,200-question rebuild; test n=193 = all questions naming 6 held-out entities (Academy Award for Best Picture, CEO of Apple, CEO of Twitter/X, GDPR, President of Brazil, UEFA Champions League winners)
- **timeqa**: hugosousa/TimeQA subset: 5,931 chunks, 900 questions (400 train / 500 test); gold = chunks containing answer, max 8
- **analyzer_training**: 17,094 rows (templated + 6 augmentations), batch 8, 7 epochs logged of 15 configured; val necessity & temporal acc 1.0

## Master table (EM / F1 / Contains / P@5 / R@5 / Temp / Ret / mean s / median s)

| Dataset | System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic | B1 no retrieval | 77 | 0.130 | 0.167 | 0.130 | 0.000 | 0.000 | 0.247 | 0.000 | 0.000 | 0.000 |
| synthetic | B2 standard | 77 | 0.649 | 0.676 | 0.727 | 0.411 | 0.655 | 0.662 | 1.000 | 1.551 | 1.432 |
| synthetic | B3 fixed ±2y | 77 | 0.935 | 0.951 | 0.987 | 0.608 | 0.943 | 1.000 | 1.000 | 1.555 | 1.452 |
| synthetic | B4 TempRALM decay | 77 | 0.662 | 0.684 | 0.727 | 0.435 | 0.695 | 0.675 | 1.000 | 0.031 | 0.030 |
| synthetic | Adaptive | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.690 | 1.573 |
| synthetic | Adaptive-Fast (sufficiency off) | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.179 | 1.290 |
| heldout | B1 no retrieval | 193 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | B2 standard | 193 | 0.332 | 0.361 | 0.409 | 0.346 | 0.577 | 0.596 | 1.000 | 1.569 | 1.489 |
| heldout | B3 fixed ±2y | 193 | 0.684 | 0.696 | 0.710 | 0.591 | 0.879 | 1.000 | 1.000 | 1.546 | 1.410 |
| heldout | B4 TempRALM decay | 193 | 0.358 | 0.369 | 0.383 | 0.383 | 0.639 | 0.497 | 1.000 | 0.040 | 0.038 |
| heldout | Adaptive | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 2.584 | 1.715 |
| heldout | Adaptive-Fast (sufficiency off) | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 0.663 | 0.649 |
| timeqa | B1 no retrieval | 500 | 0.000 | 0.013 | 0.014 | 0.000 | 0.000 | — | 0.000 | 0.000 | 0.000 |
| timeqa | B2 standard | 500 | 0.006 | 0.077 | 0.260 | 0.235 | 0.468 | — | 1.000 | 6.056 | 6.118 |
| timeqa | B3 fixed ±2y | 500 | 0.010 | 0.089 | 0.270 | 0.237 | 0.392 | — | 1.000 | 6.074 | 6.116 |
| timeqa | B4 TempRALM decay | 500 | 0.010 | 0.084 | 0.234 | 0.174 | 0.358 | — | 1.000 | 0.062 | 0.059 |
| timeqa | Adaptive | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 8.736 | 6.492 |
| timeqa | Adaptive-Fast (sufficiency off) | 500 | 0.008 | 0.088 | 0.298 | 0.243 | 0.491 | — | 1.000 | 3.184 | 3.160 |

## Paired significance (Adaptive − other; exact McNemar; 10,000-resample bootstrap, seed 0)

| Dataset | Comparison | Metric | A−B | 95% CI | McNemar p | only A | only B |
|---|---|---|---:|---|---:|---:|---:|
| synthetic | adaptive vs B2_standard_rag | em | +0.299 | [+0.208, +0.403] | 2.38e-07 | 23 | 0 |
| synthetic | adaptive vs B2_standard_rag | contains | +0.260 | [+0.169, +0.364] | 1.91e-06 | 20 | 0 |
| synthetic | adaptive vs B3_fixed_temporal | em | +0.013 | [+0.000, +0.039] | 1.000 | 1 | 0 |
| synthetic | adaptive vs B3_fixed_temporal | contains | +0.000 | [+0.000, +0.000] | 1.000 | 0 | 0 |
| synthetic | adaptive vs tempralm_decay | em | +0.286 | [+0.182, +0.390] | 2.98e-06 | 23 | 1 |
| synthetic | adaptive vs tempralm_decay | contains | +0.260 | [+0.156, +0.364] | 1.10e-05 | 21 | 1 |
| heldout | adaptive vs B2_standard_rag | em | +0.430 | [+0.363, +0.498] | 4.00e-19 | 83 | 0 |
| heldout | adaptive vs B2_standard_rag | contains | +0.352 | [+0.285, +0.420] | 4.68e-19 | 68 | 0 |
| heldout | adaptive vs B3_fixed_temporal | em | +0.078 | [+0.041, +0.119] | 6.10e-05 | 15 | 0 |
| heldout | adaptive vs B3_fixed_temporal | contains | +0.052 | [+0.021, +0.083] | 0.002 | 10 | 0 |
| heldout | adaptive vs tempralm_decay | em | +0.404 | [+0.316, +0.492] | 3.06e-15 | 92 | 14 |
| heldout | adaptive vs tempralm_decay | contains | +0.378 | [+0.290, +0.466] | 4.80e-14 | 87 | 14 |
| heldout | adaptive vs entity_cut_off | em | +0.062 | [+0.031, +0.098] | 4.88e-04 | 12 | 0 |
| heldout | adaptive vs entity_cut_off | contains | +0.062 | [+0.031, +0.098] | 4.88e-04 | 12 | 0 |
| timeqa | adaptive vs B2_standard_rag | em | +0.002 | [+0.000, +0.006] | 1.000 | 1 | 0 |
| timeqa | adaptive vs B2_standard_rag | contains | +0.036 | [+0.012, +0.060] | 0.006 | 29 | 11 |
| timeqa | adaptive vs B3_fixed_temporal | em | -0.002 | [-0.010, +0.004] | 1.000 | 1 | 2 |
| timeqa | adaptive vs B3_fixed_temporal | contains | +0.026 | [-0.004, +0.056] | 0.117 | 36 | 23 |
| timeqa | adaptive vs tempralm_decay | em | -0.002 | [-0.010, +0.004] | 1.000 | 1 | 2 |
| timeqa | adaptive vs tempralm_decay | contains | +0.062 | [+0.028, +0.096] | 5.39e-04 | 54 | 23 |
| timeqa | adaptive vs sentence_selector_off | em | +0.006 | [+0.000, +0.014] | 0.250 | 3 | 0 |
| timeqa | adaptive vs sentence_selector_off | contains | +0.128 | [+0.084, +0.172] | 2.89e-08 | 99 | 35 |
| timeqa | adaptive vs entity_cut_off | em | +0.002 | [+0.000, +0.006] | 1.000 | 1 | 0 |
| timeqa | adaptive vs entity_cut_off | contains | +0.044 | [+0.024, +0.066] | 1.13e-04 | 27 | 5 |
| timeqa | adaptive vs window_off | em | -0.002 | [-0.010, +0.004] | 1.000 | 1 | 2 |
| timeqa | adaptive vs window_off | contains | +0.024 | [-0.004, +0.052] | 0.134 | 33 | 21 |
| timeqa | adaptive vs sufficiency_off | em | +0.000 | [+0.000, +0.000] | 1.000 | 0 | 0 |
| timeqa | adaptive vs sufficiency_off | contains | -0.002 | [-0.006, +0.000] | 1.000 | 0 | 1 |

## Eval runtime (`artifacts/eval_timings.json`)
- synthetic_seconds: 407.2253013
- heldout_seconds: 1125.3883383
- timeqa_seconds: 10466.7595792
- ablations_seconds: 13767.4655516
- b4_seconds: 75.0132418
- notes: Four-system frozen Checkpoint B eval via scripts/evaluate.py --final --config configs/final.yaml. Janet Frame hang probe: 8.184s.

## Hang fix
- Janet Frame probe ('Janet Frame received which award in 1990?'): 20974.8 s before → 8.184 s after (max 3 retrieves, 60 s timeout, dateutil long-passage cap)

## Smoke test
- scripts/e2e_smoke_test.py: PASS=72 FAIL=0 WARN=2 (missing OpenRouter and OpenCode keys)

## Papers
- Full paper 15 pages; workshop paper 4 pages; tectonic exit 0.

## 100k leak-free router dataset (`data/router_large/stats.json`)
```json
{
 "reference_year": 2026,
 "seed": 42,
 "total": 100000,
 "by_split": {
  "train": 79982,
  "val": 10077,
  "test": 9941
 },
 "by_source": {
  "triviaqa": 15827,
  "nq_open": 26701,
  "squad_v2": 22858,
  "commonsense_qa": 2535,
  "sciq": 3174,
  "arc": 1906,
  "openbookqa": 1407,
  "qasc": 2013,
  "timeqa": 16880,
  "situatedqa": 6699
 },
 "by_temporal_class": {
  "current": 1729,
  "historical": 49135,
  "timeless": 49136
 },
 "by_needs_retrieval": {
  "1": 50864,
  "0": 49136
 },
 "by_split_and_class": {
  "train": {
   "current": 1383,
   "historical": 39291,
   "timeless": 39308
  },
  "val": {
   "current": 173,
   "historical": 4990,
   "timeless": 4914
  },
  "test": {
   "current": 173,
   "historical": 4854,
   "timeless": 4914
  }
 },
 "synthetic_rows": 0,
 "synthetic_fraction": 0.0,
 "constraints": {
  "min_rows": true,
  "synthetic_cap": true,
  "no_residual_leaks": true
 },
 "labeling_policy": "Explicit years map to historical when the latest year is before reference_year-1, else current. Explicit current cues, present-tense changing offices, SituatedQA original temporally dependent questions, and fresh/realtime sources map to current. TimeQA and other temporal sets default to historical. General QA without a time cue (NQ, TriviaQA, SQuAD v2, ARC, commonsense, OpenBookQA, SciQ, QASC) maps to timeless and needs_retrieval=0. Current and historical labels use needs_retrieval=1. Test of Time rows, when loaded, are marked synthetic and share the 20% cap."
}
```
Funnel:
```json
{
 "loaded_by_source": {
  "timeqa": 35097,
  "situatedqa": 12227,
  "nq_open": 87925,
  "triviaqa": 50000,
  "squad_v2": 80000,
  "commonsense_qa": 10962,
  "openbookqa": 5844,
  "sciq": 13678,
  "qasc": 9040,
  "arc": 7729,
  "timequestions": 0,
  "tempquestions": 0,
  "freshqa": 0,
  "realtimeqa": 0,
  "mintaka": 0,
  "tot": 0,
  "temporalwiki": 0,
  "synthetic": 812
 },
 "labeled_by_source": {
  "timeqa": 35097,
  "situatedqa": 12227,
  "nq_open": 87925,
  "triviaqa": 50000,
  "squad_v2": 80000,
  "commonsense_qa": 10962,
  "openbookqa": 5844,
  "sciq": 13678,
  "qasc": 9040,
  "arc": 7729,
  "synthetic_queries": 406,
  "synthetic_heldout": 406
 },
 "labeled_total": 313314,
 "after_dedup": 310977,
 "dedup_dropped": 2337,
 "leak_dropped": {
  "template": 10647,
  "entity": 1227,
  "exact": 87
 },
 "after_leak_filter": 299016,
 "after_leak_filter_by_class": {
  "current": 1729,
  "historical": 89620,
  "timeless": 207667
 },
 "balance": {
  "targets": {
   "current": 1729,
   "historical": 49135,
   "timeless": 49136
  },
  "available": {
   "current": 1729,
   "historical": 89620,
   "timeless": 207667
  },
  "real_available": {
   "current": 1729,
   "historical": 89619,
   "timeless": 207663
  },
  "synthetic_available": {
   "current": 0,
   "historical": 1,
   "timeless": 4
  },
  "selected": {
   "current": 1729,
   "historical": 49135,
   "timeless": 49136
  },
  "synthetic_selected": 0,
  "synthetic_fraction": 0.0
 }
}
```
Leakage: {'forbidden_question_count': 6117, 'forbidden_entity_count': 226, 'forbidden_template_count': 335, 'timeqa_official_test_questions_added': 6075, 'residual_leaks': 0}

## Kaggle training run (in progress — no metrics yet)
```json
{
 "note": "from the Kaggle notebook launch (parent session), not repo; run in progress, no metrics",
 "model": "microsoft/deberta-v3-base",
 "precision": "fp32 single cuda:0, DataParallel off",
 "epochs": 5,
 "patience": 2,
 "lr": 2e-05,
 "batch_size": 16,
 "max_length": 128,
 "steps_per_epoch": 4999,
 "checkpoint_every_steps": 500,
 "accelerator": "Kaggle GPU T4 x2 (one used)",
 "notebook": "atrag-router-large-100k-training (private)"
}
```

## Hardware
- Dell Latitude 5420, Intel Core i7-1165G7 @ 2.80GHz (4C/8T), 32 GB RAM (33,723,650,048 bytes), Intel Iris Xe Graphics, Windows 11 Pro; evals CPU-only
- Kaggle GPU T4 x2 for the 100k router run.

## Software (repo .venv)
- python: 3.11.9
- torch: 2.14.1
- transformers: 5.20.2
- sentence-transformers: 6.2.0
- faiss-cpu: 1.15.1
- streamlit: 1.66.1
- plotly: 7.1.0
- numpy: 2.4.6
- pandas: 3.1.0
- scipy: 1.17.1
- python-dateutil: 3.0.0
- PyYAML: 6.0.3
- web_ui: Next.js 15.5 / React 19 (web/, lexical TS port of policy)

## Live UI observations (captured 24 Nov 2026, 3:05 AM IST)
- **historical**: Who was the President of the United States in 2018? -> Donald Trump; policy historical_adaptive_window; target_year 2018; initial/final window 1y, max 4, expansions 0; sufficiency 0.98; latency 2.629s; top doc 2018-04-16 score=1.046
- **current**: Who is the current Prime Minister of India? -> Narendra Modi; policy current; confidence 0.746; sufficiency 0.905; latency 0.621s
- **timeless**: What is 2 + 2? -> 4; policy no_retrieval; needs_retrieval 0; latency 0.065s; No retrieval

## Important accuracy caveats
- The frozen Adaptive system uses dense FAISS (BGE-small) + cross-encoder rerank. BM25 is used ONLY in B4 (dense ∪ BM25 + exp decay). Do not call Adaptive a hybrid dense+BM25 retriever.
- The frozen router is DistilBERT; DeBERTa-v3-base 100k router is experimental / training in progress and not in any result table.
- Confidence head is untrained. Learned path is experimental and excluded.
- B4 does not run the cross-encoder, so its latency is not matched. Adaptive-Fast latencies come from the later Adaptive-only ablation run.
- TimeQA Contains scores a selected sentence; EM uses span extraction and stays near zero.
- Temporal accuracy undefined on TimeQA. Reference date 1 Oct 2025.
- Evals ran on laptop CPU; no GPU.