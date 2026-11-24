# arXiv submission metadata

Paste these fields into the arXiv form. The abstract below is plain text for that form. The PDF abstract is the same wording with math notation.

## Title

Adaptive Temporal Retrieval for Time-Sensitive Question Answering

## Authors

Daksh Agarwal and Shashank Kumar

Affiliation (both authors): University School of Automation and Robotics (USAR), Guru Gobind Singh Indraprastha University

Dr. Annu Priya is the supervisor and is acknowledged in the paper. Do not list Dr. Annu Priya as an author.

## Abstract

The pipeline can label a question as present, past, or neither, keep a date window around the asked time, and read a sentence whose date agrees with the question. This paper reports what the saved runs change, including the full tests dated 18 November 2026, and where the measurements stop.

On the templated test of 77 questions, adaptive exact match is 73 of 77 and a fixed +/- 2 year filter (B3) is 35 of 77. Containment is 76 of 77 against 75 of 77. On 37 of the 38 questions that only the adaptive row matches, B3's prediction already contains the gold string and the adaptive string. That exact-match gap is answer projection, which B2 and B3 do not run. Precision at 5 is 0.650 against 0.585. Skipping retrieval fires on 10 of those 77 questions, only when a classifier says "no" and the text hits a hardcoded list of trivia. It does not make the system faster: adaptive mean latency is 1.491 seconds and the median is 1.560, against 1.380 and 1.305 for B3.

On 193 templated questions about six entity names, adaptive exact match is 147 of 193 against 49 of 193 for B3, and containment is 147 of 193 against 137 of 193. Of the 98 exact-match gains, 88 are questions whose B3 string already contains the gold answer. Retrieval is never skipped. The method still gives the adaptive system an entity field the baselines do not, so this is not a pure comparison of date windows. Mean latency is 0.905 seconds against 0.709; the medians are 0.779 and 0.668.

The 500-question TimeQA run was completed. Exact match is 0 for B2, B3, and the adaptive system. Containment is 148 of 500, 129 of 500, and 96 of 500. Median latencies are 3.730, 3.541, and 3.585 seconds. The adaptive mean is 47.486 seconds because one query took 20974.8 seconds; the mean of the other 499 is 5.547 seconds. A 16-question slice is not this result. There is still no ablation and no comparison with a published temporal retriever.

## Categories

- Primary: cs.CL
- Cross-list: cs.IR

## Comments

17 pages, 5 tables, 0 figures

## MSC / ACM (optional)

- MSC: 68T50
- ACM: H.3.3; I.2.7

## License suggestion

CC BY 4.0 is a reasonable choice for a preprint. Confirm with Dr. Annu Priya before selecting it. arXiv also offers a non-exclusive license to distribute; that choice is separate from the public copyright license. This draft does not embed a license command in `main.tex`.

## Source package

`arxiv_source.zip` contains `main.tex` only. The bibliography is an embedded `thebibliography` environment, so no `.bbl` is generated. The paper includes no figures, so there is no `figures/` directory. Do not upload `main.pdf`, `main.aux`, `main.log`, `main.out`, or `main.xdv`.

## Pre-submission checklist

These are the weak points still in the draft. They are stated in the paper. Do not sand them off in the arXiv comments or in a talk.

- Adaptive exact-match gains on the office full tests come mainly from answer projection that B2 and B3 do not run. On the templated file, 37 of 38 adaptive-only exact matches already have the gold string inside the B3 prediction, and containment differs by one question (76 of 77 against 75 of 77). On the held-out file the same pattern covers 88 of 98 exact-match gains; containment differs by 10 questions (147 of 193 against 137). Do not describe those exact-match gaps as a retrieval win.
- The entity-field cut is part of the method and is not applied to B2 or B3. The saved full-test rows do not record whether the cut fired. Precision at 5 is higher for the adaptive system, and that comparison is confounded.
- No ablation of the router, the window, the sufficiency loop, the surname or entity cut, or the sentence selector is in these files.
- No published temporal system is reimplemented. There is no row for Adaptive-RAG, TempRALM, or a date-aware re-ranker on top of B2.
- The TimeQA index is chunks of the pages attached to the sampled questions, not a retrieval corpus over Wikipedia. Exact match is 0 of 500 for every heuristic system. Containment (148, 129, and 96 of 500) is close to "did the gold string occur in the chosen sentence?" These scores are not comparable to Chen, Wang, and Wang (2021).
- Three saves are different programs. Checkpoint A ties adaptive and B3 exact match at 0.468 on 77 questions. The 18 November 2026 templated file does not. The 16-question TimeQA slice reaches 13 of 16 adaptive containment after rules written on that slice. The 500-question file is 148 of 500. Do not merge these tables.
- Report latency as a median and a mean. On TimeQA the adaptive median is 3.730 seconds and the mean is 47.486 seconds. Query `tqq_c61f3b840e35` ("Janet Frame received which award in 1990?") took 20974.779 seconds under policy `historical_adaptive_window`. B3 contained the same gold string in 4.737 seconds. The file does not record why the query hung. The mean of the other 499 adaptive questions is 5.547 seconds. Fifty adaptive queries exceeded 10 seconds, and four exceeded 30 seconds. No B2 or B3 query on that file exceeded 10 seconds.
- Skipping retrieval is a trivia whitelist. It fires on 10 of 77 templated questions and on none of the held-out or TimeQA questions. It does not reduce latency.
- B3 temporal accuracy of 1.000 restates the fixed year filter. It is not answer quality. The router's class accuracy is a different quantity, and the saved scores (1.000 on TimeQA and the held-out file; 76 of 77 necessity and 77 of 77 temporal on the templated file) are not broken down by cue.
- The confidence head is untrained. Window widths, blend weights, and the other constants in the method were chosen by hand. Some TimeQA rules were chosen while looking at the 16-question slice. There is no untouched development split for those choices.
- The learned path is reported only on 16-question slices. Its router accuracy was computed on the 48 questions used to pick the temperature. It is absent from the full-test files, and it is the worst row on the TimeQA slice.
- The held-out split removes six entity names from that file's train and validation questions. The analyzer used at test time was trained on the main file, which contains those names. The split does not hold the entities out of the router.
- B1 is the extractive reader with an empty document list plus hardcoded trivia replies. It is not a closed-book language model. The result files label it `B1_llm_only`.
- Abstention accuracy on TimeQA is 0.820 for all four systems, so it does not separate them.
- The current-question branch uses a frozen reference day of 1 October 2025. FreshQA and RealTime QA were not run.
- The paper does not give a public code URL. The reference day, the student-project origin, and the missing ablations should stay visible. Confirm author order and whether the university must approve the deposit before submitting.
