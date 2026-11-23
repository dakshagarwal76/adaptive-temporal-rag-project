# Context4Agents @ WSDM 2027 workshop paper

Anonymized 4-page ACM sigconf paper. The numbers are the frozen Checkpoint B table in `artifacts/final_results.md` (commit parent `0db77f6`, plus the B4 decay run and the paired tests in `artifacts/significance.json`).

## Submission checklist

Source: [Context4Agents CFP](https://context4agents.github.io/), read 24 November 2026.

- [x] English, self-contained.
- [x] ACM `acmart`, `\documentclass[sigconf,anonymous,review]{acmart}`, US Letter.
- [x] Double-blind: the PDF prints `Anonymous Author(s)` and the running header `Anon.`
- [x] Body is under the 6-page limit. References are extra. There is no appendix.
- [ ] Human: create and activate an OpenReview profile (institutional email may be moderated). Portal opened 24 October 2026, 1:00 PM AoE.
- [ ] Human: submit `main.pdf` by **30 November 2026, 11:00 PM AoE**.
- [ ] Human: if accepted, an in-person talk on **19 February 2027** in Hong Kong is required. Add the real author block only in the camera-ready.

## How to compile

```text
C:\Users\DELL\AppData\Local\tectonic\tectonic.exe main.tex
```

Run that from this directory. The comparison tables are `\input{../paper/final_tables.tex}`.

## Page count

`main.pdf` is 4 pages. References start on page 4. There is no appendix.

## What the PDF quotes

B1–B3, B4 (`tempralm_decay`), Adaptive, and Adaptive-Fast (the saved sufficiency-off ablation) on the templated test (n=77), the six-entity test (n=193), and TimeQA (n=500). The 15 Adaptive-only ablations and the paired McNemar / bootstrap intervals are the same rows as `artifacts/final_results.md`. B4 is a TempRALM-style decay ranker written for this comparison. It is not the published TempRALM system. Turning the sufficiency loop off does not lower the answer scores.
