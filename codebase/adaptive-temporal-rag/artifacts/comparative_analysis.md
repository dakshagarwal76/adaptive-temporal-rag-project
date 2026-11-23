# Comparative Analysis: Adaptive Temporal RAG vs Baselines

Evaluated on **77** held-out test queries (same split for all systems).

## Systems compared

| ID | System | Behavior |
|----|--------|----------|
| B1 | LLM-only | Never retrieves |
| B2 | Standard RAG | Always retrieves (semantic only) |
| B3 | Fixed Temporal RAG | Always retrieves with fixed time policy |
| Ours | Adaptive Temporal RAG | Conditional retrieval + class-specific temporal policy |

## Overall results

| System | N | Contains | EM | Token F1 | Temp Acc | P@5 | R@5 | Ret Rate | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 LLM-only | 77 | 0.130 | 0.117 | 0.158 | 0.247 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 Standard RAG | 77 | 0.714 | 0.390 | 0.529 | 0.662 | 0.393 | 0.655 | 1.000 | 0.594 |
| B3 Fixed Temporal | 77 | 0.987 | 0.468 | 0.674 | 1.000 | 0.612 | 0.954 | 1.000 | 0.561 |
| Adaptive (ours) | 77 | 1.000 | 0.468 | 0.676 | 1.000 | 0.673 | 0.954 | 0.870 | 0.560 |

## Key takeaways

- Adaptive improves answer contains by +28.6% and temporal accuracy by +33.8% vs Standard RAG.
- Adaptive retrieves on 87% of queries vs 100% for Standard/Fixed RAG (saves ~13% retrieval calls).
- vs Fixed Temporal: temporal accuracy gap is +0.0%; contains gap is +1.3% (often from skipping retrieval on timeless items under extractive eval).
- Per-query vs Standard RAG (contains): win 29% / tie 71% / loss 0%.

## Results by temporal class (Contains)

| System | current | historical | timeless |
| --- | --- | --- | --- |
| Adaptive (ours) | 1.000 | 1.000 | 1.000 |
| B1 LLM-only | 0.000 | 0.000 | 0.385 |
| B2 Standard RAG | 0.500 | 0.733 | 1.000 |
| B3 Fixed Temporal | 1.000 | 0.933 | 1.000 |

## Results by temporal class (Temporal accuracy)

| System | current | historical | timeless |
| --- | --- | --- | --- |
| Adaptive (ours) | 1.000 | 1.000 | 1.000 |
| B1 LLM-only | 0.000 | 0.000 | 0.731 |
| B2 Standard RAG | 0.333 | 0.867 | 1.000 |
| B3 Fixed Temporal | 1.000 | 1.000 | 1.000 |

## Deltas: Adaptive − baseline

| Baseline | Δ Contains | Δ Temp Acc | Δ Ret Rate | Δ Token F1 |
| --- | --- | --- | --- | --- |
| B1 LLM-only | +0.870 | +0.753 | +0.870 | +0.517 |
| B2 Standard RAG | +0.286 | +0.338 | -0.130 | +0.146 |
| B3 Fixed Temporal | +0.013 | +0.000 | -0.130 | +0.001 |

## Interpretation notes

- **B1** is weak on time-sensitive facts without a parametric LLM answerer (extractive stub returns little).
- **B2** often ranks stale passages highly for “current” questions → lower temporal accuracy.
- **B3** is a strong always-on temporal upper bound but retrieves on every query.
- **Adaptive** matches B3 on *current* / *historical* answer quality while skipping many timeless/no-need queries; under extractive eval, skipped retrieval lowers timeless *contains* (re-run with `hf`/`api` generator for fairer QA scores).
