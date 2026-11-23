# Comparative Analysis: Adaptive Temporal RAG vs Baselines

Authoritative frozen Checkpoint B numbers: [`final_results.md`](final_results.md) and [`final_results.json`](final_results.json). Config: `configs/final.yaml`.

B1/B2/B3/Adaptive share the extractive reader, answer projection, and entity/surname cut. They differ on retrieval decision, the time window, and the sufficiency loop. Do not quote older 0.600 / 0.844 / 0.468 figures.

## Master table

| Dataset | System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic | B1 no retrieval | 77 | 0.130 | 0.167 | 0.130 | 0.000 | 0.000 | 0.247 | 0.000 | 0.000 | 0.000 |
| synthetic | B2 standard | 77 | 0.649 | 0.676 | 0.727 | 0.411 | 0.655 | 0.662 | 1.000 | 1.551 | 1.432 |
| synthetic | B3 fixed ±2y | 77 | 0.935 | 0.951 | 0.987 | 0.608 | 0.943 | 1.000 | 1.000 | 1.555 | 1.452 |
| synthetic | Adaptive | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.690 | 1.573 |
| heldout | B1 no retrieval | 193 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| heldout | B2 standard | 193 | 0.332 | 0.361 | 0.409 | 0.346 | 0.577 | 0.596 | 1.000 | 1.569 | 1.489 |
| heldout | B3 fixed ±2y | 193 | 0.684 | 0.696 | 0.710 | 0.591 | 0.879 | 1.000 | 1.000 | 1.546 | 1.410 |
| heldout | Adaptive | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 2.584 | 1.715 |
| timeqa | B1 no retrieval | 500 | 0.000 | 0.013 | 0.014 | 0.000 | 0.000 | — | 0.000 | 0.000 | 0.000 |
| timeqa | B2 standard | 500 | 0.006 | 0.077 | 0.260 | 0.235 | 0.468 | — | 1.000 | 6.056 | 6.118 |
| timeqa | B3 fixed ±2y | 500 | 0.010 | 0.089 | 0.270 | 0.237 | 0.392 | — | 1.000 | 6.074 | 6.116 |
| timeqa | Adaptive | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 8.736 | 6.492 |

Source: `artifacts/general_eval/final/{synthetic,heldout,timeqa}/results_summary.json`.

## Where Adaptive helps

- **Held-out (n=193):** EM 0.762 vs B3 0.684 and B2 0.332; Contains 0.762 vs 0.710 and 0.409; P@5 0.682 vs 0.591.
- **TimeQA (n=500):** Contains 0.296 vs B3 0.270 and B2 0.260; R@5 0.488 vs B3 0.392 (the ±2y filter on stored timestamps hurts B3 recall).
- **Synthetic (n=77):** P@5 0.668 vs B3 0.608; one extra exact match (0.948 vs 0.935). Skip fires on 10 of 77 queries (ret 0.870).

## Where Adaptive does not help

- **Synthetic Contains** ties B3 at 0.987.
- **TimeQA EM** is 0.008 vs B3 0.010 (four vs five questions) even with span extraction. Token F1 is 0.088 vs 0.089.
- **Latency** is never better: synthetic 1.690s vs 1.555s; held-out 2.584s vs 1.546s; TimeQA 8.736s vs 6.074s.
- **Skip** does not fire on held-out or TimeQA (retrieval rate 1.000).

## Ablations

From `final_results.md`. On synthetic and held-out, only turning off the surname/entity cut moves EM/Contains (0.948→0.922 and 0.762→0.699). Router, window, sufficiency, and sentence selector leave those scores unchanged. On TimeQA the sentence selector is the largest drop (Contains 0.296→0.168), then the entity cut (0.252). Window off falls to 0.272 Contains / 0.397 R@5. Sufficiency off is 0.298 Contains: the loop does not help that file. Router off matches the full system because retrieval rate is already 1.
