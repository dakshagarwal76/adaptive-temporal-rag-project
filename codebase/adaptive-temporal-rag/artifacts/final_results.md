# Frozen Checkpoint B results

Source: `artifacts/general_eval/final/`. Config: `configs/final.yaml`.
B1/B2/B3/B4/Adaptive share the extractive reader, answer projection, entity/surname cut, and sentence selector.
B1–B3 and Adaptive differ on retrieval decision, the time window, and the sufficiency loop.
B4 is dense+BM25 with exponential year-distance decay (lambda 2 years), no hard window and no sufficiency loop.
Adaptive-Fast is the saved sufficiency-off ablation, not a new four-system run. Its latency is from that later Adaptive-only run.
Learned path is experimental and is not in these tables.
TimeQA EM/F1 use span extraction on the selected sentence; Contains still uses the sentence.
Temporal accuracy is omitted for TimeQA (index timestamps are not fact dates).

## Master table

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

## synthetic (n=77)

| System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 no retrieval | 77 | 0.130 | 0.167 | 0.130 | 0.000 | 0.000 | 0.247 | 0.000 | 0.000 | 0.000 |
| B2 standard | 77 | 0.649 | 0.676 | 0.727 | 0.411 | 0.655 | 0.662 | 1.000 | 1.551 | 1.432 |
| B3 fixed ±2y | 77 | 0.935 | 0.951 | 0.987 | 0.608 | 0.943 | 1.000 | 1.000 | 1.555 | 1.452 |
| B4 TempRALM decay | 77 | 0.662 | 0.684 | 0.727 | 0.435 | 0.695 | 0.675 | 1.000 | 0.031 | 0.030 |
| Adaptive | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.690 | 1.573 |
| Adaptive-Fast (sufficiency off) | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.179 | 1.290 |

File: `artifacts\general_eval\final\synthetic\results_summary.json`

## heldout (n=193)

| System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 no retrieval | 193 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 standard | 193 | 0.332 | 0.361 | 0.409 | 0.346 | 0.577 | 0.596 | 1.000 | 1.569 | 1.489 |
| B3 fixed ±2y | 193 | 0.684 | 0.696 | 0.710 | 0.591 | 0.879 | 1.000 | 1.000 | 1.546 | 1.410 |
| B4 TempRALM decay | 193 | 0.358 | 0.369 | 0.383 | 0.383 | 0.639 | 0.497 | 1.000 | 0.040 | 0.038 |
| Adaptive | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 2.584 | 1.715 |
| Adaptive-Fast (sufficiency off) | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 0.663 | 0.649 |

File: `artifacts\general_eval\final\heldout\results_summary.json`

## timeqa (n=500)

| System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 no retrieval | 500 | 0.000 | 0.013 | 0.014 | 0.000 | 0.000 | — | 0.000 | 0.000 | 0.000 |
| B2 standard | 500 | 0.006 | 0.077 | 0.260 | 0.235 | 0.468 | — | 1.000 | 6.056 | 6.118 |
| B3 fixed ±2y | 500 | 0.010 | 0.089 | 0.270 | 0.237 | 0.392 | — | 1.000 | 6.074 | 6.116 |
| B4 TempRALM decay | 500 | 0.010 | 0.084 | 0.234 | 0.174 | 0.358 | — | 1.000 | 0.062 | 0.059 |
| Adaptive | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 8.736 | 6.492 |
| Adaptive-Fast (sufficiency off) | 500 | 0.008 | 0.088 | 0.298 | 0.243 | 0.491 | — | 1.000 | 3.184 | 3.160 |

File: `artifacts\general_eval\final\timeqa\results_summary.json`

## Ablations (Adaptive only)

Each ablation turns off one Adaptive component. Baselines are not re-run here.

### synthetic

| Ablation | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive (full) | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.690 | 1.573 |
| router off (always retrieve) | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 1.000 | 2.040 | 1.456 |
| adaptive window off (fixed ±2y) | 77 | 0.948 | 0.960 | 0.987 | 0.635 | 0.989 | 1.000 | 0.870 | 1.313 | 1.349 |
| sufficiency loop off | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.179 | 1.290 |
| surname/entity cut off | 77 | 0.922 | 0.934 | 0.961 | 0.555 | 0.925 | 0.974 | 0.870 | 1.325 | 1.305 |
| sentence selector off | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.282 | 1.276 |

### heldout

| Ablation | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive (full) | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 2.584 | 1.715 |
| router off (always retrieve) | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 1.124 | 0.817 |
| adaptive window off (fixed ±2y) | 193 | 0.762 | 0.766 | 0.762 | 0.640 | 0.960 | 1.000 | 1.000 | 0.945 | 0.689 |
| sufficiency loop off | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 0.663 | 0.649 |
| surname/entity cut off | 193 | 0.699 | 0.703 | 0.699 | 0.526 | 0.862 | 0.964 | 1.000 | 0.931 | 0.670 |
| sentence selector off | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 0.917 | 0.655 |

### timeqa

| Ablation | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive (full) | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 8.736 | 6.492 |
| router off (always retrieve) | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 4.446 | 3.185 |
| adaptive window off (fixed ±2y) | 500 | 0.010 | 0.089 | 0.272 | 0.236 | 0.397 | — | 1.000 | 3.333 | 3.141 |
| sufficiency loop off | 500 | 0.008 | 0.088 | 0.298 | 0.243 | 0.491 | — | 1.000 | 3.184 | 3.160 |
| surname/entity cut off | 500 | 0.006 | 0.076 | 0.252 | 0.167 | 0.338 | — | 1.000 | 6.650 | 5.430 |
| sentence selector off | 500 | 0.002 | 0.048 | 0.168 | 0.241 | 0.488 | — | 1.000 | 6.551 | 4.300 |

## Significance

Paired tests on the saved per-query rows. No retrieval was re-run.
McNemar p-values are exact two-sided binomial tests on discordant pairs.
95% intervals are paired bootstrap percentile intervals of (A − B) with 10000 resamples, seed 0.
A is the first system named in each row.

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

File: `artifacts/significance.json`.
