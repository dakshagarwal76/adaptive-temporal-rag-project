# Frozen Checkpoint B results

Source: `artifacts/general_eval/final/`. Config: `configs/final.yaml`.
B1/B2/B3/Adaptive share the extractive reader, answer projection, and entity/surname cut.
They differ on retrieval decision, the time window, and the sufficiency loop.
Learned path is experimental and is not in these tables.
TimeQA EM/F1 use span extraction on the selected sentence; Contains still uses the sentence.
Temporal accuracy is omitted for TimeQA (index timestamps are not fact dates).

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

## synthetic (n=77)

| System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 no retrieval | 77 | 0.130 | 0.167 | 0.130 | 0.000 | 0.000 | 0.247 | 0.000 | 0.000 | 0.000 |
| B2 standard | 77 | 0.649 | 0.676 | 0.727 | 0.411 | 0.655 | 0.662 | 1.000 | 1.551 | 1.432 |
| B3 fixed ±2y | 77 | 0.935 | 0.951 | 0.987 | 0.608 | 0.943 | 1.000 | 1.000 | 1.555 | 1.452 |
| Adaptive | 77 | 0.948 | 0.960 | 0.987 | 0.668 | 0.954 | 1.000 | 0.870 | 1.690 | 1.573 |

File: `artifacts\general_eval\final\synthetic\results_summary.json`

## heldout (n=193)

| System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 no retrieval | 193 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B2 standard | 193 | 0.332 | 0.361 | 0.409 | 0.346 | 0.577 | 0.596 | 1.000 | 1.569 | 1.489 |
| B3 fixed ±2y | 193 | 0.684 | 0.696 | 0.710 | 0.591 | 0.879 | 1.000 | 1.000 | 1.546 | 1.410 |
| Adaptive | 193 | 0.762 | 0.766 | 0.762 | 0.682 | 0.912 | 1.000 | 1.000 | 2.584 | 1.715 |

File: `artifacts\general_eval\final\heldout\results_summary.json`

## timeqa (n=500)

| System | N | EM | F1 | Contains | P@5 | R@5 | Temp | Ret | mean s | median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 no retrieval | 500 | 0.000 | 0.013 | 0.014 | 0.000 | 0.000 | — | 0.000 | 0.000 | 0.000 |
| B2 standard | 500 | 0.006 | 0.077 | 0.260 | 0.235 | 0.468 | — | 1.000 | 6.056 | 6.118 |
| B3 fixed ±2y | 500 | 0.010 | 0.089 | 0.270 | 0.237 | 0.392 | — | 1.000 | 6.074 | 6.116 |
| Adaptive | 500 | 0.008 | 0.088 | 0.296 | 0.241 | 0.488 | — | 1.000 | 8.736 | 6.492 |

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

