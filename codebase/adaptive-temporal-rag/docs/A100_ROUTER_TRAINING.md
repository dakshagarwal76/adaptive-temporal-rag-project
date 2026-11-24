# A100 router training

Train a new query analyzer on `data/router_large` with `microsoft/deberta-v3-base` (or `microsoft/deberta-v3-large`). The notebook writes a new bundle under Google Drive. It does not replace `artifacts/analyzer.pt` or the DistilBERT `artifacts/analyzer_meta.json`.

Load the saved bundle with `src.router_large.load_large_router`. That helper calls `QueryAnalyzerService.from_pretrained_bundle` and then applies `temperature` and `blend_confidence` from the new meta file. `load_query_analyzer` is unchanged.

The numbers below are from the local build on 23 Nov 2026 (`seed=42`, `reference_year=2026`). Training metrics (macro-F1, ECE) are not in this file. The notebook writes them to Drive after the A100 run.

## Dataset that was built

| Split | Rows | current | historical | timeless | needs retrieval 1 | needs retrieval 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 40091 | 1383 | 19400 | 19308 | 20783 | 19308 |
| val | 4951 | 173 | 2364 | 2414 | 2537 | 2414 |
| test | 4959 | 173 | 2372 | 2414 | 2545 | 2414 |
| total | 50001 | 1729 | 24136 | 24136 | 25865 | 24136 |

Selected rows by source:

| Source | Rows in the 50,001 |
| --- | ---: |
| nq_open | 13624 |
| squad_v2 | 11435 |
| timeqa | 8262 |
| triviaqa | 7968 |
| situatedqa | 3309 |
| sciq | 1581 |
| commonsense_qa | 1222 |
| arc | 978 |
| qasc | 954 |
| openbookqa | 668 |

Synthetic rows in the final file: 0 (fraction 0.0). The 20% cap was not used.

Rows loaded before dedup and the leak filter:

| Source | Loaded |
| --- | ---: |
| nq_open (`google-research-datasets/nq_open`, train) | 87925 |
| squad_v2 (`rajpurkar/squad_v2`, train, random cap) | 80000 |
| triviaqa (`mandarjoshi/trivia_qa`, `unfiltered.nocontext`, train, random cap) | 50000 |
| timeqa (`hugosousa/TimeQA` train + validation parquet only) | 35097 |
| sciq | 13678 |
| situatedqa (GitHub `temp.train` / `temp.dev` / `temp.test` JSONL) | 12227 |
| commonsense_qa | 10962 |
| qasc | 9040 |
| arc (ARC-Easy + ARC-Challenge) | 7729 |
| openbookqa | 5844 |
| synthetic non-test, non-paraphrase (`data/queries.jsonl` 406 + `data/heldout/queries.jsonl` 406) | 812 |

Labeled total 313314. After dedup 310977 (2337 dropped). After the leak filter 299016, of which current 1729, historical 89620, timeless 207667. Every surviving current row was kept. Historical and timeless were downsampled to 24136 each so the file stays at 50001. The trainer uses inverse-frequency class weights because current is the small class.

Label rule (`src/router_large.label_question`, reference year 2026): an explicit year is historical when the latest year is before 2025, otherwise current. Explicit cues (`currently`, `today`, `this year`, `right now`, and similar) and present-tense office questions map to current. TimeQA without a recent year maps to historical. General QA with no time cue maps to timeless and `needs_retrieval=0`. Current and historical rows use `needs_retrieval=1`. A scan of the written current rows found 3 questions whose cue is the word `today` or `tonight` inside a non-temporal phrase (two capacitor questions, one Lion King song title).

## Leakage check

Forbidden index used for the filter:

| Ban | Count |
| --- | ---: |
| Exact questions | 6117 |
| Entities | 226 |
| Templates | 335 |

The exact set starts from `data/queries.jsonl` test, `data/heldout/queries.jsonl` test, and the local TimeQA 500 (`data/general/timeqa/queries.jsonl`, `split=test`). The self-check on that local index printed 623 questions, 77 entities, and 335 templates. The official TimeQA test split (6075 questions) was then exact-banned; 5494 of those strings were new. Wiki titles for the local 500 were added as entities; 149 unique titles were new, bringing the entity count to 226.

Rows removed by the filter: template 10647, entity 1227, exact 87. The largest template families removed were `what is {entity}` (6483), the TimeQA position frames (`what position did {entity} take in {mon} {year}` 435, and the related position/hold frames), and `who won the {entity} in {year}` (201).

Residual scan of the written 50001 rows: 0 leaks. The builder exits with code 2 if the total is under 50000, the synthetic fraction is over 0.20, or any residual leak remains. This run exited 0. `constraints` in `data/router_large/stats.json` are `min_rows`, `synthetic_cap`, and `no_residual_leaks`, all true.

## Sources that did not load

| Source | Error |
| --- | --- |
| TimeQuestions | GitHub URLs under `zhenjia2017/EXAQT` and `semantic-systems/KGQA-datasets` returned HTTP 404. `AmazonScience/timequestions`, `timequestions`, and `lmqg/qg_timequestions` are not on the Hub. |
| TempQuestions | GitHub URLs under `zhenjia2017/TEMPQUESTIONS` returned HTTP 404. `tempquestions` and `AmazonScience/tempquestions` are not on the Hub. |
| FreshQA | `freshllms/FreshQA` and `yentinglin/freshqa` are not on the Hub. |
| RealtimeQA | `realtimeqa/realtimeqa` and `realtimeqa` are not on the Hub. |
| Mintaka | `AmazonScience/mintaka` still ships `mintaka.py`. Current `datasets` rejects dataset scripts. |
| Test of Time | `baharef/ToT` requires a config (`tot_arithmetic`, `tot_semantic`, `tot_semantic_large`). It was not loaded. Those rows would have been marked synthetic and counted toward the 20% cap. |
| TemporalWiki | `seonghyeonye/TemporalWiki` files are triples and passages (`TWiki_Probes.zip`, `TWiki_Diffsets.zip`, `Wikipedia_Full.zip`), not questions. They were not turned into templated questions. `Wikipedia_Full.zip` was not downloaded. |

## Files

New files:

- `scripts/build_large_router_dataset.py`
- `src/router_large.py` (`label_question`, `LeakageIndex`, `train_large_router`, `sweep_hyperparams`, `load_large_router`, `evaluate_rows`)
- `data/router_large/train.jsonl`, `val.jsonl`, `test.jsonl`, `stats.json`
- `notebooks/06_a100_router_training.ipynb`
- `docs/A100_ROUTER_TRAINING.md`
- `artifacts/router_large_colab_bundle.zip`

Existing `artifacts/analyzer.pt`, `artifacts/analyzer_meta.json`, `data/queries.jsonl`, `data/heldout`, and `data/general` were not overwritten.

## Colab Pro A100

1. In Colab: Runtime, Change runtime type, GPU, A100.
2. Upload `artifacts/router_large_colab_bundle.zip` to Google Drive at:

   `MyDrive/adaptive-temporal-rag/router_large_colab_bundle.zip`

3. Upload `notebooks/06_a100_router_training.ipynb` to Colab (File, Upload notebook) and run the cells from top to bottom.
4. The second cell mounts Drive and unzips the bundle to `/content/router_bundle` when `src/router_large.py` is not already there. Outputs go to:

   `MyDrive/adaptive-temporal-rag/router_large_a100/`

5. Leave `USE_LARGE = False` for `microsoft/deberta-v3-base`, batch choices 16 and 32, gradient accumulation 1. Set `USE_LARGE = True` only for `microsoft/deberta-v3-large` (batch 4 or 8, accumulation 4, gradient checkpointing, no `torch.compile`).
6. `RUN_SWEEP = True` runs 6 Optuna trials over learning rate `{5e-6, 1e-5, 2e-5}`, the batch choices above, and max epochs `{3, 5}`. If Optuna cannot be imported, the same budget uses a fixed 6-point grid. Each trial early-stops on validation macro-F1 with patience 1 and does not write a checkpoint. The winning learning rate, batch size, and epoch cap are trained once more with patience 2.
7. Training uses bf16 autocast when the GPU supports it, cosine decay with warmup, class-weighted cross-entropy, and a confidence-head loss applied outside `MultitaskQueryAnalyzer.forward` (weight 0.3). `torch.compile` is probed on the first step and dropped if that step fails. A temperature in `[0.5, 3.0]` is fit on the validation temporal logits and stored in `analyzer_meta.json`.
8. The last cell evaluates the leak-free test split, `data/heldout/queries.jsonl` where `split=test`, and `data/general/timeqa/queries.jsonl` where `split=test`. It prints macro-F1, per-class F1, both confusion matrices, and 15-bin ECE, then writes `eval_report.json` next to `analyzer.pt`.

Budget for the 6-trial sweep plus the final fit is about 2–3 hours on an A100. This machine did not run that training.

After the run, load the bundle from the project (or from the unzipped bundle) with:

```python
from src.router_large import load_large_router
service = load_large_router("/content/drive/MyDrive/adaptive-temporal-rag/router_large_a100")
```

`service.model_name` in the meta file will be the DeBERTa checkpoint. DistilBERT loading of `artifacts/analyzer.pt` still goes through `load_query_analyzer`.
