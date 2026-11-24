# Learned temporal router (r4)

This folder holds the round-4 learned router: training and labelling scripts, the year-rule fix, small eval-only sets, bundle metadata, and the frozen retrieval-eval JSON files.

The presentable Colab notebook is [`../notebooks/train_learned_router.ipynb`](../notebooks/train_learned_router.ipynb).

## What the router does

A multitask DeBERTa classifier predicts whether a question needs retrieval and which temporal class it has (`current`, `historical`, `timeless`). Retrieval is skipped when the router predicts `needs_retrieval = 0`. There is no trivia whitelist in the published r4 runs.

## Published r4 result

On the frozen Checkpoint B evaluation the learned router **ties** the rule-based Adaptive system (0/0 discordant answers, McNemar *p* = 1) on all three sets. Numbers below are from `results/` and `bundles/*/eval_report_v2.json`. Exact match is reported on the templated synthetic test (n = 77) and the six-entity held-out test (n = 193). TimeQA uses containment (n = 500). ECE is the test-set confidence head versus both heads being correct. Training time is the training loop only, from `analyzer_meta.json`.

| Model | Templated EM | Held-out EM | TimeQA Contains | probe36 both (temporal) | challenge both | Test ECE | Train time |
|-------|-------------:|------------:|----------------:|------------------------:|---------------:|---------:|-----------:|
| r4_base (DeBERTa-v3-base, 3 epochs, bs 64, lr 3e-5) | 0.948 | 0.762 | 0.296 | 29/36 (32) | 65/72 | 0.0559 | 171.4 s |
| r4_large (DeBERTa-v3-large, 2 epochs, bs 32, lr 1e-5) | 0.948 | 0.762 | 0.296 | 29/36 (33) | 66/72 | 0.0478 | 406.5 s |

Training data after the year-rule fix and 274 added pairs: 80,145 / 10,018 / 9,901 (train / val / test). The year rule relabelled 3,810 training questions (4,855 across all splits). The published training GPU was an NVIDIA RTX PRO 6000 Blackwell Server Edition.

The learned router does not beat the rules. It matches them without a whitelist and is better calibrated than the earlier 100k router (templated EM 0.740, *p* = 3.05e-5, ECE 0.198).

## Train on Colab

Open `notebooks/train_learned_router.ipynb`. Point `DATA_DIR` (or `DATA_ZIP`) at a copy of the labelled `train.jsonl` / `val.jsonl` / `test.jsonl`. The notebook applies the year rule, trains base and large, and prints the committed retrieval table even if you skip training.

Command-line equivalent, from `codebase/adaptive-temporal-rag`:

```bash
python router_v2/train_v2.py \
  --repo . \
  --data /path/to/data_r4 \
  --out router_v2/bundles/run_base \
  --model microsoft/deberta-v3-base \
  --epochs 3 --bs 64 --lr 3e-5 \
  --use-llm-labels --bf16
```

`fix_year_labels.py` and `aug_r4.py` rebuild the r4 labels from a Qwen-labelled directory. `llm_label_v2.py` is the optional 7B relabeller (GPU). `eval_router_v2.py` and `run_eval_pipeline.sh` run the frozen retrieval eval; they need the corpus, index, and `configs/final.yaml` already in this repo.

## Not in git

- `analyzer.pt` checkpoints (about 702 MB base, 1.7 GB large). SHA-256 files are in `bundles/r4_*_fix/`.
- The 80k-row training JSONL splits.
- The workshop paper and arXiv sources (kept on the local machine only).
