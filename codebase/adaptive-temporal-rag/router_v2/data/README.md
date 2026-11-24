# Router eval-only sets

`probe36.json` (36 questions) and `challenge_v2.jsonl` (72 questions) are held out of training and are used to score the router heads.

The labelled train / val / test JSONL files (80,145 / 10,018 / 9,901 rows) are not stored in git. Point the Colab notebook's `DATA_DIR` or `DATA_ZIP` at a local copy to retrain.
