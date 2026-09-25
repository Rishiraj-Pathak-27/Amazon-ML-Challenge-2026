# Business Entity Resolution — Pipeline

End-to-end, two-stage entity resolution pipeline:

1. **Blocking** (`src/blocking.py`) — character-n-gram TF-IDF + cosine
   nearest-neighbour search (robust to typos/transliteration/word
   reordering), unioned with an exact first-name-token blocking pass for
   recall. Produces the candidate set fed to the model.
2. **Matching** (`src/features.py` + LightGBM in `src/train.py` /
   `src/predict.py`) — pairwise string-similarity features (Jaccard,
   Levenshtein, token-sort, partial-ratio, country match, etc.) scored by
   a LightGBM binary classifier, with the decision threshold tuned to
   maximise the competition's macro **F₀.₅** metric.

No external data, APIs, or lookups are used anywhere — only the provided
`business_name` / `business_address` / `country` fields — per the
challenge's fair-play rules. The model (LightGBM, a gradient-boosted
tree ensemble) is Apache/MIT-licensed open source and has far fewer than
8B parameters, so it satisfies the model constraint by construction (it
isn't an LLM at all).

## Setup

```bash
pip install -r requirements.txt
```

## Run

Directory layout expected (matches the challenge's `student_resource/`):

```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

From the `business_entity_resolution/` directory:

```bash
# 1. Train (also holds out a validation split and tunes the threshold)
python -m src.train --train-dir dataset/train --model-dir model

# 2. Predict on the test set
python -m src.predict --test-dir dataset/test --model-dir model --output-dir output
```

This writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`
in the exact format the challenge requires. Run the challenge's own
`utils/validate_submission.py` against them before uploading.

## Key parameters

- `--top-k` (default 20 for train, uses the value saved in `model/config.json`
  at predict time) — how many nearest neighbours the blocking stage pulls
  per Source-1 entity, per other source. Higher = better recall ceiling,
  slower and noisier candidate set.
- `--val-fraction` (default 0.15) — fraction of Source-1 training entities
  held out (by entity, so no leakage) to tune the classification threshold.

## Tuning ideas / things to try next

- Increase `--top-k` if validation recall (before thresholding) looks low
  — check how many val positives actually make it into the candidate set.
- Add more features: e.g. numeric-token overlap (street numbers, PIN/ZIP
  codes), soundex/metaphone phonetic keys for name matching, or per-country
  address-format-aware parsing.
- Try per-country threshold tuning if US/India/France show different
  optimal precision-recall trade-offs.
- Swap LightGBM for CatBoost/XGBoost if it helps precision — the code is
  structured so this only touches `src/train.py`.
- The `pandas.apply` row loop in `features.build_feature_frame` is
  straightforward but not the fastest possible; if the full dataset is
  very large, vectorize the string-similarity computations or parallelize
  with `multiprocessing`.

## Files

```
requirements.txt
src/
  normalize.py   # text cleaning + abbreviation expansion
  blocking.py    # candidate generation (TF-IDF NN + token blocking)
  features.py    # pairwise similarity features
  evaluate.py    # macro F_0.5 scorer (matches competition formula)
  train.py       # trains model, tunes threshold, saves model/
  predict.py     # scores test candidates, writes output/
```
