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

## Training on Google Colab TPU (via VS Code Extension)

You can train the **Deep Residual Entity Matcher** on Google Colab's cloud **TPU** (Tensor Processing Unit) directly inside VS Code:

1. Open `train_tpu.ipynb` in VS Code / Antigravity IDE.
2. In the top-right corner of the notebook editor, click **Select Kernel** > **Google Colab** (or `Cmd+Shift+P` -> `Colab: Connect to a Colab runtime`).
3. Sign in to your Google Account when prompted.
4. Select the runtime accelerator: **TPU** (v2/v3/v5e).
5. If your dataset is in Google Drive, run the Drive mount cell or use `Cmd+Shift+P` -> `Colab: Mount Google Drive to Server...`.
6. Run the notebook cells to train the Deep Residual Entity Matcher using PyTorch-XLA (`torch_xla`), calibrate the $F_{0.5}$ decision threshold, and generate `matching_results.tsv`.

Alternatively, from the command line on any TPU instance:
```bash
python -m src.train_tpu --train-dir dataset/train --model-dir model --epochs 10 --batch-size 2048
```

## Files

```
requirements.txt
train_tpu.ipynb    # Google Colab TPU training and evaluation notebook
src/
  normalize.py     # text cleaning + abbreviation expansion
  blocking.py      # candidate generation (multi-key inverted index blocking)
  features.py      # pairwise similarity features
  evaluate.py      # macro F_0.5 scorer (matches competition formula)
  model_tpu.py     # Deep Residual Entity Matcher PyTorch neural network
  train_tpu.py     # TPU training pipeline with PyTorch-XLA & threshold tuning
  train.py         # trains LightGBM model, tunes threshold, saves model/
  predict.py       # scores test candidates, writes output/
```

