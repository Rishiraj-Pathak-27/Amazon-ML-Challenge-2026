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

## Repository Structure

The codebase is organized into modular directories:

```
business_entity_resolution/
├── pipelines/                     # End-to-end inference & submission pipelines
│   ├── run_pipeline_v4_production.py  # Production pipeline (High-Precision V5)
│   ├── run_pipeline_v3.py         # Calibrated blocking & matching pipeline V3
│   ├── run_pipeline_v2.py         # End-to-end local training & inference V2
│   ├── run_optimized_pipeline.py  # Multi-channel inverted index pipeline
│   └── run_high_recall_pipeline.py# 10+ channel high-recall pipeline
│
├── benchmarks/                    # Offline evaluation, parameter tuning & benchmarks
│   ├── benchmark_v6.py            # Latest benchmark suite with refined metrics
│   ├── benchmark_v5.py            # Multi-channel candidate benchmark
│   ├── test_tuned_v3.py           # Calibration and sample validation
│   ├── inspect_v5_errors.py       # False-positive / false-negative error analysis
│   └── optimize_calibration.py    # Matcher threshold grid-search optimizer
│
├── training/                      # Model training & dataset synthesis
│   ├── train_local_m4.py          # Native Apple Silicon M4 MPS / Neural Engine training
│   ├── regenerate_training_data.py# Dataset generation with hard negative mining
│   └── train_tpu.ipynb            # Google Colab TPU training notebook
│
├── colab/                         # Google Colab cloud runner & setup scripts
│   ├── setup_colab.sh             # TPU/GPU environment setup script
│   ├── run_colab_gpu.py           # Colab GPU automated execution driver
│   └── colab_all_in_one.py        # All-in-one Colab submission generator
│
├── kaggle_runner/                 # Kaggle automated kernel runner
│   ├── kaggle_pipeline.py         # Self-contained Kaggle runner pipeline
│   └── kernel-metadata.json       # Kaggle kernel push configuration
│
├── kaggle_dataset/                # Kaggle dataset upload configuration & assets
│   └── dataset-metadata.json
│
├── testoutput/                    # Unstop validation test suite & hidden testcases
│   ├── simulate_unstop_portal.py  # Unstop Portal submission validator & F0.5 scorer
│   ├── test_all_outputs.py        # Symlink entrypoint to simulate_unstop_portal.py
│   ├── hidden_testcases.tsv       # 30,000 real ground-truth evaluation entities
│   └── leaderboard_summary.json   # Benchmark ranking summary
│
├── src/                           # Core reusable Python package
│   ├── normalize.py               # Text cleaning, abbreviation & legal suffix normalization
│   ├── blocking.py                # Multi-key inverted index candidate blocking
│   ├── features.py                # Pairwise similarity feature extraction
│   ├── evaluate.py                # Official Macro F0.5 scoring metric
│   ├── model_tpu.py               # Deep Residual Entity Matcher PyTorch neural network
│   ├── train.py                   # LightGBM classifier training
│   ├── train_local_m4.py          # M4-optimized model training
│   └── predict.py                 # Candidate scoring & submission generation
│
├── model/                         # Saved weights & model hyperparameters
│   ├── model_m4.pt                # Trained Deep Residual Entity Matcher weights
│   ├── config.json                # Model & threshold configuration
│   └── sklearn_entity_matcher.joblib # Trained scikit-learn matcher
│
├── output/                        # Competition submissions and outputs
│   ├── submmision/                # Competition submission TSV files (matching_results_V*.tsv)
│   └── archive/                   # Archived outputs
│
├── logs/                          # Execution logs
│   ├── amazon-ml-entity-matcher.log
│   └── test_run.log
│
├── student_resource/              # Official competition dataset & guidelines
├── cleaned_dataset/               # Preprocessed null-free dataset
├── dataset_rishiraj/              # Train/validation split datasets
├── Amazon-ML-Submission/          # Submission zip archive & delivery bundle
├── approach_summary.md            # Comprehensive solution architecture document
├── AGENTS.md                      # Agent guidelines & submission rules
└── requirements.txt               # Project dependencies
```

## Quick Start & Common Workflows

### 1. Run Production Pipeline (Generate Submission)
```bash
python3 pipelines/run_pipeline_v4_production.py
```
Outputs `matching_results_V*.tsv` in `output/submmision/`.

### 2. Validate & Benchmark Submission Files
```bash
python3 testoutput/simulate_unstop_portal.py
```
Validates portal formatting constraints and computes Macro $F_{0.5}$ across Public & Private leaderboard splits.

### 3. Run Offline Benchmark
```bash
python3 benchmarks/benchmark_v6.py --eval-sample 5000
```

### 4. Train Model Locally on Apple Silicon M4
```bash
python3 training/train_local_m4.py --epochs 5 --batch-size 2048
```


