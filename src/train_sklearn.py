"""Train the available candidate-pair table with a leak-free entity split.

This script deliberately depends only on pandas, scikit-learn, and joblib so it
can run in the local environment when the optional PyTorch/LightGBM stack is
not installed.  It writes a separately named model and never generates
submission candidate-pair files.

Example:
    python3 -m src.train_sklearn \
      --dataset dataset_rishiraj/train/training_data.csv --model-dir model
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit

from .evaluate import macro_f_beta


FEATURE_COLUMNS = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio",
    "core_name_ratio", "address_ratio", "address_token_set_ratio",
    "address_number_match", "country_match", "name_exact_match",
    "core_name_exact_match", "address_exact_match",
]


def _predictions(frame, threshold, all_entity_ids):
    """Aggregate accepted pair scores into the competition's entity mapping."""
    predictions = {entity_id: set() for entity_id in all_entity_ids}
    accepted = frame.loc[frame["probability"] >= threshold]
    for source1_id, match_id in zip(
        accepted["source1_entity_id"], accepted["matched_entity_id"]
    ):
        predictions[source1_id].add(match_id)
    return predictions


def _ground_truth(frame, all_entity_ids):
    truth = {entity_id: set() for entity_id in all_entity_ids}
    positives = frame.loc[frame["label"] == 1]
    for source1_id, match_id in zip(
        positives["source1_entity_id"], positives["matched_entity_id"]
    ):
        truth[source1_id].add(match_id)
    return truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_rishiraj/train/training_data.csv")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    required = {"source1_entity_id", "matched_entity_id", "label", *FEATURE_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature table contains missing values; audit/fix it before training.")

    groups = data["source1_entity_id"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.val_fraction, random_state=args.seed
    )
    train_idx, val_idx = next(splitter.split(data[FEATURE_COLUMNS], data["label"], groups))
    train = data.iloc[train_idx]
    validation = data.iloc[val_idx].copy()

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        max_features=1.0,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train[FEATURE_COLUMNS], train["label"])
    validation["probability"] = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]

    val_entity_ids = validation["source1_entity_id"].unique()
    truth = _ground_truth(validation, val_entity_ids)
    threshold_scores = {}
    for threshold in np.unique(np.r_[np.arange(0.50, 0.991, 0.01), 0.995, 0.999]):
        threshold = float(np.round(threshold, 3))
        threshold_scores[str(threshold)] = macro_f_beta(
            _predictions(validation, threshold, val_entity_ids), truth, beta=0.5
        )
    best_threshold_text, best_f05 = max(threshold_scores.items(), key=lambda item: item[1])
    best_threshold = float(best_threshold_text)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "sklearn_entity_matcher.joblib")
    config_path = os.path.join(args.model_dir, "sklearn_entity_matcher_config.json")
    joblib.dump(model, model_path)
    config = {
        "dataset": args.dataset,
        "feature_columns": FEATURE_COLUMNS,
        "model": "ExtraTreesClassifier",
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "validation": {
            "split": "GroupShuffleSplit by source1_entity_id",
            "val_fraction": args.val_fraction,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_entities": int(train["source1_entity_id"].nunique()),
            "validation_entities": int(len(val_entity_ids)),
            "best_macro_f0.5": best_f05,
            "threshold": best_threshold,
            "threshold_scores": threshold_scores,
        },
    }
    with open(config_path, "w") as output:
        json.dump(config, output, indent=2)

    print(f"Train rows/entities: {len(train):,}/{train['source1_entity_id'].nunique():,}")
    print(f"Validation rows/entities: {len(validation):,}/{len(val_entity_ids):,}")
    print(f"Best validation macro F0.5: {best_f05:.6f} at threshold {best_threshold:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
