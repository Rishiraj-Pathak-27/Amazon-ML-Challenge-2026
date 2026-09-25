"""
Train the pairwise match / no-match classifier.

Pipeline:
  1. Load train_source1/2/3.tsv + train_ground_truth.tsv
  2. Split Source-1 entities into a train / validation split (entity-level,
     so no leakage between splits)
  3. Run blocking on each split, label candidate pairs from ground truth
  4. Compute pairwise features
  5. Train a LightGBM binary classifier on the train split
  6. Sweep the decision threshold on the validation split to maximise the
     competition's macro F_0.5 metric
  7. Save model.joblib + config.json (threshold, top_k) to --model-dir

Usage:
    python -m src.train --train-dir dataset/train --model-dir model
"""
import argparse
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import build_all_candidates
from .features import build_feature_frame, FEATURE_COLUMNS
from .evaluate import macro_f_beta


def _load_train(train_dir):
    s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    return s1, s2, s3, gt


def _ground_truth_map(gt_df):
    gt_map = {}
    for s1_id, matched in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        ids = set(x for x in str(matched).split(",") if x)
        gt_map[s1_id] = ids
    return gt_map


def _candidates_to_pairs_df(candidates):
    rows = []
    for s1_id, cand_set in candidates.items():
        for cand_id in cand_set:
            rows.append((s1_id, cand_id))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def _build_labeled_features(s1, s2, s3, gt_map, top_k):
    candidates = build_all_candidates(s1, s2, s3, top_k=top_k)
    pairs_df = _candidates_to_pairs_df(candidates)

    if len(pairs_df) == 0:
        empty = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMNS + ["label"])
        return empty, candidates

    pairs_df["is_source2"] = pairs_df["candidate_entity_id"].str.startswith("S2-")
    p2 = pairs_df[pairs_df["is_source2"]].drop(columns=["is_source2"])
    p3 = pairs_df[~pairs_df["is_source2"]].drop(columns=["is_source2"])

    feat2 = build_feature_frame(p2, s1, s2)
    feat3 = build_feature_frame(p3, s1, s3)
    all_feats = pd.concat([feat2, feat3], ignore_index=True)

    all_feats["label"] = all_feats.apply(
        lambda r: int(r["candidate_entity_id"] in gt_map.get(r["source1_entity_id"], set())),
        axis=1,
    )
    return all_feats, candidates


def _threshold_to_predictions(feats_with_prob, threshold, candidates):
    """Build {source1_entity_id: set(matched_ids)}, keeping every
    Source-1 id present (even with an empty prediction / no candidates)."""
    preds = {eid: set() for eid in candidates.keys()}
    if len(feats_with_prob) == 0:
        return preds
    positive = feats_with_prob[feats_with_prob["prob"] >= threshold]
    for s1_id, cand_id in zip(positive["source1_entity_id"], positive["candidate_entity_id"]):
        preds.setdefault(s1_id, set()).add(cand_id)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--top-k", type=int, default=20,
                         help="Neighbours per source pulled by blocking (higher = better recall, slower).")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-entities", type=int, default=40000,
                         help="Cap on number of S1 entities to train/validate on for fast iteration.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    s1, s2, s3, gt = _load_train(args.train_dir)
    gt_map = _ground_truth_map(gt)

    if args.max_train_entities and len(s1) > args.max_train_entities:
        s1 = s1.sample(n=args.max_train_entities, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(s1)} Source 1 entities for fast baseline training.")

    rng = np.random.RandomState(args.seed)
    s1_ids = s1["entity_id"].to_numpy().copy()
    rng.shuffle(s1_ids)
    n_val = max(1, int(len(s1_ids) * args.val_fraction))
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"Train entities: {len(s1_train)}  Val entities: {len(s1_val)}")

    train_feats, _ = _build_labeled_features(s1_train, s2, s3, gt_map, args.top_k)
    val_feats, val_candidates = _build_labeled_features(s1_val, s2, s3, gt_map, args.top_k)

    print(f"Train pairs: {len(train_feats)}  positives: {int(train_feats['label'].sum())}")
    print(f"Val pairs: {len(val_feats)}  positives: {int(val_feats['label'].sum())}")

    X_train = train_feats[FEATURE_COLUMNS]
    y_train = train_feats["label"]

    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    # ---- Threshold tuning on the validation split to maximise macro F0.5 ----
    X_val = val_feats[FEATURE_COLUMNS]
    val_feats = val_feats.copy()
    val_feats["prob"] = model.predict_proba(X_val)[:, 1] if len(X_val) else []

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.05, 0.96, 0.05):
        preds = _threshold_to_predictions(val_feats, threshold, val_candidates)
        score = macro_f_beta(preds, gt_map, beta=0.5)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    print(f"Best validation F0.5 = {best_score:.4f} at threshold = {best_threshold:.2f}")

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump({"threshold": best_threshold, "top_k": args.top_k, "val_f0.5": best_score}, f, indent=2)

    print(f"Saved model + config to {args.model_dir}/")


if __name__ == "__main__":
    main()
