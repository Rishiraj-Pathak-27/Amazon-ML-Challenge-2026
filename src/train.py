"""
Train LightGBM Entity Matcher with Mined Hard Negatives and Calibrated Thresholding.

Usage:
    python3 -m src.train --train-dir dataset/train --model-dir model --sample-size 25000
"""
import argparse
import json
import os
import random
import time
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from collections import defaultdict

from .normalize import norm_country, clean_name, core_name, clean_addr, extract_addr_features
from .blocking import build_inverted_index
from .features import compute_pairwise_features, FEATURE_NAMES
from .evaluate import macro_f_beta


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Entity Resolution Model")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory containing training TSV files")
    parser.add_argument("--model-dir", default="model", help="Directory to save trained model artifacts")
    parser.add_argument("--sample-size", type=int, default=25000, help="Number of S1 queries to sample for training")
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Fraction of queries held out for validation")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print(" Amazon ML Challenge 2026 — LightGBM Entity Matcher Training")
    print("=" * 80, flush=True)

    # 1. Load Ground Truth
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    print(f"[*] Loading Ground Truth from {gt_path}...", flush=True)
    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_map = {}
    for row in gt_df.iter_rows():
        gt_map[row[0]] = set(row[1].split(",")) if row[1] else set()

    all_s1 = list(gt_map.keys())
    sample_n = min(len(all_s1), args.sample_size)
    sampled_s1 = set(random.sample(all_s1, sample_n))
    print(f"[+] Sampled {sample_n:,} Source 1 query entities.", flush=True)

    # 2. Load Source 1
    s1_path = os.path.join(args.train_dir, "train_source1.tsv")
    print(f"[*] Loading Source 1 records from {s1_path}...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t").filter(pl.col("entity_id").is_in(list(sampled_s1)))
    n_s1 = len(s1_df)

    s1_eids = s1_df["entity_id"].to_list()
    s1_raw_c = s1_df["country"].to_list()
    s1_raw_nm = s1_df["business_name"].to_list()
    s1_raw_ad = s1_df["business_address"].to_list()
    del s1_df

    s1_cc = [norm_country(x) for x in s1_raw_c]
    s1_cn = [clean_name(x) for x in s1_raw_nm]
    s1_ca = [clean_addr(s1_raw_ad[i], s1_cc[i]) for i in range(n_s1)]
    s1_cr = [core_name(x) for x in s1_cn]

    s1_nums, s1_hnums = [], []
    for i in range(n_s1):
        _, hnum, nums_set, _, _ = extract_addr_features(s1_raw_ad[i], s1_ca[i], s1_cc[i])
        s1_nums.append(nums_set)
        s1_hnums.append(hnum)

    s1_lookup = {
        s1_eids[i]: (s1_cn[i], s1_ca[i], s1_cr[i], s1_nums[i], s1_hnums[i], s1_cc[i])
        for i in range(n_s1)
    }

    # Build inverted index
    print("[*] Building multi-channel inverted index...", flush=True)
    index = build_inverted_index(s1_eids, s1_raw_c, s1_raw_nm, s1_raw_ad, s1_ca)
    print(f"[+] Inverted index built with {len(index):,} active keys.", flush=True)

    # 3. Stream Candidates & Extract Features
    pairs = []
    labels = []
    s1_pair_ids = []

    def scan_target_source(path, label):
        print(f"[*] Scanning {label} for candidate pairs and hard negatives...", flush=True)
        t_src = time.time()
        reader = pl.read_csv_batched(path, separator="\t", batch_size=500_000)
        total_rows = 0

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            b = batches[0]
            total_rows += len(b)
            eids = b["entity_id"].to_list()
            countries = b["country"].to_list()
            names = b["business_name"].to_list()
            addrs = b["business_address"].to_list()
            del b

            for t_id, c, nm, ad in zip(eids, countries, names, addrs):
                nc = norm_country(c)
                cn = clean_name(nm)
                ca = clean_addr(ad, nc)
                cr = core_name(cn)

                matched_s1_indices = set()
                if cn:
                    k1 = ("N_EXACT", nc, cn)
                    if k1 in index:
                        matched_s1_indices.update(index[k1])
                    if cr and cr != cn:
                        k2 = ("N_CORE", nc, cr)
                        if k2 in index:
                            matched_s1_indices.update(index[k2])
                _, hnum, nums_set, swords, multi_h = extract_addr_features(ad, ca, nc)
                for h in multi_h:
                    if h and swords:
                        k_ns = ("NUM_STREET", nc, f"{h}_{swords[0]}")
                        if k_ns in index:
                            matched_s1_indices.update(index[k_ns])
                    if h and cn:
                        w0 = cn.split()[0]
                        if len(w0) >= 3:
                            k_na = ("NA_NUM", nc, f"{h}_{w0}")
                            if k_na in index:
                                matched_s1_indices.update(index[k_na])

                if not matched_s1_indices:
                    continue

                t_meta = (cn, ca, cr, nums_set, hnum, nc)
                for s1_idx in matched_s1_indices:
                    s1_id = s1_eids[s1_idx]
                    s1_meta = s1_lookup[s1_id]
                    feats = compute_pairwise_features(s1_meta, t_meta)
                    is_match = 1 if t_id in gt_map.get(s1_id, set()) else 0
                    pairs.append(feats)
                    labels.append(is_match)
                    s1_pair_ids.append((s1_id, t_id))

        print(f"[+] Scanned {total_rows:,} {label} rows in {time.time()-t_src:.1f}s.", flush=True)

    scan_target_source(os.path.join(args.train_dir, "train_source2.tsv"), "Source 2")
    scan_target_source(os.path.join(args.train_dir, "train_source3.tsv"), "Source 3")

    X = np.array(pairs, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"\n[+] Total Candidate Pairs: {len(y):,}")
    print(f"[+] Positives: {int(y.sum()):,} ({y.mean()*100:.2f}%) | Mined Hard Negatives: {len(y) - int(y.sum()):,}")

    # 4. Group-Shuffle Split by S1 entity
    unique_s1 = list(sampled_s1)
    random.shuffle(unique_s1)
    split_idx = int(len(unique_s1) * (1.0 - args.val_fraction))
    train_s1 = set(unique_s1[:split_idx])
    val_s1 = set(unique_s1[split_idx:])

    train_mask = np.array([s1 in train_s1 for s1, _ in s1_pair_ids])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs = [pid for pid, is_val in zip(s1_pair_ids, val_mask) if is_val]

    print(f"[+] Training Split: {len(X_train):,} pairs (Pos: {int(y_train.sum()):,})")
    print(f"[+] Validation Split: {len(X_val):,} pairs (Pos: {int(y_val.sum()):,})")

    # 5. Train LightGBM Classifier
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos

    print(f"\n[*] Training LightGBM Classifier (n_estimators={args.n_estimators}, scale_pos_weight={scale_pos_weight:.2f})...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 6. Sweep Threshold for Macro-F0.5
    print("\n[*] Calibrating threshold for Macro-F0.5 on held-out validation queries...", flush=True)
    probs_val = clf.predict_proba(X_val)[:, 1]
    val_gt = {s1: gt_map.get(s1, set()) for s1 in val_s1}

    best_tau = 0.50
    best_score = -1.0

    for tau in np.arange(0.60, 0.99, 0.02):
        tau = float(np.round(tau, 3))
        preds = {s1: set() for s1 in val_s1}
        for (s1, t), prob in zip(val_pairs, probs_val):
            if prob >= tau:
                preds[s1].add(t)
        score = macro_f_beta(preds, val_gt, beta=0.5)
        print(f"     Threshold tau = {tau:.3f} -> Validation Macro-F0.5 = {score:.4f}")
        if score > best_score:
            best_score = score
            best_tau = tau

    print(f"\n[+] Optimal Decision Threshold: tau = {best_tau:.3f} (Macro-F0.5: {best_score:.4f})")

    # Feature Importance
    print("\n[+] Feature Importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"     {name:25s}: {imp}")

    # 7. Save Model & Config
    model_file = os.path.join(args.model_dir, "lgbm_entity_matcher.joblib")
    config_file = os.path.join(args.model_dir, "config.json")
    joblib.dump(clf, model_file)

    config = {
        "model_type": "LightGBM Classifier (Gradient Boosted Trees)",
        "framework": "lightgbm",
        "license": "MIT License",
        "features": FEATURE_NAMES,
        "n_estimators": args.n_estimators,
        "threshold": float(best_tau),
        "validation_metric": "Macro-F0.5",
        "val_macro_f05": float(best_score),
        "train_pairs": int(len(X_train)),
        "val_pairs": int(len(X_val)),
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[✔] Training Successfully Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"    Saved Model   : {model_file} ({os.path.getsize(model_file)/(1024*1024):.2f} MB)")
    print(f"    Saved Config  : {config_file}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
