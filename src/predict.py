"""
Inference script using trained LightGBM entity matcher and multi-channel inverted indexing.

Usage:
    python3 -m src.predict --test-dir dataset/test --model-dir model --output-dir output
"""
import argparse
import gc
import json
import os
import shutil
import time
from collections import defaultdict
import joblib
import numpy as np
import polars as pl

from .normalize import norm_country, clean_name, core_name, clean_addr, extract_addr_features
from .blocking import build_inverted_index
from .features import compute_pairwise_features, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Run LightGBM Inference for Business Entity Resolution")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory containing test TSV files")
    parser.add_argument("--model-dir", default="model", help="Directory containing trained model and config")
    parser.add_argument("--output-dir", default="output", help="Directory to save output TSV files")
    parser.add_argument("--batch-size", type=int, default=500_000, help="Batch size for streaming target files")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" Amazon ML Challenge 2026 — LightGBM Entity Matcher Inference")
    print("=" * 80, flush=True)

    # 1. Load Model & Config
    model_path = os.path.join(args.model_dir, "lgbm_entity_matcher.joblib")
    config_path = os.path.join(args.model_dir, "config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first via python3 -m src.train")

    print(f"[*] Loading model from {model_path}...", flush=True)
    clf = joblib.load(model_path)
    with open(config_path) as f:
        config = json.load(f)
    tau = float(config.get("threshold", 0.90))
    print(f"[+] Model loaded. Calibrated decision threshold: tau = {tau:.3f}", flush=True)

    # 2. Load & Index Source 1 Queries
    s1_path = os.path.join(args.test_dir, "test_source1.tsv")
    print(f"\n[Step 1/3] Loading and indexing Source 1 queries from {s1_path}...", flush=True)
    t0 = time.time()
    s1_df = pl.read_csv(s1_path, separator="\t")
    n_s1 = len(s1_df)

    s1_eids = s1_df["entity_id"].to_list()
    s1_raw_c = s1_df["country"].to_list()
    s1_raw_nm = s1_df["business_name"].to_list()
    s1_raw_ad = s1_df["business_address"].to_list()
    del s1_df
    gc.collect()

    s1_cc = [norm_country(x) for x in s1_raw_c]
    s1_cn = [clean_name(x) for x in s1_raw_nm]
    s1_ca = [clean_addr(s1_raw_ad[i], s1_cc[i]) for i in range(n_s1)]
    s1_cr = [core_name(x) for x in s1_cn]

    s1_nums, s1_hnums = [], []
    for i in range(n_s1):
        _, hnum, nums_set, _, _ = extract_addr_features(s1_raw_ad[i], s1_ca[i], s1_cc[i])
        s1_nums.append(nums_set)
        s1_hnums.append(hnum)

    s1_lookup = [
        (s1_cn[i], s1_ca[i], s1_cr[i], s1_nums[i], s1_hnums[i], s1_cc[i])
        for i in range(n_s1)
    ]

    index = build_inverted_index(s1_eids, s1_raw_c, s1_raw_nm, s1_raw_ad, s1_ca)
    print(f"[+] Loaded and indexed {n_s1:,} queries ({len(index):,} active keys) in {time.time()-t0:.1f}s.", flush=True)

    # 3. Stream Targets & ML Scoring
    print("\n[Step 2/3] Streaming targets & scoring candidate pairs with LightGBM...", flush=True)
    matches_dict = defaultdict(list)
    candidates_dict = defaultdict(list)

    def stream_and_score(path, label):
        t_src = time.time()
        print(f"[*] Processing {label} from {path}...", flush=True)
        reader = pl.read_csv_batched(path, separator="\t", batch_size=args.batch_size)
        total_rows = 0
        total_pairs = 0

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

            batch_pairs_s1 = []
            batch_pairs_t = []
            batch_features = []

            for t_id, c, nm, ad in zip(eids, countries, names, addrs):
                nc = norm_country(c)
                cn = clean_name(nm)
                ca = clean_addr(ad, nc)
                cr = core_name(cn)

                matched_s1 = set()
                if cn:
                    k1 = ("N_EXACT", nc, cn)
                    if k1 in index:
                        matched_s1.update(index[k1])
                    if cr and cr != cn:
                        k2 = ("N_CORE", nc, cr)
                        if k2 in index:
                            matched_s1.update(index[k2])
                _, hnum, nums_set, swords, multi_h = extract_addr_features(ad, ca, nc)
                for h in multi_h:
                    if h and swords:
                        k_ns = ("NUM_STREET", nc, f"{h}_{swords[0]}")
                        if k_ns in index:
                            matched_s1.update(index[k_ns])
                    if h and cn:
                        w0 = cn.split()[0]
                        if len(w0) >= 3:
                            k_na = ("NA_NUM", nc, f"{h}_{w0}")
                            if k_na in index:
                                matched_s1.update(index[k_na])

                if not matched_s1:
                    continue

                t_meta = (cn, ca, cr, nums_set, hnum, nc)
                for s1_idx in matched_s1:
                    feats = compute_pairwise_features(s1_lookup[s1_idx], t_meta)
                    batch_pairs_s1.append(s1_idx)
                    batch_pairs_t.append(t_id)
                    batch_features.append(feats)

            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                probs = clf.predict_proba(X_batch)[:, 1]
                total_pairs += len(probs)

                for s1_idx, t_id, prob in zip(batch_pairs_s1, batch_pairs_t, probs):
                    candidates_dict[s1_idx].append(t_id)
                    if prob >= (tau - 0.05):
                        matches_dict[s1_idx].append((t_id, float(prob)))

            if total_rows % 1_000_000 == 0:
                print(f"     {label}: {total_rows:,} rows, {total_pairs:,} candidate pairs scored ({time.time()-t_src:.1f}s)", flush=True)

        print(f"[+] {label}: {total_rows:,} rows, {total_pairs:,} candidates scored in {time.time()-t_src:.1f}s.", flush=True)
        gc.collect()

    s2_path = os.path.join(args.test_dir, "test_source2.tsv")
    s3_path = os.path.join(args.test_dir, "test_source3.tsv")
    stream_and_score(s2_path, "Source 2")
    stream_and_score(s3_path, "Source 3")
    del index
    gc.collect()

    # 4. Generate Output TSV Files
    print("\n[Step 3/3] Generating Submission TSVs...", flush=True)
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    n_with_match = 0
    total_matches = 0

    with open(match_path, "w", encoding="utf-8") as f_match, open(cand_path, "w", encoding="utf-8") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, eid in enumerate(s1_eids):
            m_list = matches_dict.get(idx, [])
            c_list = candidates_dict.get(idx, [])

            # Deduplicate candidate IDs preserving order
            cands = []
            seen_cands = set()
            for cid in c_list:
                if cid not in seen_cands:
                    seen_cands.add(cid)
                    cands.append(cid)
            cands = cands[:15]  # Cap candidate pairs

            top = []
            if m_list:
                best_prob = max(p for _, p in m_list)
                if best_prob >= tau:
                    # Adaptive windowing around top probability
                    top_candidates = [tid for tid, p in sorted(m_list, key=lambda x: -x[1]) if p >= max(tau, best_prob - 0.05)][:5]
                    seen_top = set()
                    for tid in top_candidates:
                        if tid not in seen_top:
                            seen_top.add(tid)
                            top.append(tid)

            # Ensure all matched IDs are present in candidates
            for tid in top:
                if tid not in seen_cands:
                    seen_cands.add(tid)
                    cands.append(tid)

            if top:
                n_with_match += 1
                total_matches += len(top)
                f_match.write(f"{eid}\t{','.join(top)}\n")
            else:
                f_match.write(f"{eid}\t\n")

            if cands:
                f_cand.write(f"{eid}\t{','.join(cands)}\n")
            else:
                f_cand.write(f"{eid}\t\n")

            if (idx + 1) % 500_000 == 0:
                print(f"     Written {idx+1:,}/{n_s1:,} entities...", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[✔] LightGBM Inference Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"    Saved Matching   : {match_path} ({os.path.getsize(match_path)/(1024*1024):.2f} MB)")
    print(f"    Saved Candidates : {cand_path} ({os.path.getsize(cand_path)/(1024*1024):.2f} MB)")
    print(f"    Entities Matched : {n_with_match:,} / {n_s1:,} ({n_with_match/n_s1*100:.1f}%)")
    print(f"    Singletons       : {n_s1 - n_with_match:,} ({(n_s1 - n_with_match)/n_s1*100:.1f}%)")
    print(f"    Total Matches    : {total_matches:,}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
