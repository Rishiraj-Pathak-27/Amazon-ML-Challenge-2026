"""
Run inference on the test set and generate:
    output/candidate_pairs.tsv   -- blocking stage output
    output/matching_results.tsv  -- final matched entities

Fast, scalable country-partitioned multi-key pipeline.
Ultra low-memory footprint (< 4.5 GB peak RAM) using sparse defaultdicts,
batched streaming, and aggressive memory reclamation.

Usage:
    python -m src.predict --test-dir dataset/test --output-dir output
"""
import argparse
import gc
import os
import re
import time
from collections import defaultdict
import polars as pl
from rapidfuzz import fuzz

from .normalize import normalize_country, LEGAL_SUFFIXES, _ADDRESS_ABBREVIATIONS


def _clean_name(s):
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    words = [w for w in s.split() if w not in LEGAL_SUFFIXES]
    return " ".join(words)


def _clean_addr(s):
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    words = [_ADDRESS_ABBREVIATIONS.get(w, w) for w in s.split()]
    return " ".join(words)


def run_prediction(test_dir="dataset/test", output_dir="output", max_cands_per_s1=30, batch_size=500_000):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"[*] Loading Source 1 from {s1_path}...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t")
    n_s1 = len(s1_df)
    print(f"[+] Loaded {n_s1:,} Source 1 entities in {time.time() - t_start:.2f}s.", flush=True)

    eids = s1_df["entity_id"].to_list()
    countries = [normalize_country(c) for c in s1_df["country"].to_list()]
    raw_names = s1_df["business_name"].to_list()
    raw_addrs = s1_df["business_address"].to_list()
    clean_names = [_clean_name(x) for x in raw_names]
    clean_addrs = [_clean_addr(x) for x in raw_addrs]

    # Free raw DataFrame memory immediately
    del s1_df, raw_names, raw_addrs
    gc.collect()

    print("[*] Building multi-channel inverted index...", flush=True)
    t_idx = time.time()
    index_N = defaultdict(list)
    index_A = defaultdict(list)
    index_NA = defaultdict(list)
    index_N2 = defaultdict(list)

    for idx in range(n_s1):
        c = countries[idx]
        cn = clean_names[idx]
        ca = clean_addrs[idx]
        if cn:
            index_N[(c, cn)].append(idx)
            words = cn.split()
            if len(words) >= 2:
                index_N2[(c, " ".join(words[:2]))].append(idx)
        if ca:
            index_A[(c, ca)].append(idx)
            nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
            if nums and cn:
                first_w = cn.split()[0]
                index_NA[(c, f"{nums[0]}_{first_w}")].append(idx)

    # Free country strings since index has absorbed them
    del countries
    gc.collect()

    # Prune overly large N2 buckets to prevent false candidate explosion
    index_N2 = {k: v for k, v in index_N2.items() if len(v) <= 25}

    print(f"[+] Index built in {time.time() - t_idx:.2f}s! (N: {len(index_N):,}, A: {len(index_A):,}, NA: {len(index_NA):,})", flush=True)

    # Use sparse defaultdicts: entities without candidates or matches consume 0 bytes
    candidates_dict = defaultdict(set)
    matches_dict = defaultdict(set)

    # Function to scan another source in memory-safe batches
    def process_source_batched(src_path, src_label):
        t_src = time.time()
        print(f"\n[*] Processing {src_label} from {src_path} in streaming batches...", flush=True)

        reader = pl.read_csv_batched(src_path, separator="\t", batch_size=batch_size)
        total_rows = 0

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
            batch_len = len(batch_df)
            total_rows += batch_len

            src_eids = batch_df["entity_id"].to_list()
            src_countries = [normalize_country(c) for c in batch_df["country"].to_list()]
            src_cnames = [_clean_name(x) for x in batch_df["business_name"].to_list()]
            src_caddrs = [_clean_addr(x) for x in batch_df["business_address"].to_list()]

            del batch_df

            # Matching loop over this batch
            for eid, c, cn, ca in zip(src_eids, src_countries, src_cnames, src_caddrs):
                # 1. Exact Name Channel (high precision)
                if cn and (c, cn) in index_N:
                    for idx in index_N[(c, cn)]:
                        matches_dict[idx].add(eid)
                        candidates_dict[idx].add(eid)

                # 2. Exact Address Channel
                if ca and (c, ca) in index_A:
                    for idx in index_A[(c, ca)]:
                        matches_dict[idx].add(eid)
                        candidates_dict[idx].add(eid)

                # 3. Numeric + Name Prefix Channel
                nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
                if nums and cn:
                    first_w = cn.split()[0]
                    k = (c, f"{nums[0]}_{first_w}")
                    if k in index_NA:
                        for idx in index_NA[k]:
                            if fuzz.ratio(clean_names[idx], cn) >= 70:
                                matches_dict[idx].add(eid)
                                candidates_dict[idx].add(eid)
                            elif len(candidates_dict[idx]) < max_cands_per_s1:
                                candidates_dict[idx].add(eid)

                # 4. First 2 Words Name Channel
                if cn:
                    words = cn.split()
                    if len(words) >= 2:
                        k = (c, " ".join(words[:2]))
                        if k in index_N2:
                            for idx in index_N2[k]:
                                s1_ca = clean_addrs[idx]
                                if fuzz.ratio(clean_names[idx], cn) >= 85 or (ca and s1_ca and fuzz.ratio(s1_ca, ca) >= 75):
                                    matches_dict[idx].add(eid)
                                    candidates_dict[idx].add(eid)
                                elif len(candidates_dict[idx]) < max_cands_per_s1:
                                    candidates_dict[idx].add(eid)

            del src_eids, src_countries, src_cnames, src_caddrs
            print(f"     [Progress] {src_label}: {total_rows:,} rows scanned in {time.time() - t_src:.1f}s", flush=True)

        print(f"[+] Completed {src_label} ({total_rows:,} rows) in {time.time() - t_src:.2f}s.", flush=True)
        gc.collect()

    # Process Source 2 and Source 3 with batched streaming
    process_source_batched(s2_path, "Source 2")
    process_source_batched(s3_path, "Source 3")

    # CRITICAL: Reclaim index memory (~3 GB) IMMEDIATELY before saving files
    print("\n[*] Freeing index memory before saving outputs...", flush=True)
    del index_N, index_A, index_NA, index_N2, clean_names, clean_addrs
    gc.collect()

    # Step 1: Write matching_results.tsv first
    match_path = os.path.join(output_dir, "matching_results.tsv")
    print(f"[*] Writing {match_path}...", flush=True)
    n_with_match = 0
    MAX_MATCHES_PER_S1 = 11  # Ground truth max is 11; prevents runaway false positives & guarantees file < 512MB
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for idx, eid in enumerate(eids):
            if idx in matches_dict:
                n_with_match += 1
                m = sorted(matches_dict[idx])[:MAX_MATCHES_PER_S1]
                match_str = ",".join(m)
                f.write(f"{eid}\t{match_str}\n")
            else:
                f.write(f"{eid}\t\n")

    # Strict File Size Verification (Portal constraint: max 512 MB)
    match_size_mb = os.path.getsize(match_path) / (1024 * 1024)
    print(f"[+] File Size Check: {match_path} is {match_size_mb:.2f} MB (Portal limit: 512.0 MB | {match_size_mb/512.0*100:.1f}%)", flush=True)
    if match_size_mb > 512.0:
        raise ValueError(f"CRITICAL: {match_path} size ({match_size_mb:.2f} MB) exceeds the 512 MB portal limit!")

    n_singletons = n_s1 - n_with_match

    # Reclaim matches memory
    del matches_dict
    gc.collect()

    # Step 2: Write candidate_pairs.tsv
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    print(f"[*] Writing {cand_path}...", flush=True)
    total_cands = 0
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for idx, eid in enumerate(eids):
            if idx in candidates_dict:
                cands = candidates_dict[idx]
                total_cands += len(cands)
                cand_str = ",".join(sorted(cands))
                f.write(f"{eid}\t{cand_str}\n")
            else:
                f.write(f"{eid}\t\n")

    # Reclaim candidates memory
    del candidates_dict, eids
    gc.collect()

    print("\n" + "=" * 60, flush=True)
    print(f"[+] Inference complete in {time.time() - t_start:.2f} seconds!", flush=True)
    print(f"Total Source 1 entities: {n_s1:,}", flush=True)
    print(f"Entities with >= 1 match: {n_with_match:,} ({n_with_match / n_s1 * 100:.1f}%)", flush=True)
    print(f"Singletons (0 matches): {n_singletons:,} ({n_singletons / n_s1 * 100:.1f}%)", flush=True)
    print(f"Total candidate pairs: {total_cands:,} (avg {total_cands / n_s1:.2f} per entity)", flush=True)
    print(f"Saved: {cand_path}", flush=True)
    print(f"Saved: {match_path}", flush=True)
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--batch-size", type=int, default=500000)
    args = parser.parse_args()

    run_prediction(test_dir=args.test_dir, output_dir=args.output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
