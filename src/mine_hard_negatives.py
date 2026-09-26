import os
import sys
import gc
import time
import json
import re
import argparse
import random
from collections import defaultdict

import numpy as np
import polars as pl
from rapidfuzz import fuzz
import pandas as pd

# (Copy functions from run_pipeline_v2.py)
LEGAL_SUFFIXES = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "llc", "gmbh", "sa", "sarl", "sl", "bv", "co", "company",
    "pvt", "private", "plc", "srl", "pty", "holdings", "group"
}

_ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "sq": "square", "ste": "suite", "apt": "apartment", "dept": "department",
    "fl": "floor", "bldg": "building", "hwy": "highway", "pkwy": "parkway",
    "str": "strasse", "pl": "place", "terr": "terrace",
}

_COUNTRY_MAP = {
    "usa": "us", "united states": "us", "uk": "gb",
    "united kingdom": "gb", "great britain": "gb",
    "deutschland": "de", "germany": "de", "france": "fr",
    "italia": "it", "italy": "it", "espana": "es", "spain": "es",
    "india": "in", "china": "cn", "japan": "jp", "canada": "ca",
    "australia": "au", "brazil": "br", "brasil": "br", "mexico": "mx",
}

def normalize_country(c):
    if not c or not isinstance(c, str):
        return ""
    c_clean = c.strip().lower()
    return _COUNTRY_MAP.get(c_clean, c_clean)

def _clean_name(s):
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"[^\w\s]", " ", str(s).lower()).strip()

def _clean_addr(s):
    if not s or not isinstance(s, str):
        return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(_ADDR_ABBR.get(w, w) for w in s.split())

def compute_inference_features(s1_name, s1_addr, s1_c, c_name, c_addr, c_c):
    s1_nm, c_nm = _clean_name(s1_name), _clean_name(c_name)
    s1_ad, c_ad = _clean_addr(s1_addr), _clean_addr(c_addr)
    s1_core = " ".join(w for w in s1_nm.split() if w not in LEGAL_SUFFIXES)
    c_core  = " ".join(w for w in c_nm.split()  if w not in LEGAL_SUFFIXES)
    s1_nums = " ".join(re.findall(r"\d+", s1_ad))
    c_nums  = " ".join(re.findall(r"\d+", c_ad))
    s1_ctry = normalize_country(s1_c)
    c_ctry  = normalize_country(c_c)
    return [
        fuzz.ratio(s1_nm, c_nm) / 100.0,
        fuzz.token_set_ratio(s1_nm, c_nm) / 100.0,
        fuzz.token_sort_ratio(s1_nm, c_nm) / 100.0,
        fuzz.ratio(s1_core, c_core) / 100.0,
        fuzz.ratio(s1_ad, c_ad) / 100.0,
        fuzz.token_set_ratio(s1_ad, c_ad) / 100.0,
        int(bool(s1_nums) and s1_nums == c_nums),
        int(bool(s1_ctry) and s1_ctry == c_ctry),
        int(bool(s1_nm) and s1_nm == c_nm),
        int(bool(s1_core) and s1_core == c_core),
        int(bool(s1_ad) and s1_ad == c_ad),
    ]

# 1. Load ground truth positive pairs
print("Loading ground truth positive pairs...")
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
positive_pairs = set()
for _, row in gt_df.iterrows():
    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]
    if pd.notna(matched) and matched.strip():
        for mid in matched.split(","):
            positive_pairs.add((s1_id, mid))

print(f"Loaded {len(positive_pairs)} positive pairs from ground truth.")

# 2. Load S1 data and build index
print("Loading train_source1.tsv and building index...")
s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
n_s1 = len(s1_df)

s1_eids      = s1_df["entity_id"].to_list()
s1_countries = [normalize_country(c) for c in s1_df["country"].to_list()]
s1_cnames    = [_clean_name(x)  for x in s1_df["business_name"].to_list()]
s1_caddrs    = [_clean_addr(x)  for x in s1_df["business_address"].to_list()]

s1_raw_data = {
    eid: (rn, ra, rc) 
    for eid, rn, ra, rc in zip(s1_eids, s1_df["business_name"].to_list(), s1_df["business_address"].to_list(), s1_df["country"].to_list())
}

index_N  = defaultdict(list)
index_A  = defaultdict(list)
index_NA = defaultdict(list)
index_N2 = defaultdict(list)

for idx in range(n_s1):
    c, cn, ca = s1_countries[idx], s1_cnames[idx], s1_caddrs[idx]
    if cn:
        index_N[(c, cn)].append(idx)
        words = cn.split()
        if len(words) >= 2:
            index_N2[(c, " ".join(words[:2]))].append(idx)
    if ca:
        index_A[(c, ca)].append(idx)
        nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
        if nums and cn:
            index_NA[(c, f"{nums[0]}_{cn.split()[0]}")].append(idx)

del s1_countries
index_N2 = {k: v for k, v in index_N2.items() if len(v) <= 25}
gc.collect()

print("Scanning Source 2 and 3 for hard negatives...")
candidates_dict = defaultdict(list)

def process_file_for_hard_negatives(path):
    print(f"Processing {path}...")
    reader = pl.read_csv_batched(path, separator="\t", batch_size=500_000)
    cand_data = {}
    
    batch_idx = 0
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        b = batches[0]
        batch_idx += 1
        print(f"  Batch {batch_idx} ({len(b)} rows)")
        
        eids = b["entity_id"].to_list()
        countries = [normalize_country(x) for x in b["country"].to_list()]
        cnames = [_clean_name(x) for x in b["business_name"].to_list()]
        caddrs = [_clean_addr(x) for x in b["business_address"].to_list()]
        raw_names = b["business_name"].to_list()
        raw_addrs = b["business_address"].to_list()
        raw_ctry  = b["country"].to_list()
        
        for eid, c, cn, ca, rn, ra, rc in zip(eids, countries, cnames, caddrs, raw_names, raw_addrs, raw_ctry):
            cand_data[eid] = (rn, ra, rc)
            
            def _add(idx):
                s1_eid = s1_eids[idx]
                if (s1_eid, eid) not in positive_pairs:
                    # It's a hard negative!
                    # Only keep up to 4 hard negatives per S1 entity
                    if len(candidates_dict[s1_eid]) < 4:
                        candidates_dict[s1_eid].append(eid)
                        
            if cn and (c, cn) in index_N:
                for idx in index_N[(c, cn)]: _add(idx)
            if ca and (c, ca) in index_A:
                for idx in index_A[(c, ca)]: _add(idx)
            nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
            if nums and cn:
                k = (c, f"{nums[0]}_{cn.split()[0]}")
                if k in index_NA:
                    for idx in index_NA[k]:
                        if fuzz.ratio(s1_cnames[idx], cn) >= 70:
                            _add(idx)
            if cn:
                words = cn.split()
                if len(words) >= 2:
                    k = (c, " ".join(words[:2]))
                    if k in index_N2:
                        for idx in index_N2[k]:
                            s1_ca = s1_caddrs[idx]
                            if (fuzz.ratio(s1_cnames[idx], cn) >= 85 or (ca and s1_ca and fuzz.ratio(s1_ca, ca) >= 75)):
                                _add(idx)
                                
    return cand_data

t0 = time.time()
cand_data_s2 = process_file_for_hard_negatives("dataset/train/train_source2.tsv")
cand_data_s3 = process_file_for_hard_negatives("dataset/train/train_source3.tsv")
print(f"Candidate scanning finished in {time.time() - t0:.1f}s")

cand_data_all = {**cand_data_s2, **cand_data_s3}
del cand_data_s2, cand_data_s3
gc.collect()

print("Generating dataset...")
training_records = []
FEATURE_COLUMNS = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio",
    "core_name_ratio", "address_ratio", "address_token_set_ratio",
    "address_number_match", "country_match",
    "name_exact_match", "core_name_exact_match", "address_exact_match",
]

# Add positives
pos_count = 0
for s1_eid, cid in positive_pairs:
    if s1_eid in s1_raw_data and cid in cand_data_all:
        rn1, ra1, rc1 = s1_raw_data[s1_eid]
        rn2, ra2, rc2 = cand_data_all[cid]
        feats = compute_inference_features(rn1, ra1, rc1, rn2, ra2, rc2)
        training_records.append([s1_eid, cid, 1] + feats)
        pos_count += 1

# Add hard negatives
neg_count = 0
for s1_eid, cids in candidates_dict.items():
    for cid in cids:
        if s1_eid in s1_raw_data and cid in cand_data_all:
            rn1, ra1, rc1 = s1_raw_data[s1_eid]
            rn2, ra2, rc2 = cand_data_all[cid]
            feats = compute_inference_features(rn1, ra1, rc1, rn2, ra2, rc2)
            training_records.append([s1_eid, cid, 0] + feats)
            neg_count += 1

print(f"Generated {pos_count} positive and {neg_count} hard negative records.")

df_train = pd.DataFrame(training_records, columns=["source1_entity_id", "matched_entity_id", "label"] + FEATURE_COLUMNS)
df_train.to_csv("cleaned_dataset/training_data.csv", index=False)
print("Saved to cleaned_dataset/training_data.csv")
