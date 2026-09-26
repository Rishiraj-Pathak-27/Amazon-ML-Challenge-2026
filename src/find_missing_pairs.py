import json
import time
import re
from collections import defaultdict
import polars as pl
import pandas as pd
from rapidfuzz import fuzz

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
    if not c or not isinstance(c, str): return ""
    c_clean = c.strip().lower()
    return _COUNTRY_MAP.get(c_clean, c_clean)

def _clean_name(s):
    if not s or not isinstance(s, str): return ""
    return re.sub(r"[^\w\s]", " ", str(s).lower()).strip()

def _clean_addr(s):
    if not s or not isinstance(s, str): return ""
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(_ADDR_ABBR.get(w, w) for w in s.split())

gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t").sample(n=10000, random_state=42)
gt_pairs = set()
s1_set = set()
for _, row in gt_df.iterrows():
    s1_id = row["source1_entity_id"]
    s1_set.add(s1_id)
    matched = row["matched_entity_ids"]
    if pd.notna(matched) and matched.strip():
        for mid in matched.split(","):
            gt_pairs.add((s1_id, mid))

s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
s1_df = s1_df.filter(pl.col("entity_id").is_in(list(s1_set)))
n_s1 = len(s1_df)

s1_eids      = s1_df["entity_id"].to_list()
s1_countries = [normalize_country(c) for c in s1_df["country"].to_list()]
s1_cnames    = [_clean_name(x)  for x in s1_df["business_name"].to_list()]
s1_caddrs    = [_clean_addr(x)  for x in s1_df["business_address"].to_list()]

s1_raw = {eid: (rn, ra, rc) for eid, rn, ra, rc in zip(s1_eids, s1_df["business_name"].to_list(), s1_df["business_address"].to_list(), s1_df["country"].to_list())}

index_N  = defaultdict(list)
index_W  = defaultdict(list)
index_NoSpace = defaultdict(list) 
index_Num = defaultdict(list) 

for idx in range(n_s1):
    c, cn, ca = s1_countries[idx], s1_cnames[idx], s1_caddrs[idx]
    if cn:
        index_N[(c, cn)].append(idx)
        for word in cn.split():
            if len(word) > 2:
                index_W[(c, word)].append(idx)
        
        cn_nospace = cn.replace(" ", "")
        if len(cn_nospace) >= 5:
            index_NoSpace[(c, cn_nospace)].append(idx)
            
    if ca:
        nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
        for num in nums:
            if len(num) >= 2:
                index_Num[(c, num)].append(idx)

index_W = {k: v for k, v in index_W.items() if len(v) <= 100}
index_Num = {k: v for k, v in index_Num.items() if len(v) <= 50}

found_pairs = set()
cand_raw = {}

def scan_file(path):
    reader = pl.read_csv_batched(path, separator="\t", batch_size=500_000)
    while True:
        batches = reader.next_batches(1)
        if not batches: break
        b = batches[0]
        
        eids = b["entity_id"].to_list()
        countries = [normalize_country(x) for x in b["country"].to_list()]
        cnames = [_clean_name(x) for x in b["business_name"].to_list()]
        caddrs = [_clean_addr(x) for x in b["business_address"].to_list()]
        raw_names = b["business_name"].to_list()
        raw_addrs = b["business_address"].to_list()
        raw_ctry  = b["country"].to_list()
        
        for eid, c, cn, ca, rn, ra, rc in zip(eids, countries, cnames, caddrs, raw_names, raw_addrs, raw_ctry):
            if eid not in cand_raw:
                cand_raw[eid] = (rn, ra, rc)
                
            def _add(idx):
                s1_eid = s1_eids[idx]
                if (s1_eid, eid) in gt_pairs:
                    found_pairs.add((s1_eid, eid))

            if cn and (c, cn) in index_N:
                for idx in index_N[(c, cn)]: _add(idx)
            
            if cn:
                for word in cn.split():
                    if len(word) > 2 and (c, word) in index_W:
                        for idx in index_W[(c, word)]:
                            if fuzz.ratio(s1_cnames[idx], cn) >= 65:
                                _add(idx)
                
                cn_nospace = cn.replace(" ", "").replace("com", "").replace("net", "").replace("org", "")
                if len(cn_nospace) >= 5 and (c, cn_nospace) in index_NoSpace:
                    for idx in index_NoSpace[(c, cn_nospace)]:
                        _add(idx)
                        
            if ca:
                nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
                for num in nums:
                    if len(num) >= 2 and (c, num) in index_Num:
                        for idx in index_Num[(c, num)]:
                            if fuzz.token_set_ratio(s1_caddrs[idx], ca) >= 65:
                                _add(idx)

scan_file("dataset/train/train_source2.tsv")
scan_file("dataset/train/train_source3.tsv")

missing = gt_pairs - found_pairs
print(f"Total Missing Pairs: {len(missing)}")

print("\nExamples of missing pairs:")
for s1_eid, cid in list(missing)[:10]:
    s1_rn, s1_ra, s1_rc = s1_raw.get(s1_eid, ("", "", ""))
    c_rn, c_ra, c_rc = cand_raw.get(cid, ("", "", ""))
    
    print("-" * 50)
    print(f"S1 : {s1_rn} | {s1_ra} | {s1_rc}")
    print(f"Cand: {c_rn} | {c_ra} | {c_rc}")

