import os
import sys
import json
import torch
import numpy as np
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz
import pandas as pd

from evaluate import macro_f_beta

# (Copy inference feature functions)
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

import re
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

# Need to load the model
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1   = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.drop1(self.act1(self.norm1(self.fc1(x))))
        x = self.drop2(self.act2(self.norm2(self.fc2(x))))
        return x + res

class DeepEntityMatcher(nn.Module):
    def __init__(self, in_features=11, hidden_dim=128, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

print("Loading ground truth...")
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
ground_truth = {}
for _, row in gt_df.iterrows():
    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]
    if pd.notna(matched) and matched.strip():
        ground_truth[s1_id] = set(matched.split(","))
    else:
        ground_truth[s1_id] = set()

# To calibrate quickly, let's take a sample of 20,000 entities from ground truth
import random
sample_eids = set(random.sample(list(ground_truth.keys()), 20000))
sampled_gt = {k: v for k, v in ground_truth.items() if k in sample_eids}

print("Loading dataset for evaluation...")
df_train = pd.read_csv("cleaned_dataset/training_data.csv")
df_val = df_train[df_train["source1_entity_id"].isin(sample_eids)].copy()

features = df_val[[
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio",
    "core_name_ratio", "address_ratio", "address_token_set_ratio",
    "address_number_match", "country_match",
    "name_exact_match", "core_name_exact_match", "address_exact_match"
]].to_numpy().astype(np.float32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = DeepEntityMatcher().to(device)
model.load_state_dict(torch.load("model/model_m4.pt", map_location=device))
model.eval()

print("Scoring candidates...")
with torch.no_grad():
    feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    probs = torch.sigmoid(model(feat_tensor)).cpu().numpy()

df_val["prob"] = probs

# Calibrate across thresholds 0.05 to 0.95
print("Calibrating macro F0.5...")
best_th = 0.5
best_f = 0.0

for th in np.arange(0.01, 0.99, 0.02):
    predictions = defaultdict(set)
    for row in df_val[df_val["prob"] >= th].itertuples():
        predictions[row.source1_entity_id].add(row.matched_entity_id)
        
    f_score = macro_f_beta(predictions, sampled_gt, beta=0.5)
    print(f"Threshold: {th:.2f} | Macro-F0.5: {f_score:.4f}")
    if f_score > best_f:
        best_f = f_score
        best_th = th

print(f"\nOptimal Macro Threshold: {best_th:.2f} (F0.5 = {best_f:.4f})")
