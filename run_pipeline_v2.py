#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 — End-to-End Pipeline V2
Local High-Performance Training, Calibration & Submission Generation

Features:
  1. Fine-tunes Deep Residual Entity Matcher on cleaned_dataset/training_data.csv
  2. Warm-starts from Amazon-ML-Submission/model_tpu.pt
  3. Uses native Apple Silicon M4 GPU / Neural Engine (MPS)
  4. Calibrates optimal threshold for Amazon ML metric (F0.5 score)
  5. Multi-key inverted index candidate generation across 1.73M test entities
  6. Outputs matching_results_V2.tsv and candidate_pairs_V2.tsv
================================================================================
"""

import os
import sys
import gc
import time
import json
import re
import argparse
from collections import defaultdict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from rapidfuzz import fuzz

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio",
    "core_name_ratio", "address_ratio", "address_token_set_ratio",
    "address_number_match", "country_match",
    "name_exact_match", "core_name_exact_match", "address_exact_match",
]
N_FEATURES = len(FEATURE_COLUMNS)

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

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# F0.5 Calibration
# ------------------------------------------------------------------------------
def calibrate_threshold(y_true, y_probs, beta=0.5):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh, best_f = 0.5, 0.0
    best_p, best_r = 0.0, 0.0
    b2 = beta ** 2

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f_score = (1 + b2) * (prec * rec) / (b2 * prec + rec)
        else:
            f_score = 0.0

        if f_score > best_f:
            best_f = f_score
            best_thresh = float(th)
            best_p, best_r = prec, rec

    return best_thresh, best_f, best_p, best_r

# ------------------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - End-to-End Pipeline V2")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size (default: 2048)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--checkpoint", type=str, default="Amazon-ML-Submission/model_tpu.pt", help="Warm-start model path")
    parser.add_argument("--dataset", type=str, default="cleaned_dataset/training_data.csv", help="Clean dataset path")
    parser.add_argument("--output-dir", type=str, default="output/submmision", help="Directory to store submission results (default: output/submmision)")
    parser.add_argument("--output-name", type=str, default=None, help="Submission output filename (default: auto-incremented matching_results_V{N}.tsv)")
    parser.add_argument("--save-candidates", "--generate-candidates", action="store_true", default=False,
                        help="Generate candidate_pairs (only needed for final submission packaging; disabled by default)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and run inference using existing model")
    args = parser.parse_args()

    # Determine auto-incrementing version
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name is None:
        version = 1
        while True:
            candidate_name = f"matching_results_V{version}.tsv"
            if not os.path.exists(os.path.join(args.output_dir, candidate_name)):
                args.output_name = candidate_name
                break
            version += 1
    else:
        # Extract version number if present to use for candidate pairs
        match = re.search(r'V(\d+)', args.output_name)
        version = int(match.group(1)) if match else 1

    cand_name = f"candidate_pairs_V{version}.tsv"

    # Hardware detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n[+] Accelerator: Apple Silicon M4 GPU / Neural Engine (MPS)", flush=True)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[+] Accelerator: NVIDIA GPU ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = torch.device("cpu")
        print("\n[*] Accelerator: CPU", flush=True)

    model = DeepEntityMatcher(in_features=N_FEATURES, hidden_dim=128, num_blocks=3, dropout=0.15).to(device)
    best_thresh = 0.5

    # ==========================================================================
    # STEP 1: Fine-Tune on Cleaned Dataset
    # ==========================================================================
    if not args.skip_training:
        print("\n" + "=" * 70)
        print("[Step 1/3] Loading Cleaned Dataset & Fine-Tuning Model...")
        print("=" * 70, flush=True)

        t_load = time.time()
        df = pl.read_csv(args.dataset)
        print(f"[+] Loaded {len(df):,} cleaned pairs in {time.time()-t_load:.2f}s", flush=True)

        labels = df["label"].to_numpy().astype(np.float32)
        groups = df["source1_entity_id"].to_numpy()
        X = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        # Group split by source1_entity_id
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_idx, val_idx = next(gss.split(X, labels, groups=groups))

        X_train, y_train = X[train_idx], labels[train_idx]
        X_val, y_val     = X[val_idx],   labels[val_idx]
        del df
        gc.collect()

        n_pos = max(1, int(y_train.sum()))
        n_neg = max(1, len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
        print(f"[*] Train: {len(X_train):,} pairs | Val: {len(X_val):,} pairs | pos_weight: {pos_weight.item():.2f}")

        # Warm-start weights
        if os.path.exists(args.checkpoint):
            print(f"[*] Loading pre-trained checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model_dict = model.state_dict()
            transferred = 0
            for k, v in ckpt.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        model_dict[k] = v
                        transferred += 1
                    elif k == "input_layer.0.weight" and v.shape[0] == model_dict[k].shape[0]:
                        model_dict[k][:, :N_FEATURES] = v[:, :N_FEATURES]
                        transferred += 1
            model.load_state_dict(model_dict)
            print(f"[+] Warm-start loaded {transferred}/{len(model_dict)} tensor blocks successfully!")

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        criterion     = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        t_train_start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_loss, n_batches = 0.0, 0
            t_ep = time.time()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                n_batches += 1

            if device.type == "mps":
                torch.mps.synchronize()

            scheduler.step()
            avg_loss = ep_loss / max(1, n_batches)
            print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] — Loss: {avg_loss:.4f} | Time: {time.time()-t_ep:.2f}s", flush=True)

        print(f"[+] Model training completed in {time.time()-t_train_start:.2f}s!")

        # Calibrate optimal threshold
        print("\n[*] Calibrating optimal threshold on validation set...")
        model.eval()
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=args.batch_size, shuffle=False)
        val_preds = []
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                val_preds.append(torch.sigmoid(model(bx)).cpu().numpy())
        val_probs = np.concatenate(val_preds)
        best_thresh, best_f05, prec, rec = calibrate_threshold(y_val, val_probs, beta=0.5)

        print(f"\n{'='*65}")
        print(f"[+] Calibration Metric (F0.5):")
        print(f"    Optimal Threshold : {best_thresh:.2f}")
        print(f"    Validation F0.5   : {best_f05:.4f}")
        print(f"    Validation Prec   : {prec:.4f}")
        print(f"    Validation Rec    : {rec:.4f}")
        print(f"{'='*65}\n")

        # Save model and config
        os.makedirs("model", exist_ok=True)
        os.makedirs("Amazon-ML-Submission/model", exist_ok=True)
        torch.save(model.state_dict(), "model/model_m4.pt")
        torch.save(model.state_dict(), "Amazon-ML-Submission/model/model_m4.pt")
        torch.save(model.state_dict(), "Amazon-ML-Submission/model_tpu.pt")

        config_data = {
            "threshold": best_thresh,
            "val_f05": best_f05,
            "val_precision": prec,
            "val_recall": rec,
            "in_features": N_FEATURES,
            "feature_columns": FEATURE_COLUMNS,
            "epochs": args.epochs,
        }
        with open("model/config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        with open("Amazon-ML-Submission/config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        print("[+] Checkpoints and configuration saved.")

    else:
        # Load existing config & weights
        if os.path.exists("Amazon-ML-Submission/config.json"):
            with open("Amazon-ML-Submission/config.json") as f:
                cfg = json.load(f)
                best_thresh = cfg.get("threshold", 0.5)
        if os.path.exists("model/model_m4.pt"):
            model.load_state_dict(torch.load("model/model_m4.pt", map_location=device))
        elif os.path.exists("Amazon-ML-Submission/model_tpu.pt"):
            model.load_state_dict(torch.load("Amazon-ML-Submission/model_tpu.pt", map_location=device), strict=False)

    # ==========================================================================
    # STEP 2: Candidate Generation on Test Set
    # ==========================================================================
    s1_path = "dataset/test/test_source1.tsv"
    s2_path = "dataset/test/test_source2.tsv"
    s3_path = "dataset/test/test_source3.tsv"

    if not os.path.exists(s1_path):
        print(f"[!] Test dataset not found at {s1_path}. Skipping inference.")
        return

    print("\n" + "=" * 70)
    print("[Step 2/3] Multi-Key Inverted Indexing & Candidate Retrieval...")
    print("=" * 70, flush=True)

    t_s1 = time.time()
    s1_df = pl.read_csv(s1_path, separator="\t")
    n_s1 = len(s1_df)
    test_eids      = s1_df["entity_id"].to_list()
    test_countries = [normalize_country(c) for c in s1_df["country"].to_list()]
    test_cnames    = [_clean_name(x)  for x in s1_df["business_name"].to_list()]
    test_caddrs    = [_clean_addr(x)  for x in s1_df["business_address"].to_list()]
    test_raw_names = s1_df["business_name"].to_list()
    test_raw_addrs = s1_df["business_address"].to_list()
    test_raw_ctry  = s1_df["country"].to_list()
    del s1_df
    gc.collect()
    print(f"[+] Loaded {n_s1:,} Test Source 1 entities in {time.time()-t_s1:.2f}s")

    index_N  = defaultdict(list)
    index_A  = defaultdict(list)
    index_NA = defaultdict(list)
    index_N2 = defaultdict(list)

    for idx in range(n_s1):
        c, cn, ca = test_countries[idx], test_cnames[idx], test_caddrs[idx]
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

    del test_countries
    index_N2 = {k: v for k, v in index_N2.items() if len(v) <= 25}
    gc.collect()

    candidates_dict = defaultdict(dict)

    def process_test_file(path, label):
        t_start = time.time()
        print(f"[*] Scanning {label} ({path})...", flush=True)
        reader = pl.read_csv_batched(path, separator="\t", batch_size=500_000)
        total_rows = 0

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            b = batches[0]
            total_rows += len(b)
            eids = b["entity_id"].to_list()
            countries = [normalize_country(x) for x in b["country"].to_list()]
            cnames = [_clean_name(x) for x in b["business_name"].to_list()]
            caddrs = [_clean_addr(x) for x in b["business_address"].to_list()]
            raw_names = b["business_name"].to_list()
            raw_addrs = b["business_address"].to_list()
            raw_ctry  = b["country"].to_list()
            del b

            for eid, c, cn, ca, rn, ra, rc in zip(
                eids, countries, cnames, caddrs, raw_names, raw_addrs, raw_ctry
            ):
                def _add(idx):
                    if len(candidates_dict[idx]) < 40:
                        candidates_dict[idx][eid] = (rn, ra, rc)

                if cn and (c, cn) in index_N:
                    for idx in index_N[(c, cn)]:
                        _add(idx)

                if ca and (c, ca) in index_A:
                    for idx in index_A[(c, ca)]:
                        _add(idx)

                nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
                if nums and cn:
                    k = (c, f"{nums[0]}_{cn.split()[0]}")
                    if k in index_NA:
                        for idx in index_NA[k]:
                            if fuzz.ratio(test_cnames[idx], cn) >= 70:
                                _add(idx)

                if cn:
                    words = cn.split()
                    if len(words) >= 2:
                        k = (c, " ".join(words[:2]))
                        if k in index_N2:
                            for idx in index_N2[k]:
                                s1_ca = test_caddrs[idx]
                                if (fuzz.ratio(test_cnames[idx], cn) >= 85 or
                                        (ca and s1_ca and fuzz.ratio(s1_ca, ca) >= 75)):
                                    _add(idx)

        print(f"[+] Finished {label}: {total_rows:,} records in {time.time()-t_start:.1f}s", flush=True)

    process_test_file(s2_path, "Test Source 2")
    process_test_file(s3_path, "Test Source 3")

    del index_N, index_A, index_NA, index_N2, test_cnames, test_caddrs
    gc.collect()

    # ==========================================================================
    # STEP 3: Re-Score with Calibrated Model & Write Submissions
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f"[Step 3/3] Scoring Candidates & Generating {args.output_name}...")
    print("=" * 70, flush=True)

    total_cand_pairs = sum(len(v) for v in candidates_dict.values())
    print(f"[*] Total Candidate Pairs to Score: {total_cand_pairs:,}")
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("Amazon-ML-Submission", exist_ok=True)

    match_paths = [
        os.path.join(args.output_dir, args.output_name),
        os.path.join("Amazon-ML-Submission", args.output_name),
    ]

    cand_paths = []
    fc_handles = []
    if args.save_candidates:
        cand_paths = [
            os.path.join(args.output_dir, cand_name),
            os.path.join("Amazon-ML-Submission", cand_name),
        ]
        fc_handles = [open(p, "w", encoding="utf-8") for p in cand_paths]
        for h in fc_handles:
            h.write("source1_entity_id\tcandidate_entity_ids\n")
        print("[*] Candidate pairs file generation enabled.")
    else:
        print("[*] Candidate pairs file generation skipped (enable via --save-candidates when preparing final submission).")

    t_score_start = time.time()
    n_with_match = 0

    fm_handles = [open(p, "w", encoding="utf-8") for p in match_paths]
    for h in fm_handles:
        h.write("source1_entity_id\tmatched_entity_ids\n")

    for idx, eid in enumerate(test_eids):
        cand_map = candidates_dict.get(idx, {})

        if not cand_map:
            line_m = f"{eid}\t\n"
            for h in fm_handles: h.write(line_m)
            if fc_handles:
                line_c = f"{eid}\t\n"
                for h in fc_handles: h.write(line_c)
            continue

        cand_ids = list(cand_map.keys())
        feats = [
            compute_inference_features(
                test_raw_names[idx], test_raw_addrs[idx], test_raw_ctry[idx],
                cand_map[cid][0], cand_map[cid][1], cand_map[cid][2]
            )
            for cid in cand_ids
        ]

        feat_tensor = torch.tensor(feats, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model(feat_tensor)).cpu().numpy()

        accepted = sorted([cid for cid, p in zip(cand_ids, probs) if p >= best_thresh])[:11]
        
        if fc_handles:
            line_c = f"{eid}\t{','.join(sorted(cand_ids))}\n"
            for h in fc_handles: h.write(line_c)

        if accepted:
            n_with_match += 1
            line_m = f"{eid}\t{','.join(accepted)}\n"
        else:
            line_m = f"{eid}\t\n"
        for h in fm_handles: h.write(line_m)

        if (idx + 1) % 250_000 == 0:
            print(f"     [Progress] Evaluated {idx+1:,}/{n_s1:,} entities (Matches so far: {n_with_match:,})", flush=True)

    for h in fm_handles: h.close()
    for h in fc_handles: h.close()

    print(f"\n[+] Scoring finished in {time.time()-t_score_start:.1f}s!")
    print(f"[+] Total entities with matches: {n_with_match:,} / {n_s1:,} ({n_with_match/n_s1*100:.1f}%)")

    print("\n" + "=" * 70)
    print("SUBMISSION ARTIFACTS READY:")
    for p in match_paths:
        print(f"  [✔] Matches File    : {p} ({os.path.getsize(p)/(1024*1024):.1f} MB)")
    if cand_paths:
        for p in cand_paths:
            print(f"  [✔] Candidates File : {p} ({os.path.getsize(p)/(1024*1024):.1f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
