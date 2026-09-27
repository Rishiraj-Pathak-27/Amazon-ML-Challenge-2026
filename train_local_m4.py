#!/usr/bin/env python3
"""
train_local_m4.py — High-Performance Local Training & Fine-Tuning on Apple Silicon M4

Features:
  - Hardware Acceleration: Native Apple Metal Performance Shaders (MPS / Neural Engine)
  - Pre-trained Warm-Start: Transfers weights from Amazon-ML-Submission/model_tpu.pt
  - Clean Dataset: Uses cleaned_dataset/training_data.csv without nulls
  - Calibration: Tunes prediction threshold on validation set for optimal F0.5 score
  - Optional Test Inference: Generates competition submission matching_results.tsv
"""

import os
import sys
import time
import json
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
# 1. Feature Definition (11 features matching cleaned_dataset/training_data.csv)
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
    import re
    return re.sub(r"[^\w\s]", " ", str(s).lower()).strip()

def _clean_addr(s):
    if not s or not isinstance(s, str):
        return ""
    import re
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(_ADDR_ABBR.get(w, w) for w in s.split())

# ------------------------------------------------------------------------------
# 2. Model Architecture
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
# 3. Metrics & F0.5 Calibration
# ------------------------------------------------------------------------------
def compute_f_beta(precision, recall, beta=0.5):
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * (precision * recall) / (b2 * precision + recall)

def calibrate_threshold(y_true, y_probs, beta=0.5):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh, best_f = 0.5, 0.0
    best_p, best_r = 0.0, 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_score = compute_f_beta(prec, rec, beta=beta)

        if f_score > best_f:
            best_f = f_score
            best_thresh = float(th)
            best_p, best_r = prec, rec

    return best_thresh, best_f, best_p, best_r

# ------------------------------------------------------------------------------
# 4. Main Training Routine
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local M4 Training & Fine-Tuning for Entity Resolution")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size (default: 2048)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4 for fine-tuning)")
    parser.add_argument("--checkpoint", type=str, default="Amazon-ML-Submission/model_tpu.pt", help="Pre-trained checkpoint path")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch without loading checkpoint")
    parser.add_argument("--dataset", type=str, default="cleaned_dataset/training_data.csv", help="Clean dataset path")
    parser.add_argument("--run-inference", action="store_true", help="Run test inference after training")
    args = parser.parse_args()

    # Hardware Detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[+] Hardware Accelerator: Apple Silicon M4 GPU / Neural Engine (MPS)", flush=True)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[+] Hardware Accelerator: NVIDIA GPU ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        device = torch.device("cpu")
        print("[*] Running on CPU.", flush=True)

    # 1. Load Clean Dataset
    print(f"\n[*] Loading cleaned dataset from: {args.dataset}")
    t0 = time.time()
    df_pl = pl.read_csv(args.dataset)
    print(f"[+] Loaded {len(df_pl):,} pairs in {time.time()-t0:.2f}s")

    labels = df_pl["label"].to_numpy()
    groups = df_pl["source1_entity_id"].to_numpy()
    X = df_pl.select(FEATURE_COLUMNS).to_numpy().astype(np.float32)

    # Check for NaN / Infs
    if np.isnan(X).any():
        print("[!] Warning: NaNs detected in features, replacing with 0.0")
        X = np.nan_to_num(X, nan=0.0)

    # Split: GroupShuffleSplit to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(X, labels, groups=groups))

    X_train, y_train = X[train_idx], labels[train_idx].astype(np.float32)
    X_val, y_val     = X[val_idx],   labels[val_idx].astype(np.float32)

    print(f"[*] Train set: {len(X_train):,} samples | Val set: {len(X_val):,} samples")
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    print(f"[*] Class balance — Positives: {n_pos:,}, Negatives: {n_neg:,} (pos_weight={pos_weight.item():.2f})")

    # 2. Initialize Model & Transfer Weights
    model = DeepEntityMatcher(in_features=N_FEATURES, hidden_dim=128, num_blocks=3, dropout=0.15).to(device)

    is_finetuning = False
    if not args.from_scratch and os.path.exists(args.checkpoint):
        print(f"\n[*] Loading pre-trained checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model_dict = model.state_dict()
        transferred = 0

        for k, v in ckpt.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    transferred += 1
                elif k == "input_layer.0.weight" and v.shape[0] == model_dict[k].shape[0]:
                    # Adapt 15-dim to 11-dim
                    model_dict[k][:, :N_FEATURES] = v[:, :N_FEATURES]
                    transferred += 1

        model.load_state_dict(model_dict)
        print(f"[+] Successfully transferred {transferred}/{len(model_dict)} tensor blocks into M4 model!")
        is_finetuning = True
    else:
        print("\n[*] Training model from scratch.")

    # 3. Setup Optimizer & DataLoader
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. Training Loop
    print(f"\n{'='*65}")
    mode_str = "Fine-Tuning" if is_finetuning else "Training from scratch"
    print(f"[*] Starting {mode_str} on {device.type.upper()} ({args.epochs} epochs)...")
    print(f"{'='*65}")

    t_train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        t_ep = time.time()

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

        if device.type == "mps":
            torch.mps.synchronize()

        scheduler.step()
        avg_loss = ep_loss / max(1, n_batches)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch [{epoch}/{args.epochs}] — Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {time.time()-t_ep:.2f}s")

    print(f"\n[+] Training finished in {time.time() - t_train_start:.2f}s!")

    # 5. Validation Evaluation & Threshold Calibration
    print("\n[*] Evaluating on Validation Set & Calibrating Optimal Threshold...")
    model.eval()
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=args.batch_size, shuffle=False)
    val_preds = []

    with torch.no_grad():
        for (bx,) in val_loader:
            bx = bx.to(device)
            probs = torch.sigmoid(model(bx))
            val_preds.append(probs.cpu().numpy())

    val_probs = np.concatenate(val_preds)
    best_th, best_f05, prec, rec = calibrate_threshold(y_val, val_probs, beta=0.5)

    print(f"\n{'='*65}")
    print(f"[+] Calibration Results (Amazon ML Metric — F0.5):")
    print(f"    Optimal Threshold : {best_th:.2f}")
    print(f"    Validation F0.5   : {best_f05:.4f}")
    print(f"    Precision         : {prec:.4f}")
    print(f"    Recall            : {rec:.4f}")
    print(f"{'='*65}")

    # 6. Save Model & Config
    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("Amazon-ML-Submission/model", exist_ok=True)

    save_targets = [
        "model/model_m4.pt",
        "Amazon-ML-Submission/model/model_m4.pt",
        "Amazon-ML-Submission/model_tpu.pt"  # Update default checkpoint
    ]
    for target in save_targets:
        torch.save(model.state_dict(), target)
        print(f"[+] Saved model checkpoint: {target}")

    config_data = {
        "threshold": best_th,
        "val_f05": best_f05,
        "val_precision": prec,
        "val_recall": rec,
        "in_features": N_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "epochs": args.epochs,
        "device": device.type,
    }
    with open("model/config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    with open("Amazon-ML-Submission/config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    print("[+] Saved calibrated config to config.json")

    # 7. Optional Test Inference
    if args.run_inference:
        test_dir = "dataset/test"
        if os.path.exists(os.path.join(test_dir, "test_source1.tsv")):
            print("\n[*] Running test inference on local dataset/test ...")
            run_test_inference(model, device, best_th)
        else:
            print("[!] dataset/test/test_source1.tsv not found — skipping inference.")

    print("\n[✔] All operations completed successfully!")

# ------------------------------------------------------------------------------
# 5. Batched Test Inference (Optional)
# ------------------------------------------------------------------------------
def run_test_inference(model, device, best_thresh):
    import re
    s1_path = "dataset/test/test_source1.tsv"
    s2_path = "dataset/test/test_source2.tsv"
    s3_path = "dataset/test/test_source3.tsv"

    print("[*] Reading test source 1...", flush=True)
    s1_df = pl.read_csv(s1_path, separator="\t")
    n_s1 = len(s1_df)
    test_eids = s1_df["entity_id"].to_list()
    test_countries = [normalize_country(c) for c in s1_df["country"].to_list()]
    test_cnames = [_clean_name(x) for x in s1_df["business_name"].to_list()]
    test_caddrs = [_clean_addr(x) for x in s1_df["business_address"].to_list()]
    test_raw_names = s1_df["business_name"].to_list()
    test_raw_addrs = s1_df["business_address"].to_list()
    test_raw_ctry  = s1_df["country"].to_list()
    del s1_df

    index_N = defaultdict(list)
    index_A = defaultdict(list)
    for idx in range(n_s1):
        c, cn, ca = test_countries[idx], test_cnames[idx], test_caddrs[idx]
        if cn:
            index_N[(c, cn)].append(idx)
        if ca:
            index_A[(c, ca)].append(idx)

    candidates_dict = defaultdict(dict)

    def process_test_file(path, label):
        print(f"[*] Processing {label}...", flush=True)
        reader = pl.read_csv_batched(path, separator="\t", batch_size=500_000)
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            b = batches[0]
            for eid, c, cn, ca, rn, ra, rc in zip(
                b["entity_id"].to_list(),
                [normalize_country(x) for x in b["country"].to_list()],
                [_clean_name(x) for x in b["business_name"].to_list()],
                [_clean_addr(x) for x in b["business_address"].to_list()],
                b["business_name"].to_list(),
                b["business_address"].to_list(),
                b["country"].to_list()
            ):
                if cn and (c, cn) in index_N:
                    for idx in index_N[(c, cn)]:
                        if len(candidates_dict[idx]) < 40:
                            candidates_dict[idx][eid] = (rn, ra, rc)
                if ca and (c, ca) in index_A:
                    for idx in index_A[(c, ca)]:
                        if len(candidates_dict[idx]) < 40:
                            candidates_dict[idx][eid] = (rn, ra, rc)

    process_test_file(s2_path, "Source 2")
    process_test_file(s3_path, "Source 3")

    print("[*] Re-scoring candidate pairs with calibrated model...", flush=True)
    match_path = "Amazon-ML-Submission/matching_results.tsv"
    cand_path  = "Amazon-ML-Submission/candidate_pairs.tsv"

    def compute_inf_feat(s1_n, s1_a, s1_c, c_n, c_a, c_c):
        s1_nm, c_nm = _clean_name(s1_n), _clean_name(c_n)
        s1_ad, c_ad = _clean_addr(s1_a), _clean_addr(c_a)
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

    with open(match_path, "w", encoding="utf-8") as fm, open(cand_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        for idx, eid in enumerate(test_eids):
            c_map = candidates_dict.get(idx, {})
            if not c_map:
                fm.write(f"{eid}\t\n")
                fc.write(f"{eid}\t\n")
                continue

            c_ids = list(c_map.keys())
            feats = [
                compute_inf_feat(
                    test_raw_names[idx], test_raw_addrs[idx], test_raw_ctry[idx],
                    c_map[cid][0], c_map[cid][1], c_map[cid][2]
                )
                for cid in c_ids
            ]
            feat_t = torch.tensor(feats, dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = torch.sigmoid(model(feat_t)).cpu().numpy()

            accepted = [cid for cid, p in zip(c_ids, probs) if p >= best_thresh][:11]
            fc.write(f"{eid}\t{','.join(sorted(c_ids))}\n")
            fm.write(f"{eid}\t{','.join(accepted)}\n")

    print(f"[+] Output submission written to: {match_path}")

if __name__ == "__main__":
    main()
