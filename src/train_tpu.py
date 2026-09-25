"""
Train the Deep Neural Entity Matcher on TPU (PyTorch-XLA) or GPU/CPU.

Pipeline:
  1. Load train_source1/2/3.tsv + train_ground_truth.tsv
  2. Split Source-1 entities into a train / validation split (entity-level)
  3. Run multi-key blocking and extract pairwise features
  4. Train DeepEntityMatcher with weighted BCE loss on TPU / GPU
  5. Sweep decision threshold on validation set to maximize macro F_0.5
  6. Save model_tpu.pt + config_tpu.json

Usage:
    python -m src.train_tpu --train-dir dataset/train --model-dir model --epochs 10 --batch-size 2048
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .blocking import build_all_candidates
from .evaluate import macro_f_beta
from .features import build_feature_frame, FEATURE_COLUMNS
from .model_tpu import DeepEntityMatcher, get_device

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def _load_train(train_dir):
    try:
        import polars as pl
        s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t").to_pandas().fillna("")
        s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t").to_pandas().fillna("")
        s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t").to_pandas().fillna("")
        gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t").to_pandas().fillna("")
    except Exception:
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
    preds = {eid: set() for eid in candidates.keys()}
    if len(feats_with_prob) == 0:
        return preds
    positive = feats_with_prob[feats_with_prob["prob"] >= threshold]
    for s1_id, cand_id in zip(positive["source1_entity_id"], positive["candidate_entity_id"]):
        preds.setdefault(s1_id, set()).add(cand_id)
    return preds


def train_model(
    train_dir="dataset/train",
    model_dir="model",
    top_k=25,
    val_fraction=0.15,
    max_train_entities=100000,
    epochs=10,
    batch_size=2048,
    lr=1e-3,
    seed=42,
):
    os.makedirs(model_dir, exist_ok=True)
    device, device_name = get_device()
    print(f"[*] Training on Device: {device_name.upper()} ({device})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[*] Loading training data from {train_dir}...")
    t0 = time.time()
    s1, s2, s3, gt = _load_train(train_dir)
    gt_map = _ground_truth_map(gt)
    print(f"[*] Loaded S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT: {len(gt)} in {time.time()-t0:.2f}s")

    if max_train_entities and len(s1) > max_train_entities:
        s1 = s1.sample(n=max_train_entities, random_state=seed).reset_index(drop=True)
        print(f"[*] Subsampled to {len(s1)} Source 1 entities for high-speed training.")

    rng = np.random.RandomState(seed)
    s1_ids = s1["entity_id"].to_numpy().copy()
    rng.shuffle(s1_ids)
    n_val = max(1, int(len(s1_ids) * val_fraction))
    val_ids = set(s1_ids[:n_val])
    train_ids = set(s1_ids[n_val:])

    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"[*] Splitting: {len(s1_train)} Train entities, {len(s1_val)} Val entities")

    print("[*] Generating candidate pairs in a single unified blocking pass...")
    t_feat = time.time()
    all_feats, all_candidates = _build_labeled_features(s1, s2, s3, gt_map, top_k)
    print(f"[*] Total pairs generated: {len(all_feats)} in {time.time()-t_feat:.2f}s")

    train_feats = all_feats[all_feats["source1_entity_id"].isin(train_ids)].reset_index(drop=True)
    val_feats = all_feats[all_feats["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    val_candidates = {eid: all_candidates.get(eid, set()) for eid in val_ids}

    print(f"[*] Train pairs: {len(train_feats)} (positives: {int(train_feats['label'].sum()) if len(train_feats) else 0})")
    print(f"[*] Val pairs: {len(val_feats)} (positives: {int(val_feats['label'].sum()) if len(val_feats) else 0})")

    X_train_np = train_feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_np = train_feats["label"].to_numpy(dtype=np.float32)

    X_val_np = val_feats[FEATURE_COLUMNS].to_numpy(dtype=np.float32) if len(val_feats) > 0 else np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32)
    y_val_np = val_feats["label"].to_numpy(dtype=np.float32) if len(val_feats) > 0 else np.zeros(0, dtype=np.float32)

    n_pos = max(1, int(y_train_np.sum()))
    n_neg = max(1, len(y_train_np) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    print(f"[*] Class imbalance: {n_pos} pos, {n_neg} neg. Loss pos_weight = {pos_weight.item():.2f}")

    train_ds = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(y_train_np))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DeepEntityMatcher(in_features=len(FEATURE_COLUMNS), hidden_dim=128, num_blocks=3, dropout=0.15)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[*] Commencing Training on {device_name.upper()} ({epochs} epochs)...", flush=True)
    model.train()
    total_batches = len(train_loader)
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        t_ep = time.time()
        for b_idx, (batch_x, batch_y) in enumerate(train_loader, 1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            if device_name == "tpu" and HAS_XLA:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()

            ep_loss += loss.item()
            n_batches += 1

            if b_idx % 20 == 0 or b_idx == total_batches:
                print(f"     Epoch {ep:02d}/{epochs:02d} | Step {b_idx:03d}/{total_batches:03d} | Current Batch Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = ep_loss / max(1, n_batches)
        print(f"[*] Epoch {ep:02d}/{epochs:02d} Complete | Avg Loss: {avg_loss:.4f} | Time: {time.time()-t_ep:.2f}s", flush=True)

    # Validation & Threshold Sweep
    print("\n[*] Evaluating on Validation Set and Tuning Macro F0.5 Threshold...", flush=True)
    model.eval()
    val_probs = []
    if len(X_val_np) > 0:
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_np)), batch_size=batch_size * 2, shuffle=False)
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs.tolist())

    val_feats_eval = val_feats.copy()
    val_feats_eval["prob"] = val_probs

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.10, 0.95, 0.05):
        preds = _threshold_to_predictions(val_feats_eval, threshold, val_candidates)
        score = macro_f_beta(preds, gt_map, beta=0.5)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    print(f"\n[+] Optimal Decision Threshold: {best_threshold:.2f}", flush=True)
    print(f"[+] Best Validation Macro F0.5: {best_score:.4f}", flush=True)

    # Save artifacts
    model_path = os.path.join(model_dir, "model_tpu.pt")
    config_path = os.path.join(model_dir, "config_tpu.json")

    torch.save(model.state_dict(), model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "top_k": top_k,
                "val_f0.5": best_score,
                "in_features": len(FEATURE_COLUMNS),
                "device": device_name,
                "feature_columns": FEATURE_COLUMNS,
            },
            f,
            indent=2,
        )

    print(f"[+] Saved model weights to {model_path}", flush=True)
    print(f"[+] Saved config to {config_path}", flush=True)
    return model, best_threshold


def main():
    parser = argparse.ArgumentParser(description="Train Deep Neural Entity Matcher on TPU/GPU")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-entities", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        model_dir=args.model_dir,
        top_k=args.top_k,
        val_fraction=args.val_fraction,
        max_train_entities=args.max_train_entities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
