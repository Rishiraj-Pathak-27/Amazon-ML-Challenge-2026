# ==============================================================================
# Amazon ML Challenge 2026 — Fine-Tune Entity Matcher with Clean Dataset
# + Automatic Sync to Google Drive & GitHub via Colab CLI
#
# ── Colab CLI Quick-Start ──────────────────────────────────────────────────────
#   pip install google-colab-cli
#
#   # One-shot: provision TPU, run, auto-teardown (--timeout is required, default is only 30s!)
#   colab run --tpu v5e1 colab_all_in_one.py --timeout 3600
#
#   # Interactive workflow with session (Recommended)
#   colab new -s trainer --tpu v5e1
#   colab drivemount -s trainer
#   colab upload -s trainer cleaned_dataset/training_data.csv /content/training_data.csv
#   colab exec  -s trainer -f colab_all_in_one.py --timeout 3600
#   colab download -s trainer /content/model/model_tpu.pt      ./model/model_tpu.pt
#   colab download -s trainer /content/model/config.json       ./model/config.json
#   colab download -s trainer /content/output/matching_results.tsv ./output/matching_results.tsv
#   colab stop  -s trainer
#
# Tip: If running inside a Colab notebook cell, paste this file and click Run.
#      Google Drive will be mounted automatically.
# ==============================================================================

import os
import sys
import subprocess
import time
import re
import gc
import json
from collections import defaultdict

# ------------------------------------------------------------------------------
# Configuration & Auto-Sync Credentials
# ------------------------------------------------------------------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
try:
    from google.colab import userdata
    if not GITHUB_TOKEN:
        GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
except Exception:
    pass

for i, arg in enumerate(sys.argv):
    if arg in ("--token", "--github-token") and i + 1 < len(sys.argv):
        GITHUB_TOKEN = sys.argv[i + 1]

REPO_OWNER_REPO = os.environ.get("GITHUB_REPO", "utkarsh232005/Amazon-ML-Challenge-2026")
for i, arg in enumerate(sys.argv):
    if arg in ("--repo", "--github-repo") and i + 1 < len(sys.argv):
        REPO_OWNER_REPO = sys.argv[i + 1]

DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Amazon-ML-Submission"

# Priority order for locating training_data.csv on the Colab VM:
#   1. Uploaded directly via `colab upload` → /content/training_data.csv
#   2. From Google Drive                    → DRIVE_BACKUP_DIR/training_data.csv
#   3. Bundled in the repo                  → cleaned_dataset/training_data.csv
CLEAN_DATASET_CANDIDATES = [
    "/content/training_data.csv",
    os.path.join(DRIVE_BACKUP_DIR, "training_data.csv"),
    "cleaned_dataset/training_data.csv",
]

# ── Hyper-parameters ───────────────────────────────────────────────────────────
FINETUNE_EPOCHS = 5     # epochs when loading an existing model
FINETUNE_LR     = 2e-4
TRAIN_EPOCHS    = 10    # epochs when training from scratch
TRAIN_LR        = 1e-3
BATCH_SIZE      = 2048
HIDDEN_DIM      = 128
NUM_BLOCKS      = 3
DROPOUT         = 0.15
VAL_FRACTION    = 0.15

# Feature columns produced by the cleaning notebook (11 features)
FEATURE_COLUMNS = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio",
    "core_name_ratio", "address_ratio", "address_token_set_ratio",
    "address_number_match", "country_match",
    "name_exact_match", "core_name_exact_match", "address_exact_match",
]
N_FEATURES = len(FEATURE_COLUMNS)  # 11


# ------------------------------------------------------------------------------
def sync_checkpoint(step_title, files_to_backup, commit_msg):
    """Syncs artifacts to Google Drive and commits/pushes to GitHub."""
    print(f"\n[Sync] >>> Checkpoint: {step_title} <<<", flush=True)

    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
        for src_path in files_to_backup:
            if os.path.exists(src_path):
                dest = os.path.join(DRIVE_BACKUP_DIR, os.path.basename(src_path))
                if os.path.isdir(src_path):
                    os.system(f"cp -rf '{src_path}' '{DRIVE_BACKUP_DIR}/'")
                else:
                    os.system(f"cp -f '{src_path}' '{dest}'")
                print(f"  [+] Saved to Google Drive: {dest}", flush=True)
    else:
        print("  [*] Google Drive not mounted; skipping Drive backup.", flush=True)

    if GITHUB_TOKEN:
        try:
            os.system('git config user.name "Colab Auto-Sync"')
            os.system('git config user.email "colab@google.com"')
            remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_OWNER_REPO}.git"
            os.system(f"git remote set-url origin '{remote_url}'")
            os.system("git pull --rebase origin main")

            for src_path in files_to_backup:
                if os.path.exists(src_path):
                    if os.path.isfile(src_path) and os.path.getsize(src_path) > 95 * 1024 * 1024:
                        gz_path = f"{src_path}.gz"
                        if not os.path.exists(gz_path):
                            print(f"  [*] Compressing {src_path} for GitHub (>95MB)...", flush=True)
                            os.system(f"gzip -c '{src_path}' > '{gz_path}'")
                        os.system(f"git add '{gz_path}'")
                    else:
                        os.system(f"git add '{src_path}'")

            os.system(f'git commit -m "{commit_msg}"')
            ret = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
            if ret.returncode == 0:
                print(f"  [+] Pushed to GitHub: {commit_msg}", flush=True)
            else:
                err_msg = ret.stderr.strip() or ret.stdout.strip()
                print(f"  [!] GitHub push note: {err_msg}", flush=True)
                print("      (Tip: Ensure token has 'repo' scope.)", flush=True)
        except Exception as e:
            print(f"  [!] GitHub sync exception: {e}", flush=True)
    else:
        print("  [*] Tip: Set GITHUB_TOKEN to enable automatic git push.", flush=True)


# ==============================================================================
# STEP 1 — Install dependencies
# ==============================================================================
print("=" * 70)
print("[Step 1/7] Installing High-Performance Libraries...")
print("=" * 70, flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "polars", "rapidfuzz", "scikit-learn", "lightgbm", "pandas", "numpy"],
    check=True
)

import polars as pl
from rapidfuzz import fuzz
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit


# ==============================================================================
# STEP 2 — Hardware detection (TPU / CUDA / CPU)
# ==============================================================================
print("\n" + "=" * 70)
print("[Step 2/7] Initializing Hardware Accelerator...")
print("=" * 70, flush=True)

HAS_TPU = False
device_name = "cpu"
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    device_name = "tpu"
    HAS_TPU = True
    print(f"[+] TPU Accelerator Active! Device: {device}", flush=True)
except Exception:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "cuda"
        print(f"[+] NVIDIA CUDA GPU Active! Device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("[!] Running on standard CPU.", flush=True)


# ==============================================================================
# STEP 3 — Mount Drive & resolve dataset paths
# ==============================================================================
print("\n" + "=" * 70)
print("[Step 3/7] Mounting Google Drive & Locating Datasets...")
print("=" * 70, flush=True)

# Only attempt interactive drive mount if not in a headless/CLI run or if explicitly enabled
if not os.path.exists("/content/drive/MyDrive"):
    # In headless CLI execution (colab run / colab exec), drive.mount hangs waiting for browser auth.
    # To mount Drive in CLI, use `colab drivemount -s <session>` instead.
    is_interactive_notebook = "IPython" in sys.modules and hasattr(sys, "ps1")
    if os.environ.get("MOUNT_DRIVE") == "1" or is_interactive_notebook:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("[+] Google Drive mounted at /content/drive", flush=True)
        except Exception as e:
            print(f"[*] Drive mount note: {e}", flush=True)
    else:
        print("[*] Headless/CLI mode detected. Skipping interactive drive.mount to avoid hanging.", flush=True)
        print("    (If using Colab CLI, run `colab drivemount -s <session>` before executing.)", flush=True)
else:
    print("[+] Google Drive is already mounted at /content/drive", flush=True)

os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ── Locate training_data.csv ──────────────────────────────────────────────────
clean_dataset_path = None
for cand in CLEAN_DATASET_CANDIDATES:
    if os.path.exists(cand):
        clean_dataset_path = cand
        print(f"[+] Found cleaned training dataset: {cand}", flush=True)
        break

if clean_dataset_path is None:
    raise FileNotFoundError(
        "Cannot find training_data.csv!\n"
        "Upload via Colab CLI:  colab upload -s trainer "
        "cleaned_dataset/training_data.csv /content/training_data.csv\n"
        "Or place it in Google Drive at: Amazon-ML-Submission/training_data.csv"
    )

# ── Locate raw test TSVs ─────────────────────────────────────────────────────
def check_dataset_dir(d):
    return (
        os.path.exists(os.path.join(d, "train", "train_source1.tsv")) and
        os.path.exists(os.path.join(d, "test",  "test_source1.tsv"))
    )

dataset_found = False
if check_dataset_dir("dataset"):
    dataset_found = True
    print("[+] Raw dataset verified in local dataset/ folder!", flush=True)
elif check_dataset_dir("student_resource/dataset"):
    os.system("ln -sfn student_resource/dataset dataset")
    dataset_found = True
    print("[+] Linked dataset from student_resource/dataset!", flush=True)
else:
    tar_candidates = [
        "/content/drive/MyDrive/student_resource.tar.gz",
        "/content/student_resource.tar.gz",
        f"/content/drive/MyDrive/{REPO_OWNER_REPO.split('/')[-1]}/student_resource.tar.gz",
    ]
    for tar_path in tar_candidates:
        if os.path.exists(tar_path):
            print(f"[*] Found {tar_path}! Extracting...", flush=True)
            subprocess.run(["tar", "-xzf", tar_path, "-C", "."], check=True)
            if check_dataset_dir("student_resource/dataset"):
                os.system("ln -sfn student_resource/dataset dataset")
                dataset_found = True
                print("[+] Extracted and linked dataset successfully!", flush=True)
                break

if not dataset_found:
    raise FileNotFoundError(
        "Raw test dataset not found! "
        "Please ensure student_resource.tar.gz is in Google Drive root."
    )


# ==============================================================================
# STEP 4 — Deep Residual Entity Matcher (11-feature version)
# ==============================================================================
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
        return x + self.drop2(
            self.act2(self.norm2(self.fc2(
                self.drop1(self.act1(self.norm1(self.fc1(x))))
            )))
        )


class DeepEntityMatcher(nn.Module):
    def __init__(self, in_features=N_FEATURES, hidden_dim=HIDDEN_DIM,
                 num_blocks=NUM_BLOCKS, dropout=DROPOUT):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# ==============================================================================
# STEP 5 — Load Cleaned Dataset → Group-Split → Fine-tune or Train from Scratch
# ==============================================================================
print("\n" + "=" * 70)
print("[Step 4/7] Loading Cleaned Training Dataset & Preparing Model...")
print("=" * 70, flush=True)

t_load = time.time()
df = pd.read_csv(clean_dataset_path, dtype={c: np.float32 for c in FEATURE_COLUMNS})
print(f"[+] Loaded {len(df):,} training pairs in {time.time()-t_load:.1f}s", flush=True)
print(f"    Class balance — Match: {int((df['label']==1).sum()):,} "
      f"| Non-match: {int((df['label']==0).sum()):,}", flush=True)

# Group-based split — same source1_entity_id always stays in the same split,
# preventing any data leakage between train and val.
groups = df["source1_entity_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=groups))

X_train = df.iloc[train_idx][FEATURE_COLUMNS].values.astype(np.float32)
y_train = df.iloc[train_idx]["label"].values.astype(np.float32)
X_val   = df.iloc[val_idx][FEATURE_COLUMNS].values.astype(np.float32)
y_val   = df.iloc[val_idx]["label"].values.astype(np.float32)

val_s1_ids   = df.iloc[val_idx]["source1_entity_id"].values
val_cand_ids = df.iloc[val_idx]["matched_entity_id"].values

# Ground-truth map for val threshold calibration
gt_map_val = defaultdict(set)
for s1_id, cand_id, lbl in zip(val_s1_ids, val_cand_ids, y_val):
    if lbl == 1.0:
        gt_map_val[s1_id].add(cand_id)

print(f"[*] Train pairs: {len(X_train):,} | Val pairs: {len(X_val):,}", flush=True)
del df
gc.collect()

n_pos = max(1, int(y_train.sum()))
n_neg = max(1, len(y_train) - n_pos)
pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
print(f"[*] Pos: {n_pos:,}, Neg: {n_neg:,}, pos_weight: {pos_weight.item():.2f}", flush=True)

# ── Check Drive / local model/ for existing weights ───────────────────────────
MODEL_WEIGHTS_CANDIDATES = [
    "model/model_tpu.pt",
    os.path.join(DRIVE_BACKUP_DIR, "model_tpu.pt"),
]

model = DeepEntityMatcher(
    in_features=N_FEATURES, hidden_dim=HIDDEN_DIM,
    num_blocks=NUM_BLOCKS, dropout=DROPOUT
).to(device)

existing_config = {}
loaded_existing = False

for w_path in MODEL_WEIGHTS_CANDIDATES:
    if os.path.exists(w_path):
        try:
            state = torch.load(w_path, map_location=device)
            model.load_state_dict(state)
            print(f"[+] Loaded existing model weights from: {w_path}", flush=True)
            loaded_existing = True
            # Try to recover previous threshold config
            cfg_path = os.path.join(os.path.dirname(w_path), "config.json")
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(DRIVE_BACKUP_DIR, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    existing_config = json.load(f)
                print(f"[+] Previous config: {existing_config}", flush=True)
        except Exception as e:
            print(f"[!] Could not load weights from {w_path}: {e}  "
                  f"— will train from scratch.", flush=True)
        break

if loaded_existing:
    epochs, lr = FINETUNE_EPOCHS, FINETUNE_LR
    print(f"[*] Fine-tuning existing model: {epochs} epochs, lr={lr}", flush=True)
else:
    epochs, lr = TRAIN_EPOCHS, TRAIN_LR
    print(f"[*] No existing model found — training from scratch: "
          f"{epochs} epochs, lr={lr}", flush=True)

# ── Training loop ─────────────────────────────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

model.train()
for ep in range(1, epochs + 1):
    ep_loss, n_batches = 0.0, 0
    t_ep = time.time()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        if HAS_TPU:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
        ep_loss   += loss.item()
        n_batches += 1
    scheduler.step()
    print(
        f"[*] Epoch {ep:02d}/{epochs} | "
        f"Avg Loss: {ep_loss/max(1, n_batches):.4f} | "
        f"Time: {time.time()-t_ep:.2f}s", flush=True
    )

del X_train, y_train, train_loader
gc.collect()


# ==============================================================================
# STEP 5b — Threshold Calibration (Macro F0.5)
# ==============================================================================
print("\n" + "=" * 70)
print("[Step 5/7] Calibrating Decision Threshold (Macro F0.5)...")
print("=" * 70, flush=True)


def macro_f05(preds, ground_truth):
    f_scores = []
    beta_sq  = 0.25
    for s1_id, gt_set in ground_truth.items():
        p_set = preds.get(s1_id, set())
        if not gt_set and not p_set:
            f_scores.append(1.0)
            continue
        if not gt_set or not p_set:
            f_scores.append(0.0)
            continue
        tp   = len(gt_set & p_set)
        prec = tp / len(p_set)
        rec  = tp / len(gt_set)
        denom = beta_sq * prec + rec
        f_scores.append((1.25 * prec * rec / denom) if denom > 0 else 0.0)
    return float(np.mean(f_scores)) if f_scores else 0.0


model.eval()
val_probs  = []
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=4096, shuffle=False)
with torch.no_grad():
    for (bx,) in val_loader:
        probs = torch.sigmoid(model(bx.to(device))).cpu().numpy()
        val_probs.extend(probs.tolist())

# Start from previous best threshold if available; finer sweep steps (0.02)
best_thresh = existing_config.get("threshold", 0.90)
best_score  = -1.0

if len(val_probs) == len(val_s1_ids):
    for thresh in np.arange(0.20, 0.96, 0.02):
        current_preds = defaultdict(set)
        for s1_id, cand_id, p in zip(val_s1_ids, val_cand_ids, val_probs):
            if p >= thresh:
                current_preds[s1_id].add(cand_id)
        score = macro_f05(dict(current_preds), dict(gt_map_val))
        if score > best_score:
            best_score, best_thresh = score, float(thresh)

print(f"[+] Optimal F0.5 Threshold: {best_thresh:.2f} (Macro F0.5: {best_score:.4f})", flush=True)

del X_val, y_val, val_probs, val_s1_ids, val_cand_ids, gt_map_val
gc.collect()

# Save updated model & config
torch.save(model.state_dict(), "model/model_tpu.pt")
with open("model/config.json", "w") as f:
    json.dump({
        "threshold":       best_thresh,
        "val_f05":         best_score,
        "finetuned":       loaded_existing,
        "feature_columns": FEATURE_COLUMNS,
        "n_features":      N_FEATURES,
        "hidden_dim":      HIDDEN_DIM,
        "num_blocks":      NUM_BLOCKS,
    }, f, indent=2)

sync_checkpoint(
    step_title="Fine-tuned Model & Threshold Config",
    files_to_backup=["model/model_tpu.pt", "model/config.json"],
    commit_msg=(
        f"checkpoint: {'finetune' if loaded_existing else 'train'} | "
        f"threshold={best_thresh:.2f} | val_f05={best_score:.4f}"
    )
)


# ==============================================================================
# STEP 6 — Text Normalization (used only during test inference)
# ==============================================================================
LEGAL_SUFFIXES = {
    "corporation", "corp", "company", "co", "incorporated", "inc",
    "limited", "ltd", "private", "pvt", "limited liability company", "llc",
    "limited liability partnership", "llp", "sa", "sarl", "sas", "sasu",
    "eurl", "sci", "group", "grp", "services", "svcs", "holdings",
    "enterprises", "trading", "industries",
}

_ADDR_ABBR = {
    "rd": "road",  "st": "street",  "ave": "avenue",  "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "ln": "lane", "dr": "drive",
    "apt": "apartment", "fl": "floor", "flr": "floor",  "ste": "suite",
    "hwy": "highway",  "ct": "court",  "pl": "place",  "sq": "square",
    "pkwy": "parkway", "cir": "circle", "aly": "alley", "bldg": "building",
}


def normalize_country(c):
    return str(c).strip().upper() if c else ""


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
    return " ".join(_ADDR_ABBR.get(w, w) for w in s.split())


def _extract_keys(c, cn, ca):
    keys = []
    if cn:
        keys.append(("N", c, cn))
        words = cn.split()
        if len(words) >= 2:
            keys.append(("N2", c, " ".join(words[:2])))
    if ca:
        keys.append(("A", c, ca))
        nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
        if nums and cn:
            keys.append(("NA", c, f"{nums[0]}_{cn.split()[0]}"))
    return keys


def compute_inference_features(s1_name, s1_addr, s1_c, c_name, c_addr, c_c):
    """Return a 11-element feature vector matching FEATURE_COLUMNS."""
    s1_nm, c_nm = _clean_name(s1_name), _clean_name(c_name)
    s1_ad, c_ad = _clean_addr(s1_addr), _clean_addr(c_addr)
    s1_core = " ".join(w for w in s1_nm.split() if w not in LEGAL_SUFFIXES)
    c_core  = " ".join(w for w in c_nm.split()  if w not in LEGAL_SUFFIXES)
    s1_nums = " ".join(re.findall(r"\d+", s1_ad))
    c_nums  = " ".join(re.findall(r"\d+", c_ad))
    s1_ctry = normalize_country(s1_c)
    c_ctry  = normalize_country(c_c)
    return [
        fuzz.ratio(s1_nm, c_nm)            / 100.0,
        fuzz.token_set_ratio(s1_nm, c_nm)  / 100.0,
        fuzz.token_sort_ratio(s1_nm, c_nm) / 100.0,
        fuzz.ratio(s1_core, c_core)        / 100.0,
        fuzz.ratio(s1_ad, c_ad)            / 100.0,
        fuzz.token_set_ratio(s1_ad, c_ad)  / 100.0,
        int(bool(s1_nums) and s1_nums == c_nums),
        int(bool(s1_ctry) and s1_ctry == c_ctry),
        int(bool(s1_nm)   and s1_nm   == c_nm),
        int(bool(s1_core) and s1_core == c_core),
        int(bool(s1_ad)   and s1_ad   == c_ad),
    ]


# ==============================================================================
# STEP 7 — Memory-Safe Batched Test Inference
# ==============================================================================
print("\n" + "=" * 70)
print("[Step 6/7] Running Memory-Safe Test Set Inference...")
print("=" * 70, flush=True)

t_inf_start  = time.time()
s1_test_path = "dataset/test/test_source1.tsv"
s2_test_path = "dataset/test/test_source2.tsv"
s3_test_path = "dataset/test/test_source3.tsv"

s1_test_df      = pl.read_csv(s1_test_path, separator="\t")
n_test_s1       = len(s1_test_df)
test_eids       = s1_test_df["entity_id"].to_list()
test_countries  = [normalize_country(c) for c in s1_test_df["country"].to_list()]
test_cnames     = [_clean_name(x)  for x in s1_test_df["business_name"].to_list()]
test_caddrs     = [_clean_addr(x)  for x in s1_test_df["business_address"].to_list()]
test_raw_names  = s1_test_df["business_name"].to_list()
test_raw_addrs  = s1_test_df["business_address"].to_list()
test_raw_ctry   = s1_test_df["country"].to_list()
del s1_test_df
gc.collect()
print(f"[+] Loaded {n_test_s1:,} Test Source 1 entities.", flush=True)

# ── Build inverted indices ────────────────────────────────────────────────────
index_N  = defaultdict(list)
index_A  = defaultdict(list)
index_NA = defaultdict(list)
index_N2 = defaultdict(list)

for idx in range(n_test_s1):
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

# candidates_dict[idx] = {cand_id: (raw_name, raw_addr, raw_ctry)}
# Raw fields are kept so we can compute real similarity features at scoring time.
candidates_dict = defaultdict(dict)
matches_dict    = defaultdict(set)   # pre-model high-confidence matches


def process_test_source_batched(src_path, src_label, batch_size=500_000):
    t_src = time.time()
    print(f"[*] Streaming {src_label} in {batch_size//1000}k chunks...", flush=True)
    reader     = pl.read_csv_batched(src_path, separator="\t", batch_size=batch_size)
    total_rows = 0

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        b_df = batches[0]
        total_rows   += len(b_df)
        src_eids      = b_df["entity_id"].to_list()
        src_countries = [normalize_country(c) for c in b_df["country"].to_list()]
        src_cnames    = [_clean_name(x) for x in b_df["business_name"].to_list()]
        src_caddrs    = [_clean_addr(x) for x in b_df["business_address"].to_list()]
        src_raw_names = b_df["business_name"].to_list()
        src_raw_addrs = b_df["business_address"].to_list()
        src_raw_ctry  = b_df["country"].to_list()
        del b_df

        for eid, c, cn, ca, rn, ra, rc in zip(
            src_eids, src_countries, src_cnames, src_caddrs,
            src_raw_names, src_raw_addrs, src_raw_ctry
        ):
            def _add(idx):
                if len(candidates_dict[idx]) < 40:
                    candidates_dict[idx][eid] = (rn, ra, rc)

            if cn and (c, cn) in index_N:
                for idx in index_N[(c, cn)]:
                    matches_dict[idx].add(eid)
                    _add(idx)

            if ca and (c, ca) in index_A:
                for idx in index_A[(c, ca)]:
                    matches_dict[idx].add(eid)
                    _add(idx)

            nums = [w for w in ca.split() if any(ch.isdigit() for ch in w)]
            if nums and cn:
                k = (c, f"{nums[0]}_{cn.split()[0]}")
                if k in index_NA:
                    for idx in index_NA[k]:
                        if fuzz.ratio(test_cnames[idx], cn) >= 70:
                            matches_dict[idx].add(eid)
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
                                matches_dict[idx].add(eid)
                            _add(idx)

        del src_eids, src_countries, src_cnames, src_caddrs
        del src_raw_names, src_raw_addrs, src_raw_ctry
        print(
            f"     [Progress] {src_label}: {total_rows:,} rows "
            f"scanned in {time.time()-t_src:.1f}s", flush=True
        )

    print(f"[+] Finished {src_label} ({total_rows:,} rows) in {time.time()-t_src:.1f}s",
          flush=True)
    gc.collect()


process_test_source_batched(s2_test_path, "Test Source 2")
process_test_source_batched(s3_test_path, "Test Source 3")

del index_N, index_A, index_NA, index_N2, test_cnames, test_caddrs
gc.collect()

# ── Re-score every candidate pair with the fine-tuned model ──────────────────
total_cand_pairs = sum(len(v) for v in candidates_dict.values())
print(f"\n[*] Re-scoring {total_cand_pairs:,} candidate pairs with fine-tuned model...",
      flush=True)
model.eval()

MAX_MATCHES_PER_S1 = 11
match_path  = "output/matching_results.tsv"
cand_path   = "output/candidate_pairs.tsv"
n_with_match = 0
total_cands  = 0

with open(match_path, "w", encoding="utf-8") as fm, \
     open(cand_path,  "w", encoding="utf-8") as fc:

    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")

    for idx, eid in enumerate(test_eids):
        cand_map = candidates_dict.get(idx, {})
        total_cands += len(cand_map)

        if not cand_map:
            fm.write(f"{eid}\t\n")
            fc.write(f"{eid}\t\n")
            continue

        cand_ids    = list(cand_map.keys())
        feats       = [
            compute_inference_features(
                test_raw_names[idx], test_raw_addrs[idx], test_raw_ctry[idx],
                cand_map[cid][0], cand_map[cid][1], cand_map[cid][2]
            )
            for cid in cand_ids
        ]
        feat_tensor = torch.tensor(feats, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model(feat_tensor)).cpu().numpy()

        accepted = sorted(
            [cid for cid, p in zip(cand_ids, probs) if p >= best_thresh]
        )[:MAX_MATCHES_PER_S1]

        fc.write(f"{eid}\t{','.join(sorted(cand_ids))}\n")
        if accepted:
            n_with_match += 1
            fm.write(f"{eid}\t{','.join(accepted)}\n")
        else:
            fm.write(f"{eid}\t\n")

del candidates_dict, matches_dict, test_eids, test_raw_names, test_raw_addrs, test_raw_ctry
gc.collect()

match_size_mb = os.path.getsize(match_path) / (1024 * 1024)
print(
    f"[+] File Size: {match_path} → {match_size_mb:.2f} MB "
    f"({match_size_mb/512.0*100:.1f}% of 512 MB limit)", flush=True
)
if match_size_mb > 512.0:
    raise ValueError(f"CRITICAL: {match_path} exceeds the 512 MB portal limit!")

sync_checkpoint(
    step_title="Final Matching Results TSV",
    files_to_backup=[match_path, cand_path],
    commit_msg="checkpoint: generated matching_results.tsv and candidate_pairs.tsv"
)
print(f"\n[+] Inference done in {time.time()-t_inf_start:.1f}s!")
print(f"    - {match_path}  (Matches: {n_with_match:,} | "
      f"Singletons: {n_test_s1-n_with_match:,})")
print(f"    - {cand_path}   (Total candidates: {total_cands:,})", flush=True)


# ==============================================================================
# STEP 7 — Validate Submission & Final Package
# ==============================================================================
print("\n" + "=" * 70)
print("[Step 7/7] Validating Submission & Final Sync...")
print("=" * 70, flush=True)

validator_script = "student_resource/utils/validate_submission.py"
if os.path.exists(validator_script):
    subprocess.run([
        sys.executable, validator_script,
        "--matching",   match_path,
        "--candidate",  cand_path,
        "--test-dir",   "dataset/test"
    ], check=True)
else:
    print("[*] Validator script not found; skipping format check.", flush=True)

sync_checkpoint(
    step_title="Final Submission Package",
    files_to_backup=[match_path, cand_path, "model"],
    commit_msg="release: final verified fine-tuned model and submission files"
)

try:
    if os.path.exists("/content/drive/MyDrive"):
        zip_path = os.path.join(DRIVE_BACKUP_DIR, "submission_package.zip")
        os.system(f"zip -jq '{zip_path}' '{match_path}' '{cand_path}'")
        print(f"[+] Created submission zip: {zip_path}", flush=True)
except Exception as ex:
    print(f"[*] Zip packaging notice: {ex}", flush=True)

print("\n" + "=" * 70)
print("ALL DONE! Model fine-tuned, outputs verified, and all files")
print("    securely backed up to Google Drive & GitHub!")
print("=" * 70, flush=True)
