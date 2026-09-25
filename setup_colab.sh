#!/usr/bin/env bash
# Amazon ML Challenge 2026 — Colab TPU Automated Setup Script

set -e

echo "============================================================"
echo "    Amazon ML Challenge 2026 — Colab TPU Environment Setup   "
echo "============================================================"

# 1. Check Hardware Accelerator
echo "[*] Verifying Hardware Accelerator (TPU)..."
python3 -c "
try:
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print(f'[+] TPU Detected and Active: {dev}')
except Exception as e:
    print(f'[!] Warning: TPU not active ({e}).')
    print('    If you want TPU, ensure your Colab runtime is set to TPU accelerator.')
"

# 2. Install Required Dependencies
echo "[*] Installing High-Performance Libraries (polars, rapidfuzz, lightgbm, scikit-learn)..."
pip install -q rapidfuzz polars lightgbm scikit-learn

# 3. Mount Google Drive if not already mounted
echo "[*] Checking Google Drive Mount..."
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        print('[*] Mounting Google Drive to /content/drive...')
        drive.mount('/content/drive')
        print('[+] Google Drive successfully mounted!')
    except Exception as ex:
        print(f'[!] Notice: Could not mount Drive automatically ({ex}).')
else:
    print('[+] Google Drive is already mounted at /content/drive/MyDrive')
"

# 4. Dataset linking
echo "[*] Checking Dataset Configuration..."
if [ -d "dataset/train" ] && [ -f "dataset/train/train_source1.tsv" ]; then
    echo "[+] Local dataset directory found and verified!"
elif [ -d "student_resource/dataset/train" ]; then
    echo "[*] Linking dataset from student_resource/dataset..."
    ln -sfn student_resource/dataset dataset
    echo "[+] Dataset linked successfully!"
elif [ -d "/content/drive/MyDrive/student_resource/dataset" ]; then
    echo "[*] Linking dataset from Google Drive (/content/drive/MyDrive/student_resource/dataset)..."
    ln -sfn /content/drive/MyDrive/student_resource/dataset dataset
    echo "[+] Dataset linked from Google Drive!"
elif [ -d "/content/drive/MyDrive/dataset" ]; then
    echo "[*] Linking dataset from Google Drive (/content/drive/MyDrive/dataset)..."
    ln -sfn /content/drive/MyDrive/dataset dataset
    echo "[+] Dataset linked from Google Drive!"
elif [ -f "/content/drive/MyDrive/student_resource.tar.gz" ]; then
    echo "[*] Found student_resource.tar.gz in Google Drive. Extracting..."
    tar -xzf /content/drive/MyDrive/student_resource.tar.gz -C .
    ln -sfn student_resource/dataset dataset
    echo "[+] Extracted and linked dataset!"
elif [ -f "/content/student_resource.tar.gz" ]; then
    echo "[*] Found /content/student_resource.tar.gz. Extracting..."
    tar -xzf /content/student_resource.tar.gz -C .
    ln -sfn student_resource/dataset dataset
    echo "[+] Extracted and linked dataset!"
else
    echo "[!] Warning: Dataset not found in standard paths."
    echo "    Please upload student_resource to Google Drive or /content, then link with:"
    echo "    ln -sfn /path/to/dataset dataset"
fi

mkdir -p model output

echo "============================================================"
echo "[+] Environment setup complete!"
echo "To train the model on TPU, run:"
echo "    python3 -m src.train_tpu --train-dir dataset/train --model-dir model --epochs 10 --batch-size 2048"
echo "============================================================"
