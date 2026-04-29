#!/bin/bash
# ============================================================
#  L3 DeBERTa v2 Training — AWS EC2 Setup Script
#  AMI     : Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.10 (Ubuntu 24.04)
#  Instance: g5.xlarge (A10G GPU, 24 GB VRAM)
#  Run this ONCE after SSH-ing into the instance as ubuntu.
#
#  We create a fresh venv with the system python3 (3.12 on U24.04),
#  then install torch 2.3 (cu121) + modern transformers stack.
#  The AMI's bundled NVIDIA driver supports CUDA 12.x.
# ============================================================

set -euo pipefail

echo "============================================"
echo "  L3 DeBERTa v2 — AWS Environment Setup (U24.04)"
echo "============================================"

# ── 1. Working directory ─────────────────────────────────────
WORK_DIR="$HOME/l3_training"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
echo "[1/5] Working directory: $WORK_DIR"

# ── 2. System packages ───────────────────────────────────────
echo "[2/5] Installing system packages ..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv tmux

# ── 3. Python virtual environment ────────────────────────────
echo "[3/5] Creating Python virtual environment ..."
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

# ── 4. Install Python dependencies ──────────────────────────
echo "[4/5] Installing Python packages (this takes 3-5 min) ..."
pip install -q --upgrade pip
# torch 2.3 + cu121 wheels — works with the AMI's NVIDIA driver (>=535).
pip install -q torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -q "transformers>=4.44,<5" "accelerate>=0.33" \
    "datasets>=2.20" "sentencepiece>=0.2" "protobuf>=4.25" \
    "scikit-learn>=1.3" "pandas>=2.0" "numpy<2" \
    "matplotlib>=3.8" "seaborn>=0.13"

# ── 5. Verify GPU + print versions ───────────────────────────
echo "[5/5] Verifying GPU access ..."
python -c "
import torch, transformers, accelerate, sys
print(f'  python       : {sys.version.split()[0]}')
print(f'  torch        : {torch.__version__}')
print(f'  transformers : {transformers.__version__}')
print(f'  accelerate   : {accelerate.__version__}')
print(f'  CUDA build   : {torch.version.cuda}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    print(f'  ✓ GPU detected: {name} ({mem:.1f} GB)  bf16={bf16}')
else:
    print('  ✗ NO GPU DETECTED — training will be very slow!')
    print('    Run \"nvidia-smi\" to debug.')
    sys.exit(1)
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Upload data:  scp reviews_enriched.csv reviewer_profiles.csv seller_profiles.csv → ~/l3_training/data/"
echo "  2. Start tmux:   tmux new -s training"
echo "  3. Activate env: cd ~/l3_training && source venv/bin/activate"
echo "  4. Run training: python l3_deberta_finetune_v2.py --data_dir ~/l3_training/data --output_root ~/l3_training --model_name microsoft/deberta-v3-base --bf16 --epochs 6 --batch_size 32"
