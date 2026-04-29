#!/bin/bash
#SBATCH --job-name=l3-deberta-v3
#SBATCH --output=l3_deberta_%j.out
#SBATCH --error=l3_deberta_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --mail-user=himanshu.jain@sjsu.edu
#SBATCH --mail-type=END,FAIL

# ── Force offline mode (compute nodes have no internet) ──────
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── Fix libstdc++ for sentencepiece (DeBERTa v3 tokenizer) ───
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/spack/opt/spack/linux-centos7-broadwell/gcc-11.2.0/gcc-12.2.0-7gle75fpui2uzq74izjwiloxtobg4v4v/lib64:${LD_LIBRARY_PATH}

# ── Load modules ─────────────────────────────────────────────
module load python3/3.11.5
module load cuda

# ── Activate venv ────────────────────────────────────────────
source ~/l3_training/venv/bin/activate

# ── Run training (P100-safe, DeBERTa v3-base) ────────────────
cd ~/l3_training

python -u l3_deberta_finetune_v2.py \
    --data_dir ~/l3_training/data \
    --output_root ~/l3_training \
    --model_name microsoft/deberta-v3-base \
    --max_length 128 \
    --epochs 4 \
    --batch_size 16 \
    --patience 2

echo "Job finished at $(date)"
