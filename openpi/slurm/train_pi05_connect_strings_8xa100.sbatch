#!/bin/bash
# Full-parameter pi0.5 fine-tuning on 8 GPUs. Adjust the partition to your cluster.
#SBATCH --job-name=pi05_connect_strings
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -euo pipefail
cd "${RECAP_ROOT:-$HOME/RECAP}/openpi"
mkdir -p logs
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
.venv/bin/python scripts/train.py pi05_connect_strings \
  --exp-name=pi05_connect_strings_repro --no-wandb-enabled --overwrite \
  --batch-size=128 --fsdp-devices=8 --num-train-steps=30001
