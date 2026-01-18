#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --account=aip-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# exit on error
set -e

# load modules
module load StdEnv/2023
module load python/3.11.5
module load cuda/12.2
module load arrow/21.0.0

# virtual environment path
VENV_PTH="$HOME/projects/aip-btaati/$USER/env/mc"

# job settings
MODEL="qwen2.5"
D_NAME="MMMU-medical"
K=1
MAX_EX=2

# setup environment
source "$VENV_PTH/bin/activate"
mkdir -p logs

# display
echo "Running $SLURM_JOB_NAME..."
echo "Model: qwen2.5vl-3b"
echo "Dataset: $D_NAME"

# run short job to check if things work
python inference.py \
  --model "$MODEL"\
  --d_name "$D_NAME" \
  --k "$K" \
  --max_examples "$MAX_EX"