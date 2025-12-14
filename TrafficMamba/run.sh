#!/bin/bash
#SBATCH --job-name=traffic-mamba-test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fatemehdoudi@tamu.edu
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --output=logs/test_main_%j.out

# ----------------------------
# Paths & cache (safe defaults)
# ----------------------------
export MY_SCRATCH=/mnt/shared-scratch/Kalathil_D/fatemehdoudi
export TORCH_HOME=$MY_SCRATCH/torch
export TMPDIR=$MY_SCRATCH/tmp
mkdir -p "$TORCH_HOME" "$TMPDIR" logs

# ----------------------------
# Activate environment
# ----------------------------
source ~/.bashrc
conda activate mamba

echo "Python:" $(which python)
python - << EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# ----------------------------
# Move to project root
# ----------------------------
cd /mnt/shared-scratch/Kalathil_D/fatemehdoudi/TrafficMamba

echo "Running main.py (forward pass test)"

# ----------------------------
# Run entry script
# ----------------------------
python main.py \
  --seq_len 64 \
  --batch_size 256 \
  --lr 1e-3 \
  --epochs 50
echo "Job finished."
