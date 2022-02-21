#!/bin/bash

#SBATCH --job-name=uORF
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

# Number of gpus. Available nodes: dev_gpu_4, gpu_4, gpu_8
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_4
#SBATCH --cpus-per-gpu=10

# Output files
#SBATCH --output=../out/T-%x.%j.out
#SBATCH --error=../out/T-%x.%j.err

# 216:00:00
# 48:00:00
# 00:30:00

echo 'Activate conda environment'
source ~/.bashrc
conda activate uorf-3090

echo 'Start training'
exec scripts/train.sh
