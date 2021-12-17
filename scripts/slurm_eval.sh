#!/bin/bash

#SBATCH --job-name=uORF_eval
#SBATCH --ntasks=1
#SBATCH --time=00:50:00

# Number of gpus. Available nodes: dev_gpu_4, gpu_4, gpu_8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4

# Output files
#SBATCH --output=../out/E-%x.%j.out
#SBATCH --error=../out/E-%x.%j.err

echo 'Activate conda environment'
source ~/.bashrc
conda activate uorf-3090

echo 'Start evaluation'
exec scripts/eval.sh