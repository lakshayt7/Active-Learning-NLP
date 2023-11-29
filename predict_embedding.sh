#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/biginference_jobid_%j.out
#SBATCH --error=logs/biginference_epoch_jobid_%j.err
#SBATCH --gres=gpu:a100:1

module load cuda90/toolkit/9.1.176
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /gpfs/scratch/lt2504/bert/bin/activate

python src/predict_embedding.py
