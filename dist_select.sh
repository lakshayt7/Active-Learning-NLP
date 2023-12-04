#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/activelearning_%j.out
#SBATCH --error=logs/activelearning%j.err


module load cuda90/toolkit/9.1.176
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

algo='rand'
encoder='sentence'

outfolder="/gpfs/home/lt2504/pathology-extractor-bert/data/select/"


infile="/gpfs/home/lt2504/pathology-extractor-bert/embeddings_filter/out_1.npy"

source /gpfs/scratch/lt2504/bert/bin/activate


python3 src/coreset_distributed.py --outdir "$outfolder"  --budget 1000 --ifname "/gpfs/home/lt2504/pathology-extractor-bert/embeddings_filter/out_${1}.npy" --text_pth "/gpfs/home/lt2504/pathology-extractor-bert/data/raw/historical_report_parts_filter/${1}_historical_reports.csv" --ofname "${1}_selected.csv"
 



