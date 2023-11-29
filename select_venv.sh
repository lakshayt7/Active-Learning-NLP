#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/activelearning_%j.out
#SBATCH --error=logs/activelearning%j.err
#SBATCH --gres=gpu:a100:1

module load cuda90/toolkit/9.1.176
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

algo='coreset-simple'
encoder='sentence'

outfolder="/gpfs/home/lt2504/pathology-extractor-bert/data/splits/active_out_{$algo}_{$encoder}/"

model_out="/gpfs/data/chopralab/ad6489/pathology-extractor-bert/models/finetuned/active_{$algo}_{$encoder}"

selectmodel="/gpfs/home/lt2504/pathology-extractor-bert/models/pretrained/historical/bignyutron49/checkpoint-7725"

source /gpfs/scratch/lt2504/bert/bin/activate 

cp /gpfs/home/lt2504/pathology-extractor-bert/data/splits/active/val.csv "$outfolder"
cp /gpfs/home/lt2504/pathology-extractor-bert/data/splits/active/test.csv "$outfolder"

python3 src/activeselect.py --outdir "$outfolder" --model_path "$selectmodel" --alg algo --encoder "$encoder"

python3 src/train_bert.py --in_folder "$outfolder" --out_folder "$model_out"




