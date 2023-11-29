#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/activelearning_%j.out
#SBATCH --error=logs/activelearning%j.err

x=1
while [ $x -le 20 ]
do
  sbatch dist_select.sh $x
  echo "$x submitted"
  x=$(( $x + 1 ))
done





