#!/bin/bash
#SBATCH --job-name=tf2
#SBATCH --account=rrg-lplevass
#SBATCH --time=07-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=47000M

source ~/.virtualenvs/diffbank-3.9.6/bin/activate

python genbank_4D_taylorf2.py --seed 1 --kind random --device gpu
