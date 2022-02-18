#!/bin/bash
#SBATCH --job-name=threePN
#SBATCH --account=rrg-lplevass
#SBATCH --time=00-48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=47000M

source ~/.virtualenvs/diffbank-3.9.6/bin/activate

python genbank_2D_threePN.py --seed 1 --kind random --device gpu
python genbank_2D_threePN.py --seed 2 --kind stochastic --device gpu
