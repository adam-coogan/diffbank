#!/bin/bash
#SBATCH --job-name=p
#SBATCH --account=rrg-lplevass
#SBATCH --time=00-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=47000M

source ~/.virtualenvs/diffbank-3.9.6/bin/activate

python threePN_est_p_variation.py --mm 0.95 &
python threePN_est_p_variation.py --mm 0.8 &

wait
