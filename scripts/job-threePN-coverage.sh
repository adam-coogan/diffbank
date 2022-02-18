#!/bin/bash
#SBATCH --job-name=3pn-neff
#SBATCH --account=rrg-lplevass
#SBATCH --time=00-48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --array=1-50

source ~/.envs/diffbank-3.9.6/bin/activate

neffs=(10 11 13 15 17 20 23 26 30 35 40 47 54 62 71 82 95 109 126 145 167 193 222 255 294 339 390 449 517 596 686 790 910 1048 1206 1389 1599 1842 2120 2442 2811 3237 3727 4291 4941 5689 6551 7543 8685 10000)
seeds=(500 501 502 503 504 505 506 507 508 509 510 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550)

# Get neff and seed for this job
python genbank_2D_threePN.py \
    --n-eff ${neffs[$SLURM_ARRAY_TASK_ID]} \
    --seed ${seeds[$SLURM_ARRAY_TASK_ID]} \
    --kind random \
    --device gpu \
    --n-eta 4000 \
    --noise analytic

wait
