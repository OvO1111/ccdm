#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ccdm
wandb offline

export PYTHONUNBUFFERED=1

echo $1 $2 $3
torchrun --nproc-per-node=$2 main.py --base ./configs/train_ruijin_ccdm.yaml --gpus $3 --name $exp -t > ./runs/$exp/out.txt 2>&1
