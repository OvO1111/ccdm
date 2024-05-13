#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ccdm
wandb offline

export PYTHONUNBUFFERED=1

echo $1 $2
accelerate launch --num_processes $2 --num_machines 1 --main_process_port 6066 main.py -cfg ./configs/train_ruijin_ccdm.yaml trainer.params.snapshot_path=/ailab/user/dailinrui/data/ccdm/$1 > ./runs/$1/out.txt 2>&1
