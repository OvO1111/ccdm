#!/bin/sh

exp="ruijin_ensemble_64_128_128_baseline"
ngpu=$1

mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/ccdm -N 1 -n $(($ngpu*6)) -J $exp --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/ensemble_ccdm_64_128_128.sh $exp $ngpu