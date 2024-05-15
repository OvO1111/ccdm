#!/bin/sh

# exp="ruijin_ccdm_64_128_128_w_itg_loss"
exp="runjin_test_replace_w_noise_vector"
ngpu=8

mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/ccdm -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/ruijin_ccdm_64_128_128.sh $exp $ngpu