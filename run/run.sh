#!/bin/sh

# exp="ruijin_ccdm_64_128_128_w_itg_loss"
exp=""
ngpu=8

i=0
igpus=""
for (( ; i<ngpu; i++ )); do
  if [ $i -ne 0 ]; then
    igpus+=","
  fi
  igpus+="$i"
done

mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/ccdm -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/ruijin_ccdm_64_128_128.sh $exp $ngpu