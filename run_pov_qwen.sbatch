#!/bin/bash
#SBATCH --job-name=pov-qwen
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pov-%j.out
#SBATCH --error=logs/pov-%j.err

set -uo pipefail
PY=/home/sethu/miniconda3/envs/iclr/bin/python
cd /mnt/data/sethu/data/iclr
export HF_HOME=/mnt/data/sethu/cache/hf
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}  start=$(date -Is)"

# One worker per GPU. Each walks all three experiments, taking its own third of
# every manifest, so the three finish together instead of one lagging on the
# 1400-row wbw_mcq. --resume makes re-submission cheap after a pre-emption.
for shard in 0 1 2; do
(
  export CUDA_VISIBLE_DEVICES=$shard
  for spec in "asl:data/asl/asl_v1" "chess:data/chess/chess_v1" "wbw_mcq:data/wbw_mcq/mcq_v1"; do
    exp=${spec%%:*}; rd=${spec#*:}
    echo "[gpu$shard] $exp start $(date -Is)"
    $PY scripts/predict_qwen.py --run-dir "$rd" \
        --num-shards 3 --shard "$shard" --resume \
        > "logs/${exp}.shard${shard}.log" 2>&1
    echo "[gpu$shard] $exp exit=$? $(date -Is)"
  done
) &
done
wait
echo "all shards done $(date -Is)"
