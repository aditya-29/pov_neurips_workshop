#!/bin/bash
#SBATCH --job-name=pov-fps
#SBATCH --partition=h200
#SBATCH --gres=gpu:nvidia_h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fps-%j.out
#SBATCH --error=logs/fps-%j.err

# Aditya's TODO 1 + 2: sweep the frame-sampling rate given to the model over
# 1/5/10/15/20/30 fps, on open-weight models only.
#
# The id-lists come from scripts/make_fps_plan.py, which drops cells where
# max_frames makes a higher fps produce byte-identical input -- 840 chess runs
# collapse to 340, 1020 asl runs to 730. Running the duplicates would report
# six equal numbers as if fps had been varied and found not to matter.
set -uo pipefail
PY=/home/sethu/miniconda3/envs/iclr/bin/python
cd /mnt/data/sethu/data/iclr
export HF_HOME=/mnt/data/sethu/cache/hf
export TOKENIZERS_PARALLELISM=false

CHESS=data/chess/20260824-001859-547f861a109a
ASL=data/asl/20260824-002004-e1225b96852f
PLAN=data/fps_sweep_plan

echo "host=$(hostname) start=$(date -Is)"

for spec in "Qwen/Qwen3.5-9B:q35_9b" "Qwen/Qwen3-VL-8B-Instruct:q3vl_8b"; do
  MODEL=${spec%%:*}; SLUG=${spec#*:}
  for pair in "chess:$CHESS" "asl:$ASL"; do
    EXP=${pair%%:*}; RD=${pair#*:}
    for F in 1 5 10 15 20 30; do
      IDS=$PLAN/${EXP}_fps${F}.ids
      [ -s "$IDS" ] || { echo "skip $EXP fps$F (empty id list)"; continue; }
      echo "[$SLUG/$EXP/fps$F] $(wc -l < $IDS) ids  $(date -Is)"
      for shard in 0 1 2; do
        (
          export CUDA_VISIBLE_DEVICES=$shard
          $PY scripts/predict_qwen.py --run-dir "$RD" \
              --model "$MODEL" --fps "$F" --only-ids "$IDS" \
              --tag "${SLUG}_fps${F}" --num-shards 3 --shard "$shard" --resume \
              > "logs/sweep.${SLUG}.${EXP}.fps${F}.s${shard}.log" 2>&1
        ) &
      done
      wait
      echo "[$SLUG/$EXP/fps$F] done $(date -Is)"
    done
  done
done
echo "sweep complete $(date -Is)"
