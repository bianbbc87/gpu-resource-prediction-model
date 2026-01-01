#!/bin/bash

EXP=perfseer_a100
CONFIG=configs/exp_a100.yaml
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  python main.py \
    --config $CONFIG \
    --seed $SEED \
    --output_dir outputs/$EXP/seed_$SEED
done
