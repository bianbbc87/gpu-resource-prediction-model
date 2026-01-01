#!/bin/bash

CONFIG=configs/exp_a100.yaml
EXP=outputs/exp_a100
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  python main.py \
    --config $CONFIG \
    --seed $SEED \
    --output_dir $EXP/seed_$SEED
done
