#!/bin/bash
set -e

# 기본값 (Argo에서 env로 override 가능)
CONFIG_PATH=${CONFIG_PATH}
DATA_DIR=${DATA_DIR}
OUTPUT_DIR=${OUTPUT_DIR}

echo "===================================="
echo "Starting training"
echo "CONFIG_PATH = $CONFIG_PATH"
echo "DATA_DIR    = $DATA_DIR"
echo "OUTPUT_DIR  = $OUTPUT_DIR"
echo "ARGS        = $@"
echo "===================================="

mkdir -p "$OUTPUT_DIR"

python train.py \
  --config "$CONFIG_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
