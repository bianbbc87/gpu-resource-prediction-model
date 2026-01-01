#!/bin/bash
set -e

CONFIG_PATH="${CONFIG_PATH}"
OUTPUT_DIR="${OUTPUT_DIR}"

echo "===================================="
echo "Starting training"
echo "CONFIG_PATH = $CONFIG_PATH"
echo "OUTPUT_DIR  = $OUTPUT_DIR"
echo "===================================="

python3 main.py \
  --config "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR"
