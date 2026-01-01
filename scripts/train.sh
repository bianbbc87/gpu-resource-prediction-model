#!/bin/bash
set -e

# ===== env from Argo =====
CONFIG_PATH="${CONFIG_PATH}"
OUTPUT_DIR="${OUTPUT_DIR}"
GRAPH_DIR="${GRAPH_DIR}"
LABEL_DIR="${LABEL_DIR}"

echo "===================================="
echo "Starting training"
echo "CONFIG_PATH = $CONFIG_PATH"
echo "OUTPUT_DIR  = $OUTPUT_DIR"
echo "GRAPH_DIR   = $GRAPH_DIR"
echo "LABEL_DIR   = $LABEL_DIR"
echo "EXTRA ARGS  = $@"
echo "===================================="

mkdir -p "$OUTPUT_DIR"

# ===== build arg list =====
ARGS=(
  --config "$CONFIG_PATH"
  --output_dir "$OUTPUT_DIR"
)

if [ -n "$GRAPH_DIR" ]; then
  ARGS+=(--graph_dir "$GRAPH_DIR")
fi

if [ -n "$LABEL_DIR" ]; then
  ARGS+=(--label_dir "$LABEL_DIR")
fi

# ===== execute =====
python3 main.py "${ARGS[@]}" "$@"
