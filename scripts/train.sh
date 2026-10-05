#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./output}"
BEST_MODEL_DIR="${BEST_MODEL_DIR:-./best_model}"
EPOCHS="${EPOCHS:-20}"

rubert-clf train \
  --output-dir "$OUTPUT_DIR" \
  --best-model-dir "$BEST_MODEL_DIR" \
  --epochs "$EPOCHS"