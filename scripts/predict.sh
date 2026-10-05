#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./best_model}"
FILE="${1:-}"

if [[ -n "$FILE" ]]; then
  rubert-clf predict --model-dir "$MODEL_DIR" --file "$FILE"
else
  rubert-clf predict --model-dir "$MODEL_DIR"
fi