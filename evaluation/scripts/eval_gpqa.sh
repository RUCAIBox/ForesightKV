#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-gpqa}"
BASE_DIR="${BASE_DIR:-./outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs_gpqa}"
EXP_NAME="${EXP_NAME:-gpqa_eval}"

"${PYTHON_BIN}" evaluation/eval_gpqa.py \
  --exp_name "${EXP_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_dir "${BASE_DIR}" \
  --dataset "${DATASET}"
