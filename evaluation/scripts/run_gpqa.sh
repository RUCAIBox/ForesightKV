#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
DATASET_PATH="${DATASET_PATH:-./data/gpqa.jsonl}"
SAVE_PATH="${SAVE_PATH:-./outputs/gpqa.jsonl}"

METHOD="${METHOD:-fullkv}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
TIMES="${TIMES:-1}"
KV_BUDGET="${KV_BUDGET:-1024}"
WINDOW_SIZE="${WINDOW_SIZE:-1500}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

mkdir -p "$(dirname "${SAVE_PATH}")"

"${PYTHON_BIN}" ./run_gpqa.py \
  --dataset_path "${DATASET_PATH}" \
  --save_path "${SAVE_PATH}" \
  --model_path "${MODEL_PATH}" \
  ${TOKENIZER_PATH:+--tokenizer_path "${TOKENIZER_PATH}"} \
  --max_length "${MAX_LENGTH}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --method "${METHOD}" \
  --kv_budget "${KV_BUDGET}" \
  --times "${TIMES}" \
  --window_size "${WINDOW_SIZE}" \
  --attn_implementation "${ATTN_IMPL}"
