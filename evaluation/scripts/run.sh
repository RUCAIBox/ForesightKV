model_path=qwen3_rl


python3 ./run_math.py --dataset_path ./data/aime24.jsonl --save_path ./outputs/qwen3_rl_aime24_1k.jsonl --model_path $model_path --max_length 32000 --eval_batch_size 32 --method futurekv  --kv_budget 1024 --retain_ratio 0.1 --times 32 --window_size 1500 &
