# ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution ([Paper](https://arxiv.org/abs/2602.03203))

**Accepted at ICML 2026**

![ForesightKV overview](./assets/foresightkv-main.png)

This repository contains the training and evaluation code for ForesightKV.

## Installation

Install the Python dependencies first:

```bash
conda create -n foresightkv python=3.10
conda activate foresightkv
pip install -r requirements.txt
```

The training scripts use `flash_attention_2`. Install a compatible `flash-attn`
build separately if you plan to run supervised training or reinforcement
learning on GPU.

## Supervised Training

```bash
cd supervised_training
python train.py \
    --model_name path/to/qwen3-base-model \
    --dataset path/to/supervised-data \
    --checkpoint_path checkpoints/r1kv-sl
```

Qwen2 variant:

```bash
cd supervised_training
python train_qwen2.py \
    --model_name path/to/qwen2-base-model \
    --dataset path/to/supervised-data \
    --checkpoint_path checkpoints/r1kv-qwen2-sl
```

Notes:

- `--dataset` should point to a Hugging Face dataset saved with `load_from_disk`.
- `train.py` and `train_qwen2.py` infer layer count and KV head layout from the
  loaded config, so they are not limited to a single model size.
- the current script expects at least 2 CUDA devices because it places the
  train model on `cuda:0` and the reference model on `cuda:1`

## Reinforcement Learning

```bash
cd reinforcment_learning
torchrun --nproc_per_node=NUM_GPUS train.py \
    --model_name checkpoints/r1kv-sl \
    --data_name path/to/reinforcement-data \
    --checkpoint_path checkpoints/r1kv-rl
```

Qwen2 variant:

```bash
cd reinforcment_learning
torchrun --nproc_per_node=NUM_GPUS train_qwen2.py \
    --model_name checkpoints/r1kv-qwen2-sl \
    --data_name path/to/reinforcement-data \
    --checkpoint_path checkpoints/r1kv-qwen2-rl \
    --judge_init_path checkpoints/r1kv-qwen2-sl
```

Notes:

- the directory name is `reinforcment_learning` in this repository
- `--data_name` should point to a Hugging Face dataset saved with `load_from_disk`
- `train.py` and `train_qwen2.py` both accept `--total_training_steps`,
  `--rollouts_per_step`, `--checkpoint_interval`, and related RL hyperparameters
  as CLI arguments

## Evaluation

Generation:

```bash
cd evaluation
python run_math.py \
    --dataset_path ./data/aime24.jsonl \
    --save_path ./outputs/example.jsonl \
    --model_path path/to/model \
    --method fullkv
```

Common arguments:

- `--method`: KV cache strategy. Supported choices are `fullkv`, `rkv`, `snapkv`, `streamingllm`, `h2o`, `foresightkv`, and `foresightkv_topk`.
- `--kv_budget`: KV retention budget used by compressed methods. Leave it unset for `fullkv`.
- `--max_length`: maximum sequence length during generation. We recommend using `32768` for long-context reasoning evaluation.
- `--eval_batch_size`: evaluation batch size. The default is `1`.
- `--times`: repeat count per example, useful when sampling multiple outputs from the same prompt.
- `--attn_implementation`: attention backend, with choices `flash_attention_2`, `sdpa`, and `eager`.

Method-related hyperparameters:

- `--window_size`: local sliding-window size used by compressed KV methods. Default is `8`.
- `--first_tokens`: always-retained prefix token count for some methods. Default is `4`.
- `--mix_lambda`: mixing weight used by specific heuristics such as `h2o`. Default is `0.1`.
- `--retain_ratio`: token retention ratio used by `rkv`. Default is `0.2`.
- `--retain_direction`: retention direction, either `last` or `first`. Default is `last`.
- `--update_kv`: whether to update the KV cache online during generation. Default is `True`.

ForesightKV model-side options:

- For `foresightkv`, `window_size` should be larger than `kv_budget + divide_length`.
- `--divide_method`: segment split rule for reasoning traces, with choices `step_length` and `newline`.
- `--divide_length`: segment length when `divide_method=step_length`. Default is `128`.
- `--compression_content`: whether to compress `all` generated content or only the `think` part.

Example with ForesightKV compression:

```bash
cd evaluation
python run_math.py \
    --dataset_path ./data/aime24.jsonl \
    --save_path ./outputs/foresightkv-aime24.jsonl \
    --model_path path/to/model \
    --method foresightkv \
    --max_length 32768 \
    --kv_budget 1024 \
    --window_size 2048 \
    --first_tokens 4 \
    --divide_method step_length \
    --divide_length 128 \
    --compression_content all
```

Scoring:

```bash
cd evaluation
python evaluation/eval_math.py \
    --exp_name example \
    --output_dir ./eval_outputs_example \
    --base_dir ./outputs \
    --dataset aime24
```

GPQA data is available at `evaluation/data/gpqa.jsonl`. The public copy keeps
only the question (`input`) and gold answer (`output`).

GPQA generation:

```bash
cd evaluation
MODEL_PATH=path/to/model bash scripts/run_gpqa.sh
```

GPQA scoring:

```bash
cd evaluation
BASE_DIR=./outputs OUTPUT_DIR=./eval_outputs_gpqa bash scripts/eval_gpqa.sh
```

## Citation

If you use this repository, please cite:

```bibtex
@article{dong2026foresightkv,
  title={ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution},
  author={Dong, Zican and Liu, Peiyu and Li, Junyi and Chen, Zhipeng and Peng, Han and Wang, Shuo and Zhao, Wayne Xin},
  journal={arXiv preprint arXiv:2602.03203},
  year={2026}
}
```
