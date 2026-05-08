# Supervised  Training
Firstly, we can train the scoring models with supervised training via the following code
```bash
cd supervised_training
python train.py \
    --model_name  Qwen3-4B \
    --dataset path/to/supervised-data \
    --checkpoint_path checkpoints/r1kv-sl
```

# Reinforcement Learning
Secondly, we can train the scoring models with reinforcement learning via the following code to refine the models obtained from the above step.

```bash
cd reinforcement_learning
python train.py \
    --model_name  checkpoints/r1kv-sl \
    --dataset path/to/reinforcement-data \
    --checkpoint_path checkpoints/r1kv-rl
```


# Evaluation
Thirdly, we can evaluate the models with the following code to compare it with other method.
```bash
cd evaluation
bash scripts/run.sh
bash scripts/eval.sh
```
