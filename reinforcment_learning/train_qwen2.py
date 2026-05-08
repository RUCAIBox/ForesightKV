import argparse
from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
# import swanlab as wandb
import math
import torch
import torch.optim as optim
import os
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
)
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from cache import init_cache_rl, cache_process_rl
from qwen2 import Qwen2ForCausalLM, Qwen2Attention

Qwen2Attention.init_cache = init_cache_rl
Qwen2Attention.cache_process = cache_process_rl


def sequence_log_probs_from_logits(
    logits: torch.tensor,
    indices: torch.tensor, 
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(log_prob * torch.exp(log_prob), dim=-1)
    return torch.gather(log_prob, dim=-1, index = indices ), entropy


def extract_attention_cache(attentions):
    if attentions is None:
        return None, None

    indices_per_layer = []
    scores_per_layer = []
    for layer_cache in attentions:
        if not layer_cache:
            return None, None
        layer_indices = [item[0] for item in layer_cache if item[0] is not None]
        layer_scores = [item[1] for item in layer_cache if item[1] is not None]
        if not layer_indices or not layer_scores:
            return None, None
        indices_per_layer.append(torch.stack(layer_indices, dim=0))
        scores_per_layer.append(torch.stack(layer_scores, dim=0))

    return torch.stack(indices_per_layer, dim=0), torch.stack(scores_per_layer, dim=0)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    dist.destroy_process_group()

def load_model(model_name_or_path, trust_remote_code=False, bf16=True, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    attn_impl = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    model = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    
   
    return model, tokenizer, None


def load_selection_weights(model: nn.Module, checkpoint_path: str) -> int:
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Judge checkpoint path does not exist: {checkpoint_dir}")

    state_dict = {}
    for shard_path in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt") as shard:
            for key in shard.keys():
                if "judge_model" in key or "reference" in key:
                    state_dict[key] = shard.get_tensor(key)

    if not state_dict:
        raise ValueError(f"No judge/reference weights found under: {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict)
    used_missing = [k for k in missing if ("judge_model" in k or "reference" in k)]
    used_unexpected = [k for k in unexpected if ("judge_model" in k or "reference" in k)]
    if used_missing or used_unexpected:
        raise ValueError(
            f"Selection weight mismatch for {checkpoint_dir}. "
            f"Missing={used_missing[:8]}, Unexpected={used_unexpected[:8]}"
        )
    return loaded


def load_rl_dataset(data_path: str):
    from datasets import load_dataset, load_from_disk

    try:
        return load_from_disk(data_path)
    except (FileNotFoundError, ValueError):
        data_dir = Path(data_path)
        parquet_files = sorted((data_dir / "parquet").glob("part-*.parquet"))
        if parquet_files:
            dataset = load_dataset("parquet", data_files={"train": [str(p) for p in parquet_files]})
            return dataset["train"]
        parquet_files = sorted((data_dir / "data").glob("train-*.parquet"))
        if parquet_files:
            dataset = load_dataset("parquet", data_files={"train": [str(p) for p in parquet_files]})
            return dataset["train"]
        raise


def get_sample_fields(sample, device: torch.device):
    input_ids = sample["input_ids"]
    mean_loss = sample.get("loss", sample.get("mean_loss"))
    token_loss = sample.get("loss2", sample.get("token_loss"))
    entropy = sample.get("entropy")

    if mean_loss is None:
        raise KeyError("Sample is missing both 'loss' and 'mean_loss'.")
    if token_loss is None:
        raise KeyError("Sample is missing both 'loss2' and 'token_loss'.")
    if entropy is None:
        # Fall back to token-wise loss as a ranking signal when entropy is unavailable.
        entropy = token_loss

    token_loss_tensor = torch.as_tensor(token_loss, device=device)
    entropy_tensor = torch.as_tensor(entropy, device=device)
    if entropy_tensor.ndim == 2 and entropy_tensor.shape[0] == 1:
        entropy_tensor = entropy_tensor[0]

    return (
        input_ids,
        float(mean_loss),
        token_loss_tensor,
        entropy_tensor,
    )

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return float(current_step+1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def custom_collate(batch):
    return list(batch)


def get_chunk_config(seq_len: int) -> tuple[int, int]:
    if seq_len > 12000:
        return 512, 2048
    return 256, 1024


def init_model_chunks(model: nn.Module, seq_len: int) -> tuple[int, int]:
    window, budget = get_chunk_config(seq_len)
    for layer in model.model.layers:
        layer.self_attn.init_chunk(budget, window)
    return window, budget

@torch.no_grad()
def rollout(model, tokenizer, input_ids, num_rollouts, generator, disable_eviction: bool = False):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    input_ids = torch.as_tensor(input_ids, device=model.device).repeat(num_rollouts, 1)
    init_model_chunks(model, input_ids.shape[-1])
    output = model(
        input_ids=input_ids,
        use_cache=True,
        labels=input_ids,
        output_attentions=True,
        output_logits=True,
        cache_drop=None,
        disable_eviction=disable_eviction,
    )

    a = input_ids[:, 1:].reshape(-1)
    loss = criterion(output.logits.float()[:, :-1].reshape(a.shape[0], -1), a)

    indices, scores = extract_attention_cache(output.attentions)
    if indices is None or scores is None:
        return None, None, None, None, None
    scores = scores.reshape(indices.shape[0], indices.shape[1], num_rollouts, indices.shape[3], -1)

    return input_ids, indices, scores, output.loss, loss

def init_rng(seed: int) -> torch.Generator:
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return torch.Generator(device="cuda").manual_seed(seed + rank)

def group_advantages(returns, rank,  eps = 1e-8) -> torch.Tensor:
    all_returns = [torch.zeros_like(returns) for _ in range(dist.get_world_size())]
    dist.all_gather(all_returns, returns)
    all_returns = torch.cat(all_returns, dim=0)
    
    mean = all_returns.mean(dim=0, keepdim=True)
    std = all_returns.std(dim=0, keepdim=True, unbiased=False) + eps
    return (returns - mean) / std, all_returns, (all_returns - mean) / std

def loss_compare(loss):
    all_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
    dist.all_gather(all_loss, loss)
    return torch.mean(torch.cat(all_loss,dim=0))


def dist_mean_scalar(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def dist_broadcast_data(data, src=0):
    payload = [data if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]
def compute_log_prob_from_indices(keep_indices, probs, eps=1e-8):
   
    
    B, T, H,  L = probs.shape
    _, _, _ , K = keep_indices.shape

    z = torch.zeros_like(probs)  
    z.scatter_(dim=3, index=keep_indices, value=1.0)  # 1 = keep, 0 = drop

    log_probs = z * torch.log(probs + eps) + (1 - z) * torch.log(1 - probs + eps)  # [B, T, L]

    log_probs = log_probs.mean(dim=-1)  # [B, T, H]

    return log_probs  # [B, T]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total_training_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--rollouts_per_step", type=int, default=32)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="r1kv-rl")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None)
    parser.add_argument("--swanlab_logdir", type=str, default=None)
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    parser.add_argument("--judge_init_path", type=str, default=None)
    parser.add_argument("--no_evict_warmup_steps", type=int, default=20)
    args = parser.parse_args()
    setup_ddp()
    rank = dist.get_rank()
    seed = 42
   
    
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    checkpoint_interval = args.checkpoint_interval
    lr = args.lr
    total_training_steps = args.total_training_steps
    warmup_steps = args.warmup_steps
    
    kl_weight = args.kl_weight
    clip_eps = args.clip_eps
    group_size = args.group_size
    rollouts_per_step = args.rollouts_per_step
    epochs_per_step = args.epochs_per_step
    max_norm = args.max_norm
    device = torch.device("cuda", rank % torch.cuda.device_count())
    generator = init_rng(seed)
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    swanlab_run = None
    if args.use_swanlab and rank == 0:
        import swanlab

        default_logdir = checkpoint_dir / "log" / "swanlab" if checkpoint_dir is not None else Path.cwd() / "swanlab"
        swanlab_logdir = Path(args.swanlab_logdir).expanduser().resolve() if args.swanlab_logdir else default_logdir
        swanlab_logdir.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            swanlab.login(api_key=api_key, save=False)

        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name,
            config={
                "model_name": model_name,
                "data_name": args.data_name,
                "checkpoint_path": str(checkpoint_dir) if checkpoint_dir is not None else None,
                "checkpoint_interval": checkpoint_interval,
                "lr": lr,
                "total_training_steps": total_training_steps,
                "warmup_steps": warmup_steps,
                "kl_weight": kl_weight,
                "clip_eps": clip_eps,
                "group_size": group_size,
                "rollouts_per_step": rollouts_per_step,
                "epochs_per_step": epochs_per_step,
                "max_norm": max_norm,
            },
            logdir=str(swanlab_logdir),
            mode=args.swanlab_mode,
        )
    model, tokenizer, model2 = load_model(model_name)
    if model.device != device:
        model.to(device)
    if args.judge_init_path:
        loaded = load_selection_weights(model, args.judge_init_path)
        if rank == 0:
            print(f"Loaded {loaded} selection tensors from {args.judge_init_path}")
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    num_layers = len(model.module.model.layers)
    for l in range(num_layers):
        with torch.no_grad():
            model.module.model.layers[l].self_attn.reference.load_state_dict(model.module.model.layers[l].self_attn.judge_model.state_dict())
            for param in model.module.model.layers[l].self_attn.reference.parameters():
                param.requires_grad = False
  
    to_optim = [p for n, p in model.named_parameters() if "judge_model" in n]
    optimizer = optim.Adam(to_optim, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    model.module.gradient_checkpointing_disable()
    dataset = load_rl_dataset(args.data_name)
    if rank == 0:
        prompt_dataloader = DataLoader(
            dataset,
            batch_size=rollouts_per_step,
            sampler=RandomSampler(dataset),
            drop_last=True,
            collate_fn=custom_collate,
        )
        prompt_loader = iter(prompt_dataloader)
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
    for k in tqdm(range(total_training_steps)):
        step_loss_value = None
        disable_eviction = k < args.no_evict_warmup_steps
        objective.kl_weight = 0.0 if k > 100 else kl_weight
        if dist.get_rank() == 0:
            try:
                prompt_batch = next(prompt_loader)
            except StopIteration:
                prompt_loader = iter(prompt_dataloader)
                prompt_batch = next(prompt_loader)
        else:
            prompt_batch = None
        prompt_batch = dist_broadcast_data(prompt_batch)
        
      
        rollout_returns = []
        replay_buffer.clear()
        torch.cuda.empty_cache()
        
        all_losses = []
        with torch.no_grad():
            for sample in prompt_batch:
                input_ids, loss, loss2, entropy = get_sample_fields(sample, device)
                input_ids, indices, scores, returns, lm_loss = rollout(
                    model.module,
                    tokenizer,
                    input_ids,
                    group_size,
                    generator,
                    disable_eviction=disable_eviction,
                )
                if indices is None:
                    continue
                
                window, budget = init_model_chunks(model.module, input_ids.shape[-1])
                with torch.no_grad():
                    init_output = model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_attentions=True,
                        cache_drop=indices,
                        output_logits=False,
                        reference=True,
                        disable_eviction=disable_eviction,
                    )
                    _, init_scores = extract_attention_cache(init_output.attentions)
                    if init_scores is None:
                        continue
                    init_scores = init_scores.reshape(indices.shape[0], indices.shape[1], group_size, indices.shape[3], -1)
                
                entropy_tail = entropy[window + budget :]
                top_20_percent_idx = int(len(entropy_tail) * 0.2)
                sorted_indices = torch.argsort(entropy_tail, descending=True)
                top_20_indices = sorted_indices[:top_20_percent_idx]
                other_indices = sorted_indices[top_20_percent_idx:]
               
                loss_high = torch.mean(lm_loss[window+budget:][top_20_indices])
                loss_low = lm_loss[window+budget:][other_indices]
                loss_low_ori  = loss2[window+budget:][other_indices]

                new_loss = torch.mean(loss_low)
                print(new_loss, torch.mean(lm_loss))
               
                advantages, all_returns, all_advantages = group_advantages(-new_loss.unsqueeze(0), rank)
                
                experience = Experience(input_ids=input_ids, indices=indices, scores=scores, advantages=advantages[:1], returns=returns, loss_ori = loss, init_scores=init_scores)
                replay_buffer.append(experience.to("cpu"))

             
                all_losses.append(-torch.mean(all_returns))
                torch.cuda.empty_cache()
        if rank==0:
            if not all_losses:
                print("No valid samples in this step, skipping.")
                continue
            step_loss = torch.mean(torch.stack(all_losses, dim=0))
            step_loss_value = step_loss.item()
            print(step_loss)
            
            
        experience_sampler = DataLoader(replay_buffer, batch_size=1, shuffle=True, drop_last=True, collate_fn=custom_collate)
        for param in model.parameters():
            param.requires_grad = False
        
        for l in range(num_layers):
           
            for param in model.module.model.layers[l].self_attn.judge_model.parameters():
                param.requires_grad = True
            for param in model.module.model.layers[l].self_attn.reference.parameters():
                param.requires_grad = False
        for step_epoch in range(epochs_per_step):
            all_entropys = []
            all_entropys_init = []
            all_entropys_new = []
            all_kl = []
            for step, exps in enumerate(experience_sampler):
                for exp in exps:
                    exp = exp.to(device)
                    input_ids, indices, scores, advantages, init_scores = exp.input_ids, exp.indices, exp.scores, exp.advantages, exp.init_scores
                    init_model_chunks(model.module, input_ids.shape[-1])
                    new_output = model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_attentions=True,
                        cache_drop=indices,
                        output_logits=False,
                        disable_eviction=disable_eviction,
                    )
                    _, new_scores = extract_attention_cache(new_output.attentions)
                    if new_scores is None:
                        continue
                    new_scores = new_scores.reshape(indices.shape[0], indices.shape[1], group_size, indices.shape[3], -1)
                    loss_total = 0.0

                    for l in range(num_layers):
                        advantages_new = advantages.clone().unsqueeze(-1).unsqueeze(-1)
                        log_scores, entropy = sequence_log_probs_from_logits(scores[l], indices[l])
                        
                        log_scores_new, entropy_new = sequence_log_probs_from_logits(new_scores[l], indices[l])
                       
                        log_scores_init, entropy_init = sequence_log_probs_from_logits(init_scores[l], indices[l])
                        all_entropys.append(torch.mean(entropy))
                        all_entropys_init.append(torch.mean(entropy_init))
                        all_entropys_new.append(torch.mean(entropy_new))
                        
                        loss, kl = objective(log_probs=log_scores_new, log_probs_past = log_scores, log_probs_init = log_scores_init, advantages=advantages_new, indices = indices[l])
                        all_kl.append(kl)
                        loss_total += loss/32
                    loss_total.backward()
                    
            clip_grad_norm_(to_optim, max_norm=max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not all_kl:
            continue
        mean_kl = dist_mean_scalar(torch.mean(torch.stack(all_kl, dim=0)))
        scheduler.step()
        if rank == 0 and swanlab_run is not None:
            log_payload = {
                "train/loss": step_loss_value,
                "train/kl_original": mean_kl.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": k + 1,
                "train/disable_eviction": int(disable_eviction),
            }
            swanlab.log(log_payload, step=k + 1)
            
        if checkpoint_dir is not None and checkpoint_interval and (k + 1) % checkpoint_interval == 0 and rank == 0:
            model.module.save_pretrained(str(checkpoint_dir / f"step_{k}"))
    if checkpoint_dir is not None and rank == 0:
        model.module.save_pretrained(str(checkpoint_dir / "step_final"))
    if swanlab_run is not None:
        swanlab.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()
