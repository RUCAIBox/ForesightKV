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
from qwen3 import Qwen3ForCausalLM, Qwen3Attention

Qwen3Attention.init_cache = init_cache_rl
Qwen3Attention.cache_process = cache_process_rl
def sequence_log_probs_from_logits(
    logits: torch.tensor,
    indices: torch.tensor, 
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(log_prob * torch.exp(log_prob), dim=-1)
    return torch.gather(log_prob, dim=-1, index = indices ), entropy
def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    dist.destroy_process_group()

def load_model(model_name_or_path, trust_remote_code=False, bf16=True, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3ForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    
   
    return model, tokenizer, None

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return float(current_step+1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def custom_collate(batch):
    return list(batch)

@torch.no_grad()
def rollout(model, tokenizer, input_ids, num_rollouts, generator):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    input_ids = torch.tensor(input_ids, device=model.device).repeat(num_rollouts, 1)
    for layer in model.model.layers:
        if input_ids.shape[-1]>12000:
            window = 512
            budget = 2048
        else:
            window=256
            budget = 1024
        layer.self_attn.init_chunk(budget, window)
    output = model(input_ids=input_ids, use_cache=True, labels=input_ids, output_attentions=True, output_logits=True, cache_drop=None)

    a = input_ids[:, 1:].reshape(-1)
    loss = criterion(output.logits.float()[:, :-1].reshape(a.shape[0], -1), a)
    
    indices = torch.stack([torch.stack([a[0] for a in index], dim=0) for index in output.attentions], dim=0)
    scores = torch.stack([torch.stack([a[1] for a in index], dim=0) for index in output.attentions], dim=0).reshape(indices.shape[0], indices.shape[1], num_rollouts, indices.shape[3], -1)

    return input_ids, indices, scores, output.loss, loss

def init_rng(seed: int) -> torch.Generator:
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return torch.Generator(device="cuda").manual_seed(seed + rank)

def group_advantages(returns, rank,  eps = 1e-8) -> torch.Tensor:
    all_returns = [torch.zeros_like(returns) for _ in range(dist.get_world_size())]
    dist.barrier()
    dist.all_gather(all_returns, returns)
    all_returns = torch.cat(all_returns, dim=0)
    
    mean = all_returns.mean(dim=0, keepdim=True)
    std = all_returns.std(dim=0, keepdim=True) + eps
    return (returns - mean) / std, all_returns, (all_returns - mean) / std

def loss_compare(loss):
    all_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
    dist.barrier()
    dist.all_gather(all_loss, loss)
    return torch.mean(torch.cat(all_loss,dim=0))
def dist_broadcast_data(data, src=0):
    data = [json.dumps(data) if dist.get_rank() == src else ""]
    dist.broadcast_object_list(data, src=src)
    return data
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
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_name", type=str, default=None)
    args = parser.parse_args()
    setup_ddp()
    rank = dist.get_rank()
    seed = 42
   
    
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    checkpoint_interval = 20
    train_batch_size = 1
    lr = 3e-4
    total_training_steps = 250  # total steps = outer loop * inner loop
    warmup_steps = 10  # 10% warmup
    
    kl_weight = 0.05
    clip_eps = 0.2
    group_size = 1
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0
    device = torch.device("cuda", rank % torch.cuda.device_count())
    generator = init_rng(seed)
    model, tokenizer, model2 = load_model(model_name)
    model.to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    for l in range(36):
        with torch.no_grad():
            model.module.model.layers[l].self_attn.reference.load_state_dict(model.module.model.layers[l].self_attn.judge_model.state_dict())
            for param in model.module.model.layers[l].self_attn.reference.parameters():
                param.requires_grad = False
  
    to_optim = [p for n, p in model.named_parameters() if "judge_model" in n]
    optimizer = optim.Adam(to_optim, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    from datasets import load_from_disk
    dataset = load_from_disk(args.data_name)
    if rank==0:
        prompt_loader = iter(DataLoader(dataset, batch_size=rollouts_per_step, sampler=RandomSampler(dataset), drop_last=True, collate_fn=custom_collate))
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)
    for k in tqdm(range(500)):
        if k> 100:
            kl_weight = 0.0
        if dist.get_rank() == 0:
            prompt_batch = next(prompt_loader)
        else:
            prompt_batch = None
        # if rank==0:
            
        prompt_batch = json.loads(dist_broadcast_data(prompt_batch)[0])
        
      
        rollout_returns = []
        replay_buffer.clear()
        torch.cuda.empty_cache()
        
        all_losses = []
        with torch.no_grad():
            for sample in prompt_batch:
                input_ids = sample["input_ids"]
                
                loss  = sample['loss']
                loss2 = torch.tensor(sample['loss2']).to(device)
                input_ids, indices, scores, returns, lm_loss = rollout(model.module, tokenizer, input_ids, group_size, generator)
                
                for layer in model.module.model.layers:
                    if input_ids.shape[-1]>12000:
                        window = 512
                        budget = 2048
                    else:
                        window=256
                        budget = 1024
                    layer.self_attn.init_chunk(budget, window)
                with torch.no_grad():
                    init_output = model(input_ids=input_ids, use_cache=True, output_attentions=True, cache_drop=indices, output_logits=False, reference=True)
                    init_scores = torch.stack([torch.stack([a[1] for a in index], dim=0) for index in init_output.attentions], dim=0).reshape(indices.shape[0], indices.shape[1], group_size, indices.shape[3], -1)
                
                entropy = torch.tensor(sample['entropy']).to(device)[0, window+budget:]
                top_20_percent_idx = int(len(entropy) * 0.2)
                sorted_indices = torch.argsort(entropy, descending=True)
                top_20_indices = sorted_indices[:top_20_percent_idx]
                other_indices = sorted_indices[top_20_percent_idx:]
               
                loss_high = torch.mean(lm_loss[window+budget:][top_20_indices])
                loss_low = lm_loss[window+budget:][other_indices]
                loss_low_ori  = loss2[window+budget:][other_indices]

                new_loss = torch.mean(((loss_low_ori  - loss_low)**2)*((loss_low_ori  - loss_low)<-1.5).float())
               
                advantages, all_returns, all_advantages = group_advantages(-new_loss.unsqueeze(0), rank)
                
                experience = Experience(input_ids=input_ids, indices=indices, scores=scores, advantages=advantages[:1], returns=returns, loss_ori = loss, init_scores=init_scores)
                replay_buffer.append(experience.to("cpu"))

             
                all_losses.append(-torch.mean(all_returns))
                torch.cuda.empty_cache()
        if rank==0:
            print(torch.mean(torch.stack(all_losses, dim=0)))
            
            
        experience_sampler = DataLoader(replay_buffer, batch_size=1, shuffle=True, drop_last=True, collate_fn=custom_collate)
        for param in model.parameters():
            param.requires_grad = False
        
        for l in range(36):
           
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
                    for layer in model.module.model.layers:
                        if input_ids.shape[-1]>12000:
                            window = 512
                            budget = 2048
                        else:
                            window=256
                            budget = 1024
                        layer.self_attn.init_chunk(budget, window)
                    new_output = model(input_ids=input_ids, use_cache=True, output_attentions=True, cache_drop=indices, output_logits=False)
                    new_scores = torch.stack([torch.stack([a[1] for a in index], dim=0) for index in new_output.attentions], dim=0).reshape(indices.shape[0], indices.shape[1], group_size, indices.shape[3], -1)
                    loss_total = 0.0


                    for l in range(36):
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
                    
            clip_grad_norm_(to_optim, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
            
        if (checkpoint_path is not None and checkpoint_interval is not None and (k + 1) % checkpoint_interval == 0 and rank == 0):
            model.module.save_pretrained(os.path.join(checkpoint_path , f"step_{k}"))
    if checkpoint_path is not None and rank == 0:
        model.module.save_pretrained(os.path.join(checkpoint_path , f"step_final"))
    cleanup_ddp()

if __name__ == "__main__":
    main()