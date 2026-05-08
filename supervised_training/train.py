from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import math
import torch
import torch.optim as optim
import os
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from torch import nn
from cache import init_cache_rl, cache_process_rl
from qwen3 import Qwen3ForCausalLM, Qwen3Attention

# Apply custom cache modifications
Qwen3Attention.init_cache = init_cache_rl
Qwen3Attention.cache_process = cache_process_rl

def sequence_log_probs_from_logits(
    logits: torch.tensor,
    indices: torch.tensor,
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return torch.gather(log_prob, dim=-1, index=indices)

def load_model(model_name_or_path, trust_remote_code=False, bf16=True):
    """Loads the models and tokenizer for single-GPU execution."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model to be trained
    model = Qwen3ForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    )
    
    # Reference model
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    )
    return model, tokenizer, model2

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Creates a learning rate scheduler with a cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def custom_collate(batch):
    """Custom collate function to handle list of dictionaries."""
    return list(batch)

def init_rng(seed: int) -> torch.Generator:
    """Initializes random number generators for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator(device="cuda").manual_seed(seed)
class PairwiseRankHingeLoss(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, model_scores_batch, oracle_scores_batch):
        
        batch_losses = []
        for model_scores, oracle_scores in zip(model_scores_batch, oracle_scores_batch):
        
            with torch.no_grad():
                oracle_diffs = oracle_scores.unsqueeze(1) - oracle_scores.unsqueeze(0)
               
                pairwise_mask = (oracle_diffs > 0)

                if not pairwise_mask.any():
                    continue

            model_diffs = model_scores.unsqueeze(1) - model_scores.unsqueeze(0)

           
            loss_matrix = torch.clamp(self.margin - model_diffs, min=0.0)
          
            relevant_losses = loss_matrix[pairwise_mask]

            batch_losses.append(relevant_losses.mean())
        if not batch_losses:
            return torch.tensor(0.0, device=model_scores_batch.device, requires_grad=True)

        final_loss = torch.mean(torch.stack(batch_losses))
        
        return final_loss
import argparse
def main():
    seed = 422
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3-4B")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints_qwen3_4b')
    parser.add_argument("--checkpoint_interval", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--total_training_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    # --- Configuration ---
    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    checkpoint_interval = args.checkpoint_interval
    train_batch_size = args.train_batch_size
    lr = args.lr
    total_training_steps = args.total_training_steps
    warmup_steps = args.warmup_steps
    max_norm = args.max_norm
    
    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = init_rng(seed)
    model, tokenizer, model2 = load_model(model_name)
    
    model.to('cuda:0')
    model2.to('cuda:1')
    
    to_optim = [p for n, p in model.named_parameters() if "judge_model" in n]
    optimizer = optim.Adam(to_optim, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # --- Data Loading ---
    dataset = load_from_disk(args.dataset)
    prompt_loader = DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        sampler=RandomSampler(dataset), 
        drop_last=True, 
        collate_fn=custom_collate
    )
    prompt_iterator = iter(prompt_loader)
    
    print(f"Starting training on {device}. Total steps: {total_training_steps}")

    for k in tqdm(range(total_training_steps), desc="Training Steps"):
        try:
            prompt_batch = next(prompt_iterator)
        except StopIteration:
            prompt_iterator = iter(prompt_loader) # Re-initialize iterator
            prompt_batch = next(prompt_iterator)

        # Set requires_grad for trainable parameters
        for param in model.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "judge_model" in name:
                param.requires_grad = True
        
        torch.cuda.empty_cache()
        
        batch_losses = []
        batch_losses2 = []
        batch_accuracies = []

        for sample in prompt_batch:
            input_ids = torch.tensor([sample["input_ids"]], device=device)
            print(f"Processing sample with sequence length: {input_ids.shape[1]}")
            all_scores = []
            all_indices = []
            all_step_scores = []
            # --- Generate Targets with Reference Model ---
            with torch.no_grad():
                output2 = model2(input_ids.to(model2.device), output_hidden_states=True)
                for layer in range(36):
                    query = output2.attentions[layer][0]
                    key = output2.attentions[layer][1].unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(*query.shape)
                    key = key.transpose(2, 3)
                    attn_mask = (1 - torch.tril(torch.ones(key.shape[-1], key.shape[-1]))).to(key.device).to(key.dtype) * torch.finfo(key.dtype).min
                    if input_ids.shape[1]<12000:
                        if input_ids.shape[1] < 6000:
                            attn_scores = torch.matmul(query, key) * (query.shape[-1] ** -0.5)
                            attn_scores = F.softmax(attn_scores + attn_mask, dim=-1, dtype=torch.float32).to(query.dtype)
                        else: # Memory-efficient calculation for long sequences
                            tmp = []
                            for j in range(8):
                                xm = torch.matmul(query[:, 4*j:4*j+4], key[:, 4*j:4*j+4]) * (query.shape[-1]**-0.5)
                                xm = F.softmax(xm + attn_mask, dim=-1, dtype=torch.float32).to(query.dtype)
                                tmp.append(xm)
                            attn_scores = torch.cat(tmp, dim=1)
                            del tmp
                   
                        pad_len = (256 - attn_scores.shape[-2] % 256) % 256
                        if pad_len > 0:
                            to_add = torch.zeros((*attn_scores.shape[:-2], pad_len, attn_scores.shape[-1]), device=attn_scores.device, dtype=attn_scores.dtype)
                            data_new = torch.cat([attn_scores, to_add], dim=-2)
                        else:
                            data_new = attn_scores
                        
                        data_new = data_new.reshape(*data_new.shape[:-2], -1, 256, data_new.shape[-1])
                        data_reshape = data_new[0][:, 5:].transpose(2, 3).transpose(1, 2)
                        data_reshape = data_reshape.reshape(8, 4, *data_reshape.shape[1:])
                        data_reshape = torch.mean(torch.mean(data_reshape, dim=1), dim=-1)
                        del attn_scores, data_new
                        all_scores.append(data_reshape)
                        indices_per_layer = []
                        scores_per_layer = []
                        for i in range(data_reshape.shape[-1]):
                            bsz, length, _ = data_reshape.shape
                            if length <= 1024 + 256: break
                            
                            data2 = torch.max(data_reshape[:, :1024, i:], dim=-1).values
                            topk_idx = torch.topk(-data2, dim=-1, k=256).indices
                            indices_per_layer.append(topk_idx)
                            scores_per_layer.append(-data2)
                            
                            mask = torch.ones((bsz, length), dtype=torch.bool, device=data_reshape.device)
                            batch_idx = torch.arange(bsz).unsqueeze(1).expand(-1, 256).to(data_reshape.device)
                            mask[batch_idx, topk_idx] = False
                            
                            data_reshape_list = [data_reshape[b][mask[b]] for b in range(bsz)]
                            max_len = max(x.shape[0] for x in data_reshape_list)
                            padded = torch.zeros((bsz, max_len, data_reshape_list[0].shape[1]), device=data_reshape_list[0].device)
                            for b in range(bsz):
                                padded[b, :data_reshape_list[b].shape[0]] = data_reshape_list[b]
                            data_reshape = padded
                        all_indices.append(torch.stack(indices_per_layer, dim=0))
                        all_step_scores.append(torch.stack(scores_per_layer, dim=0))
                    else:
                        indices_per_layer_large = []
                        scores_per_layer_large = []
                        for j in range(8):
                            data_new = []
                            for p in range(4):
                                xm = torch.matmul(query[:, 4*j+p:4*j+p+1], key[:, 4*j+p:4*j+p+1]) * (query.shape[-1]**-0.5)
                                xm = F.softmax(xm + attn_mask, dim=-1, dtype=torch.float32).to(query.dtype)
                                # tmp.append(xm)
                                pad_len = (256 - xm.shape[-2] % 256) % 256
                                
                                if pad_len > 0:
                                    to_add = torch.zeros((*xm.shape[:-2], pad_len, xm.shape[-1]), device=xm.device, dtype=xm.dtype)
                                    data_new.append(torch.cat([xm, to_add], dim=-2))
                                else:
                                    data_new.append(xm)
                            data_new = torch.cat(data_new, dim=1)
                            
                            data_new = data_new.reshape(*data_new.shape[:-2], -1, 256, data_new.shape[-1])
                            data_reshape = data_new[0][:, 5:].transpose(2, 3).transpose(1, 2)
                            data_reshape = data_reshape.reshape(1, 4, *data_reshape.shape[1:])
                            data_reshape = torch.mean(torch.mean(data_reshape, dim=1), dim=-1)
                            del xm, data_new
                            all_scores.append(data_reshape)
                            indices_per_layer = []
                            scores_per_layer = []
                            for i in range(data_reshape.shape[-1]):
                                bsz, length, _ = data_reshape.shape
                                if length <= 1024 + 256: break
                                
                                data2 = torch.max(data_reshape[:, :1024, i:], dim=-1).values
                                topk_idx = torch.topk(-data2, dim=-1, k=256).indices
                                indices_per_layer.append(topk_idx)
                                scores_per_layer.append(-data2)
                                
                                mask = torch.ones((bsz, length), dtype=torch.bool, device=data_reshape.device)
                                batch_idx = torch.arange(bsz).unsqueeze(1).expand(-1, 256).to(data_reshape.device)
                                mask[batch_idx, topk_idx] = False
                                
                                data_reshape_list = [data_reshape[b][mask[b]] for b in range(bsz)]
                                max_len = max(x.shape[0] for x in data_reshape_list)
                                padded = torch.zeros((bsz, max_len, data_reshape_list[0].shape[1]), device=data_reshape_list[0].device)
                                for b in range(bsz):
                                    padded[b, :data_reshape_list[b].shape[0]] = data_reshape_list[b]
                                data_reshape = padded
                            indices_per_layer_large.append(torch.stack(indices_per_layer, dim=0))
                            scores_per_layer_large.append(torch.stack(scores_per_layer, dim=0))
                        all_indices.append(torch.cat(indices_per_layer_large, dim=1))
                        all_step_scores.append(torch.cat(scores_per_layer_large, dim=1))

            indices = torch.stack(all_indices, dim=0).to(model.device) # [layer, iter, bsz, k]
            step_scores = torch.stack(all_step_scores, dim=0).to(model.device)
            output = model(input_ids=input_ids.to(model.device), use_cache=False, cache_drop=indices.unsqueeze(2), output_attentions=True)
    
            scores = torch.stack([torch.stack([a[1] for a in index], dim=0) for index in output.attentions], dim=0)[:, :indices.shape[1]]
            target = torch.zeros_like(scores)
            target.scatter_(3, indices, 1.0)
           
            # if input_ids.shape[1]>15000:
            scores_reshaped = scores.reshape(scores.shape[0],-1, scores.shape[-1])
            step_scores_reshaped = step_scores.reshape(scores.shape[0], -1, target.shape[-1]).detach()
            all_loss = 0
            for layer in range(scores.shape[0]):
                loss_function = PairwiseRankHingeLoss(margin=0.01)
                loss = loss_function(scores_reshaped[layer], step_scores_reshaped[layer])
                loss.backward(retain_graph=True)
                all_loss+=loss.item()
            batch_losses.append(all_loss)
          
            with torch.no_grad():
                _, topk_indices = torch.topk(scores, 256, dim=-1)
                preds_topk = torch.zeros_like(scores)
                preds_topk.scatter_(-1, topk_indices, 1)
                accuracy = torch.mean((preds_topk * target).sum(dim=-1) / 256, dim=1)
                batch_accuracies.append(accuracy)

        # --- Optimizer Step and Logging ---
        clip_grad_norm_(to_optim, max_norm=max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        avg_loss = torch.mean(torch.tensor(batch_losses))
        avg_loss2 = torch.mean(torch.tensor(batch_losses2))
        avg_accuracy = torch.mean(torch.stack(batch_accuracies, dim=0))
        
        
        # --- Checkpointing ---
        if checkpoint_path and checkpoint_interval and (k + 1) % checkpoint_interval == 0:
            step_checkpoint_path = os.path.join(checkpoint_path, f"step_{k}")
            step_checkpoint_path_rm = os.path.join(checkpoint_path, f"step_{k-5*checkpoint_interval}")
            model.save_pretrained(step_checkpoint_path)
            print(f"Saved checkpoint to {step_checkpoint_path}")
            


    # --- Final Save ---
    if checkpoint_path:
        final_checkpoint_path = os.path.join(checkpoint_path, f"step_{k}_final")
        model.save_pretrained(final_checkpoint_path)
        print(f"Saved final model to {final_checkpoint_path}")

if __name__ == "__main__":
    main()