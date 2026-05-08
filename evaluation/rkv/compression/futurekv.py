import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import cal_similarity, compute_attention_scores


def topk_prob_sample(input_tensor, top_k=1024, sample_k=256):
  
    topk_vals, topk_indices = torch.topk(input_tensor, top_k, dim=-1) 
    probs = F.softmax(topk_vals, dim=-1)  # shape [..., 8, 512]

    orig_shape = probs.shape[:-1]  # [..., 8]
    probs_flat = probs.reshape(-1, top_k)  # [B, 512]
    
    sampled_local_indices = torch.multinomial(probs_flat, sample_k, replacement=False)  # [B, 256]
    sampled_local_indices = sampled_local_indices.view(*orig_shape, sample_k)  # [..., 8, 256]

    topk_indices_flat = topk_indices.view(-1, top_k)  # [B, 512]
    final_indices = torch.gather(topk_indices_flat, 1, sampled_local_indices.view(-1, sample_k))  # [B, 256]
    final_indices = final_indices.view(*orig_shape, sample_k)  # [..., 8, 256]

    return final_indices


def compute_attention_scores_plus(query_states, key_states, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
    else:
        # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )

        # shape: [batch_size, kv_heads, 1, kv_len, head_dim]
        key_states = key_states.unsqueeze(2)

        # shape: [batch_size, kv_heads, query_group_size, q_len, kv_len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)
    mask = torch.full((attn_weights.shape[-2], attn_weights.shape[-2]), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    attention_mask = mask[None, None, None, :, :]
    attn_weights[:, :, : , :, -attn_weights.shape[-2]: ]+=attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    ).to(query_states.dtype)
      
    return attn_weights

class FutureKV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        record_kept_token_indices=False,
        judge_model=None,
        **kwargs,
    ):
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.attn_acc = None
        self.attn_acc_decay = None
        self.length = 128
        self.judge_model = judge_model
      

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            self.attn_acc = None
            self.attn_acc_decay = None
            return key_states, value_states
        else:
            if self.attn_acc is not None:
                query_states = query_states[:, :, -self.length:, :]
            batch_size, q_heads, q_len, head_dim = query_states.shape
            kv_heads = key_states.shape[1]
            query_group_size = q_heads // kv_heads
            attn_weights = compute_attention_scores_plus(query_states, key_states)

            attn_weights_sum = attn_weights[:, :, :].sum(dim = -2).transpose(-1,-2)
            attn_weights_max = attn_weights[:, :, -self.length:].max(dim = -2)[0].transpose(-1,-2)
        
            attn_weights_min = attn_weights[:, :, -self.length:].min(dim = -2)[0].transpose(-1,-2)

            attn_weights_sum1 = attn_weights[:, :, -8:].sum(dim = -2).transpose(-1,-2)

            attn_weights_sum2 = attn_weights[:, :, -16:].sum(dim = -2).transpose(-1,-2)

            attn_weights_sum3 = attn_weights[:, :, -32:].sum(dim = -2).transpose(-1,-2)
            if self.attn_acc is None:
                self.attn_acc = attn_weights_sum
                self.attn_acc_decay = attn_weights_sum
            else:
                self.attn_acc += attn_weights_sum[:,:,:self.attn_acc.shape[2]]
                self.attn_acc = torch.cat([self.attn_acc, attn_weights_sum[:,:,self.attn_acc.shape[2]:]], dim=2)


                self.attn_acc_decay = attn_weights_sum[:,:,:self.attn_acc_decay.shape[2]] + 0.9* self.attn_acc_decay
                self.attn_acc_decay = torch.cat([self.attn_acc, attn_weights_sum[:,:,self.attn_acc.shape[2]:]], dim=2)
            attn_info = torch.cat([attn_weights_max[...,: -self.length,:].detach(),
                attn_weights_min[...,: -self.length,:].detach(),
                attn_weights_sum[...,: -self.length,:].detach(),
                attn_weights_sum2[...,: -self.length,:].detach(),
                attn_weights_sum1[...,: -self.length,:].detach(),
                attn_weights_sum3[...,: -self.length,:].detach(),
                self.attn_acc[...,: -self.length,:].detach(),self.attn_acc_decay[...,: -self.length,:].detach(),], dim=-1)
            importance_scores=self.judge_model(
                key_states[...,: -self.length,:].detach(),
                value_states[...,: -self.length,:].detach(),
                attn_info,
            ).transpose(1,2).squeeze(-1)
           
            
            drop_budget = importance_scores.shape[-1]+self.length-self.budget
          
            if drop_budget<=0:
                return key_states, value_states
            drop_indices = topk_prob_sample(importance_scores, 2*drop_budget, drop_budget)
            all_indices = torch.arange(importance_scores.shape[-1], device=drop_indices.device).unsqueeze(0).unsqueeze(0).expand(drop_indices.shape[0], drop_indices.shape[1], -1)
            keep_mask = torch.ones((drop_indices.shape[0], drop_indices.shape[1], importance_scores.shape[-1]), dtype=torch.bool, device=all_indices.device)
            
            keep_mask.scatter_(2, drop_indices, False)

            keep_indices = torch.masked_select(all_indices, keep_mask).reshape(drop_indices.shape[0],drop_indices.shape[1],-1)
           
            indices = keep_indices
          
            self.attn_acc = torch.cat([self.attn_acc[:, :, :-self.length].gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1, query_group_size)), self.attn_acc[:, :, -self.length:]], dim=2)


            self.attn_acc_decay = torch.cat([self.attn_acc_decay[:, :, :-self.length].gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1, query_group_size)), self.attn_acc_decay[:, :, -self.length:]], dim=2)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, : -self.length, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.length, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.length :, :]
            v_cur = value_states[:, :, -self.length :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states
