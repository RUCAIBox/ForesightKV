from torch import nn
import torch
from torch.nn import functional as F
def z_score_normalize(tensor, dim=0, eps=1e-8):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + eps)

import math

def topk_prob_sample(input_tensor, top_k=1024, sample_k=256):
   
    topk_vals, topk_indices = torch.topk(input_tensor, top_k, dim=-1)  # shape [..., 8, 512]

    probs = F.softmax(topk_vals, dim=-1)  
    orig_shape = probs.shape[:-1]  # [..., 8]
    probs_flat = probs.reshape(-1, top_k)  # [B, 512]
    
    sampled_local_indices = torch.multinomial(probs_flat, sample_k, replacement=False)  # [B, 256]
    sampled_local_indices = sampled_local_indices.view(*orig_shape, sample_k)  # [..., 8, 256]

    topk_indices_flat = topk_indices.view(-1, top_k)  # [B, 512]
    final_indices = torch.gather(topk_indices_flat, 1, sampled_local_indices.view(-1, sample_k))  # [B, 256]
    final_indices = final_indices.view(*orig_shape, sample_k)  # [..., 8, 256]

    return final_indices
def cache_process_rl(self, q, k, v, cache_drop=None, topk=False, reference=False):
    if reference and cache_drop is not None:
        k, v, indices, score=  self.h2ocluster.update_kv(self.reference, k,q,v, cache_drop, topk)
    else:
        k, v, indices, score=  self.h2ocluster.update_kv(self.judge_model, k,q,v, cache_drop, topk)
    return k, v, indices, score

def init_cache_rl(self,  q, k, v, layer_idx, reference=False):
    self.h2ocluster = KVCluster( window_size = self.step_size, 
     max_capacity_prompt = self.initial_chunk, num_key_value_groups = self.num_key_value_groups, layer_idx = layer_idx)
    self.h2ocluster.init_kv(k,q)




def cache_process_distillation(self, q, k, v, cache_drop=None):
    k, v, indices, score=  self.h2ocluster.update_kv(self.judge_model, k,q,v, cache_drop)
    return k, v, indices, score

def init_cache_distillation(self,  q, k, v):
    self.h2ocluster = KVClusterDistillation( window_size = self.step_size, 
     max_capacity_prompt = self.initial_chunk, num_key_value_groups = self.num_key_value_groups)
    self.h2ocluster.init_kv(k,q)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class KVCluster():
    def __init__(self,  window_size = 256, max_capacity_prompt = 1024, num_key_value_groups = 4, layer_idx=0):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.num_key_value_groups = num_key_value_groups
        assert self.max_capacity_prompt - self.window_size > 0
        self.layer_idx = layer_idx
    def init_kv(self, key_states, query_states):

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
       
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -q_len:, -q_len:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights.sum(dim = -2).reshape(bsz, num_heads//self.num_key_value_groups, self.num_key_value_groups, self.max_capacity_prompt).transpose(-2, -1)

        self.cumsum_attn = attn_weights_sum
        self.cumsum_attn_decay  = attn_weights_sum.clone()
    def update_kv(self, judge_model, key_states, query_states, value_states, cache_drop=None, topk = True):
        
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        key_states_p = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states_p.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -q_len:, -q_len:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype).reshape(bsz, num_heads//self.num_key_value_groups, self.num_key_value_groups, q_len, self.max_capacity_prompt+q_len)
        attn_weights_sum = attn_weights[:, :, :].sum(dim = -2).transpose(-1,-2)
        
        attn_weights_max = attn_weights[:, :, :].max(dim = -2)[0].transpose(-1,-2)

        attn_weights_min = attn_weights[:, :, :].min(dim = -2)[0].transpose(-1,-2)

        attn_weights_sum1 = attn_weights[:, :, -8:].sum(dim = -2).transpose(-1,-2)

        attn_weights_sum2 = attn_weights[:, :, -16:].sum(dim = -2).transpose(-1,-2)

        attn_weights_sum3 = attn_weights[:, :, -32:].sum(dim = -2).transpose(-1,-2)
       
        self.cumsum_attn += attn_weights_sum[...,: self.max_capacity_prompt,:]
        self.cumsum_attn_decay = self.cumsum_attn_decay*0.9 + attn_weights_sum[...,: self.max_capacity_prompt,:]
       
        attn_info = torch.cat([attn_weights_max[...,: self.max_capacity_prompt,:].detach(),
                attn_weights_min[...,: self.max_capacity_prompt,:].detach(),
                attn_weights_sum[...,: self.max_capacity_prompt,:].detach(),
                attn_weights_sum2[...,: self.max_capacity_prompt,:].detach(),
                attn_weights_sum1[...,: self.max_capacity_prompt,:].detach(),
                attn_weights_sum3[...,: self.max_capacity_prompt,:].detach(),
                self.cumsum_attn[...,: self.max_capacity_prompt,:].detach(),self.cumsum_attn_decay[...,: self.max_capacity_prompt,:].detach(),], dim=-1)
        if cache_drop is not None:
            importance_scores=judge_model(
                key_states[...,: self.max_capacity_prompt,:].detach(),
                value_states[...,: self.max_capacity_prompt,:].detach(),
                attn_info,
            ).transpose(1,2).reshape(bsz*num_heads//self.num_key_value_groups, self.max_capacity_prompt)
        else:
            importance_scores = judge_model(
                key_states[...,: self.max_capacity_prompt,:].detach(),
                value_states[...,: self.max_capacity_prompt,:].detach(),
                attn_info,
            ).transpose(1,2).reshape(bsz*num_heads//self.num_key_value_groups, self.max_capacity_prompt)
     
        if cache_drop is not None:
            drop_indices = cache_drop.reshape(-1, self.window_size)
        elif not topk:
            drop_indices  = topk_prob_sample(importance_scores, top_k=self.window_size*2, sample_k=self.window_size)
        else:
            drop_indices = torch.topk(importance_scores,k=self.window_size, dim=-1).indices
        all_indices = torch.arange(self.max_capacity_prompt, device=attn_weights_max.device).unsqueeze(0).expand(drop_indices.shape[0], -1)
        
        keep_mask = torch.ones(drop_indices.shape[0], self.max_capacity_prompt, dtype=torch.bool, device=all_indices.device)
       
        keep_mask.scatter_(1, drop_indices, False)

        keep_indices = torch.masked_select(all_indices, keep_mask).reshape(-1, self.max_capacity_prompt-self.window_size)
       
        indices = keep_indices.reshape(bsz, num_heads//self.num_key_value_groups, self.max_capacity_prompt-self.window_size)
        self.cumsum_attn  = self.cumsum_attn.gather(dim=2, index = indices.unsqueeze(-1).repeat(1,1,1, self.num_key_value_groups))
        self.cumsum_attn = torch.cat([self.cumsum_attn, attn_weights_sum[...,self.max_capacity_prompt:,:]],dim=-2)


        self.cumsum_attn_decay  = self.cumsum_attn_decay.gather(dim=2, index = indices.unsqueeze(-1).repeat(1,1,1, self.num_key_value_groups))
        self.cumsum_attn_decay = torch.cat([self.cumsum_attn_decay, attn_weights_sum[...,self.max_capacity_prompt:,:]],dim=-2)
        k_past_compress = key_states[:, :, :self.max_capacity_prompt, :].gather(dim = 2, index = indices.unsqueeze(-1).repeat(1,1,1, key_states.size(-1)))
        v_past_compress = value_states[:, :, :self.max_capacity_prompt, :].gather(dim = 2, index = indices.unsqueeze(-1).repeat(1,1,1, key_states.size(-1)))
        
        k_cur = key_states[:, :, self.max_capacity_prompt:, :]
        v_cur = value_states[:, :, self.max_capacity_prompt:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        indices2 = drop_indices.reshape(bsz, num_heads//self.num_key_value_groups, self.window_size)
        return key_states, value_states, indices2, importance_scores




class KVClusterDistillation():
    def __init__(self,  window_size = 256, max_capacity_prompt = 1024, num_key_value_groups = 4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.num_key_value_groups = num_key_value_groups
        assert self.max_capacity_prompt - self.window_size > 0
    @torch.compile
    def init_kv(self, key_states, query_states):

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -q_len:, -q_len:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights.sum(dim = -2).reshape(bsz, num_heads//self.num_key_value_groups, self.num_key_value_groups, self.max_capacity_prompt).transpose(-2, -1)

        self.cumsum_attn = attn_weights_sum
        self.cumsum_attn_decay  = attn_weights_sum.clone()
    @torch.compile
    def update_kv(self, judge_model, key_states, query_states, value_states, cache_drop=None):
        
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        key_states_p = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states_p.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -q_len:, -q_len:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, :].sum(dim = -2).reshape(bsz, num_heads//self.num_key_value_groups, self.num_key_value_groups, self.max_capacity_prompt+q_len).transpose(-2, -1)
        attn_weights_max = attn_weights[:, :, :].max(dim = -2)[0].reshape(bsz, num_heads//self.num_key_value_groups, self.num_key_value_groups, self.max_capacity_prompt+q_len).transpose(-2, -1)
        attn_weights_min = attn_weights[:, :, :].min(dim = -2)[0].reshape(bsz, num_heads//self.num_key_value_groups, self.num_key_value_groups, self.max_capacity_prompt+q_len).transpose(-2, -1)
        self.cumsum_attn += attn_weights_sum[...,: self.max_capacity_prompt,:]
        self.cumsum_attn_decay = self.cumsum_attn_decay*0.9 + attn_weights_sum[...,: self.max_capacity_prompt,:]
        importance_scores = judge_model(
            key_states[...,: self.max_capacity_prompt,:].detach(),
            value_states[...,: self.max_capacity_prompt,:].detach(),
            attn_weights_max[...,: self.max_capacity_prompt,:].detach(),
            attn_weights_min[...,: self.max_capacity_prompt,:].detach(),
            attn_weights_sum[...,: self.max_capacity_prompt,:].detach(),
            self.cumsum_attn[...,: self.max_capacity_prompt,:].detach(),
            self.cumsum_attn_decay[...,: self.max_capacity_prompt,:].detach(),
        ).transpose(1,2).reshape(bsz*num_heads//self.num_key_value_groups, self.max_capacity_prompt)
        
        attn_x = torch.sum(self.cumsum_attn_decay, dim=-1)
      
        top_info = attn_x
       
        if cache_drop is not None:
            drop_indices = cache_drop.reshape(-1, self.window_size)
        else:
            drop_indices  = torch.topk(-top_info, dim=-1, k=self.window_size).indices.reshape(-1, self.window_size)  
        target = z_score_normalize(-torch.log(top_info), dim=-1)
        all_indices = torch.arange(self.max_capacity_prompt, device=importance_scores.device).unsqueeze(0).expand(drop_indices.shape[0], -1)

        keep_mask = torch.ones(drop_indices.shape[0], self.max_capacity_prompt, dtype=torch.bool, device=all_indices.device)
       
        keep_mask.scatter_(1, drop_indices, False)
        
        keep_indices = torch.masked_select(all_indices, keep_mask).reshape(-1, self.max_capacity_prompt-self.window_size)
       
        
        indices = keep_indices.reshape(bsz, num_heads//self.num_key_value_groups, self.max_capacity_prompt-self.window_size)
        self.cumsum_attn  = self.cumsum_attn.gather(dim=2, index = indices.unsqueeze(-1).repeat(1,1,1, self.num_key_value_groups))
        self.cumsum_attn = torch.cat([self.cumsum_attn, attn_weights_sum[...,self.max_capacity_prompt:,:]],dim=-2)


        self.cumsum_attn_decay  = self.cumsum_attn_decay.gather(dim=2, index = indices.unsqueeze(-1).repeat(1,1,1, self.num_key_value_groups))
        self.cumsum_attn_decay = torch.cat([self.cumsum_attn_decay, attn_weights_sum[...,self.max_capacity_prompt:,:]],dim=-2)
        k_past_compress = key_states[:, :, :self.max_capacity_prompt, :].gather(dim = 2, index = indices.unsqueeze(-1).repeat(1,1,1, key_states.size(-1)))
        v_past_compress = value_states[:, :, :self.max_capacity_prompt, :].gather(dim = 2, index = indices.unsqueeze(-1).repeat(1,1,1, key_states.size(-1)))
        
        k_cur = key_states[:, :, self.max_capacity_prompt:, :]
        v_cur = value_states[:, :, self.max_capacity_prompt:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        indices2 = drop_indices.reshape(bsz, num_heads//self.num_key_value_groups, self.window_size)

        return key_states, value_states, target, importance_scores


def transform_matrix(x):
   
    n = x.shape[-1] 
    k = min(512, n) 
    if k == 0:
        return torch.full_like(x, -0.5)
    
    _, topk_indices = torch.topk(x, k=k, dim=-1)  
    out = torch.full_like(x, -0.5)
    
    zeros_src = torch.zeros_like(topk_indices, dtype=x.dtype)
    out.scatter_(-1, topk_indices, zeros_src)
    
    k0 = min(256, k) 
    if k0 > 0:
        top256_indices = topk_indices[..., :k0]
        twos_src = torch.full(top256_indices.shape, 0.5, dtype=x.dtype, device=x.device)
        out.scatter_(-1, top256_indices, twos_src)
    
    return out

def cal_similarity(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    k = key_states[0]
    num_heads = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # shape: [num_heads, seq_len, seq_len]
    similarity_mask = similarity_cos > threshold

    seq_len = similarity_mask.size(-1)
    k = int(seq_len * retain_ratio)

    indices = torch.where(
        similarity_mask,
        torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    # find the last True index in each row
    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]

    # find the first True index in each row
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]

    # keep the last_percent% elements
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]

    # keep the first_percent% elements
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

    # create indices for zeroing
    batch_idx = (
        torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
    )
    seq_idx = torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)

    # zero the specified positions in similarity_cos
    similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

    return similarity_cos.mean(dim=1).softmax(dim=-1)