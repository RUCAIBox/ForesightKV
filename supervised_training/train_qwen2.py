import argparse
import math
import os
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_from_disk
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import cache_process_rl, init_cache_rl
from qwen2 import Qwen2Attention, Qwen2ForCausalLM


Qwen2Attention.init_cache = init_cache_rl
Qwen2Attention.cache_process = cache_process_rl

LONG_SEQUENCE_THRESHOLD = 12000
SPLIT_ATTENTION_THRESHOLD = 6000


def load_model(model_name_or_path, trust_remote_code=False, bf16=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    attn_impl = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")

    model = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    )
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    )
    return model, tokenizer, model2


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def custom_collate(batch):
    return list(batch)


def init_rng(seed: int) -> torch.Generator:
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
                pairwise_mask = oracle_diffs > 0
                if not pairwise_mask.any():
                    continue

            model_diffs = model_scores.unsqueeze(1) - model_scores.unsqueeze(0)
            loss_matrix = torch.clamp(self.margin - model_diffs, min=0.0)
            batch_losses.append(loss_matrix[pairwise_mask].mean())

        if not batch_losses:
            return torch.tensor(0.0, device=model_scores_batch.device, requires_grad=True)

        return torch.mean(torch.stack(batch_losses))


def get_qwen2_layout(config) -> tuple[int, int, int]:
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    kv_repeat = config.num_attention_heads // config.num_key_value_heads
    return num_layers, num_kv_heads, kv_repeat


def pad_query_blocks(attn_scores: torch.Tensor, window_size: int) -> torch.Tensor:
    pad_len = (window_size - attn_scores.shape[-2] % window_size) % window_size
    if pad_len <= 0:
        return attn_scores

    pad = torch.zeros(
        (*attn_scores.shape[:-2], pad_len, attn_scores.shape[-1]),
        device=attn_scores.device,
        dtype=attn_scores.dtype,
    )
    return torch.cat([attn_scores, pad], dim=-2)


def collect_eviction_schedule(data_reshape: torch.Tensor, budget: int, window_size: int):
    indices_per_layer = []
    scores_per_layer = []

    for step_idx in range(data_reshape.shape[-1]):
        bsz, length, _ = data_reshape.shape
        if length <= budget + window_size:
            break

        data2 = torch.max(data_reshape[:, :budget, step_idx:], dim=-1).values
        topk_idx = torch.topk(-data2, dim=-1, k=window_size).indices
        indices_per_layer.append(topk_idx)
        scores_per_layer.append(-data2)

        mask = torch.ones((bsz, length), dtype=torch.bool, device=data_reshape.device)
        batch_idx = torch.arange(bsz, device=data_reshape.device).unsqueeze(1).expand(-1, window_size)
        mask[batch_idx, topk_idx] = False

        data_reshape_list = [data_reshape[b][mask[b]] for b in range(bsz)]
        max_len = max(x.shape[0] for x in data_reshape_list)
        padded = torch.zeros(
            (bsz, max_len, data_reshape_list[0].shape[1]),
            device=data_reshape.device,
            dtype=data_reshape.dtype,
        )
        for batch_id in range(bsz):
            padded[batch_id, : data_reshape_list[batch_id].shape[0]] = data_reshape_list[batch_id]
        data_reshape = padded

    if not indices_per_layer:
        return None, None

    return torch.stack(indices_per_layer, dim=0), torch.stack(scores_per_layer, dim=0)


def reduce_attention_scores(
    attn_scores: torch.Tensor,
    budget: int,
    window_size: int,
    num_kv_heads: int,
    kv_repeat: int,
):
    data_new = pad_query_blocks(attn_scores, window_size)
    data_new = data_new.reshape(*data_new.shape[:-2], -1, window_size, data_new.shape[-1])
    start_block = 1 + budget // window_size
    data_reshape = data_new[0][:, start_block:].transpose(2, 3).transpose(1, 2)
    data_reshape = data_reshape.reshape(num_kv_heads, kv_repeat, *data_reshape.shape[1:])
    return torch.mean(torch.mean(data_reshape, dim=1), dim=-1)


def build_oracle_targets(
    attentions,
    seq_len: int,
    budget: int,
    window_size: int,
    num_layers: int,
    num_kv_heads: int,
    kv_repeat: int,
):
    all_indices = []
    all_step_scores = []

    if attentions is None:
        return None, None

    for layer in range(min(num_layers, len(attentions))):
        query = attentions[layer][0]
        key = attentions[layer][1].unsqueeze(2).repeat(1, 1, kv_repeat, 1, 1).reshape(*query.shape)
        key = key.transpose(2, 3)
        attn_mask = (
            1 - torch.tril(torch.ones(key.shape[-1], key.shape[-1], device=key.device, dtype=key.dtype))
        ) * torch.finfo(key.dtype).min

        if seq_len < LONG_SEQUENCE_THRESHOLD:
            if seq_len < SPLIT_ATTENTION_THRESHOLD:
                attn_scores = torch.matmul(query, key) * (query.shape[-1] ** -0.5)
                attn_scores = F.softmax(attn_scores + attn_mask, dim=-1, dtype=torch.float32).to(query.dtype)
            else:
                grouped_scores = []
                for group_idx in range(num_kv_heads):
                    start = kv_repeat * group_idx
                    end = start + kv_repeat
                    xm = torch.matmul(query[:, start:end], key[:, start:end]) * (query.shape[-1] ** -0.5)
                    xm = F.softmax(xm + attn_mask, dim=-1, dtype=torch.float32).to(query.dtype)
                    grouped_scores.append(xm)
                attn_scores = torch.cat(grouped_scores, dim=1)

            data_reshape = reduce_attention_scores(attn_scores, budget, window_size, num_kv_heads, kv_repeat)
            layer_indices, layer_scores = collect_eviction_schedule(data_reshape, budget, window_size)
        else:
            start_block = 1 + budget // window_size
            indices_per_group = []
            scores_per_group = []

            for group_idx in range(num_kv_heads):
                data_new = []
                start = kv_repeat * group_idx
                for head_offset in range(kv_repeat):
                    head_idx = start + head_offset
                    xm = torch.matmul(query[:, head_idx : head_idx + 1], key[:, head_idx : head_idx + 1])
                    xm = F.softmax(xm * (query.shape[-1] ** -0.5) + attn_mask, dim=-1, dtype=torch.float32).to(
                        query.dtype
                    )
                    data_new.append(pad_query_blocks(xm, window_size))

                data_new = torch.cat(data_new, dim=1)
                data_new = data_new.reshape(*data_new.shape[:-2], -1, window_size, data_new.shape[-1])
                data_reshape = data_new[0][:, start_block:].transpose(2, 3).transpose(1, 2)
                data_reshape = data_reshape.reshape(1, kv_repeat, *data_reshape.shape[1:])
                data_reshape = torch.mean(torch.mean(data_reshape, dim=1), dim=-1)

                group_indices, group_scores = collect_eviction_schedule(data_reshape, budget, window_size)
                if group_indices is None or group_scores is None:
                    return None, None
                indices_per_group.append(group_indices)
                scores_per_group.append(group_scores)

            layer_indices = torch.cat(indices_per_group, dim=1)
            layer_scores = torch.cat(scores_per_group, dim=1)

        if layer_indices is None or layer_scores is None:
            return None, None

        all_indices.append(layer_indices)
        all_step_scores.append(layer_scores)

    if not all_indices:
        return None, None

    return torch.stack(all_indices, dim=0), torch.stack(all_step_scores, dim=0)


def main():
    seed = 422
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--total_training_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=256)
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError("train_qwen2.py expects at least 2 CUDA devices: one for the train model and one for the reference model.")

    checkpoint_dir = None
    if args.checkpoint_path:
        checkpoint_dir = os.path.abspath(args.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

    generator = init_rng(seed)
    model, tokenizer, model2 = load_model(args.model_name)
    num_layers, num_kv_heads, kv_repeat = get_qwen2_layout(model.config)

    del generator, tokenizer

    model.to("cuda:0")
    model2.to("cuda:1")

    to_optim = [p for n, p in model.named_parameters() if "judge_model" in n]
    optimizer = optim.Adam(to_optim, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_training_steps)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = load_from_disk(args.dataset)
    prompt_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=RandomSampler(dataset),
        drop_last=True,
        collate_fn=custom_collate,
    )
    prompt_iterator = iter(prompt_loader)

    print(f"Starting Qwen2 supervised training. Total steps: {args.total_training_steps}")

    for k in tqdm(range(args.total_training_steps), desc="Training Steps"):
        try:
            prompt_batch = next(prompt_iterator)
        except StopIteration:
            prompt_iterator = iter(prompt_loader)
            prompt_batch = next(prompt_iterator)

        for param in model.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "judge_model" in name:
                param.requires_grad = True

        torch.cuda.empty_cache()
        batch_losses = []
        batch_accuracies = []

        for sample in prompt_batch:
            input_ids = torch.tensor([sample["input_ids"]], device="cuda:0")
            print(f"Processing sample with sequence length: {input_ids.shape[1]}")

            with torch.no_grad():
                output2 = model2(
                    input_ids.to(model2.device),
                    output_attentions=True,
                    output_hidden_states=True,
                )
                all_indices, all_step_scores = build_oracle_targets(
                    output2.attentions,
                    input_ids.shape[1],
                    args.budget,
                    args.window_size,
                    num_layers,
                    num_kv_heads,
                    kv_repeat,
                )

            if all_indices is None or all_step_scores is None:
                continue

            indices = all_indices.to(model.device)
            step_scores = all_step_scores.to(model.device)
            output = model(
                input_ids=input_ids.to(model.device),
                use_cache=False,
                cache_drop=indices.unsqueeze(2),
                output_attentions=True,
            )

            scores = torch.stack([torch.stack([item[1] for item in index], dim=0) for index in output.attentions], dim=0)
            scores = scores[:, : indices.shape[1]]
            target = torch.zeros_like(scores)
            target.scatter_(3, indices, 1.0)

            scores_reshaped = scores.reshape(scores.shape[0], -1, scores.shape[-1])
            step_scores_reshaped = step_scores.reshape(scores.shape[0], -1, target.shape[-1]).detach()

            all_loss = 0.0
            for layer in range(scores.shape[0]):
                loss_function = PairwiseRankHingeLoss(margin=0.01)
                loss = loss_function(scores_reshaped[layer], step_scores_reshaped[layer])
                loss.backward(retain_graph=True)
                all_loss += loss.item()
            batch_losses.append(all_loss)

            with torch.no_grad():
                _, topk_indices = torch.topk(scores, args.window_size, dim=-1)
                preds_topk = torch.zeros_like(scores)
                preds_topk.scatter_(-1, topk_indices, 1)
                accuracy = torch.mean((preds_topk * target).sum(dim=-1) / args.window_size, dim=1)
                batch_accuracies.append(accuracy)

        if not batch_losses or not batch_accuracies:
            optimizer.zero_grad(set_to_none=True)
            continue

        clip_grad_norm_(to_optim, max_norm=args.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        avg_loss = torch.mean(torch.tensor(batch_losses))
        avg_accuracy = torch.mean(torch.stack(batch_accuracies, dim=0))
        print(f"step={k + 1} loss={avg_loss.item():.4f} accuracy={avg_accuracy.item():.4f}")

        if checkpoint_dir and args.checkpoint_interval and (k + 1) % args.checkpoint_interval == 0:
            step_checkpoint_path = os.path.join(checkpoint_dir, f"step_{k}")
            model.save_pretrained(step_checkpoint_path)
            print(f"Saved checkpoint to {step_checkpoint_path}")

    if checkpoint_dir:
        final_checkpoint_path = os.path.join(checkpoint_dir, f"step_{k}_final")
        model.save_pretrained(final_checkpoint_path)
        print(f"Saved final model to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
