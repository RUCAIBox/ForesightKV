import argparse
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from cache import cache_process_rl, init_cache_rl
from loss import GRPOLoss
from qwen3 import Qwen3Attention, Qwen3ForCausalLM
from replay_buffer import Experience, ReplayBuffer


Qwen3Attention.init_cache = init_cache_rl
Qwen3Attention.cache_process = cache_process_rl


def sequence_log_probs_from_logits(
    logits: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_prob = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(log_prob * torch.exp(log_prob), dim=-1)
    return torch.gather(log_prob, dim=-1, index=indices), entropy


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
    model = Qwen3ForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


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
        entropy = token_loss

    token_loss_tensor = torch.as_tensor(token_loss, device=device)
    entropy_tensor = torch.as_tensor(entropy, device=device)
    if token_loss_tensor.ndim == 2 and token_loss_tensor.shape[0] == 1:
        token_loss_tensor = token_loss_tensor[0]
    if entropy_tensor.ndim == 2 and entropy_tensor.shape[0] == 1:
        entropy_tensor = entropy_tensor[0]

    return input_ids, float(mean_loss), token_loss_tensor, entropy_tensor


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        return 1.0

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
def rollout(model, input_ids, num_rollouts):
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
    )

    a = input_ids[:, 1:].reshape(-1)
    loss = criterion(output.logits.float()[:, :-1].reshape(a.shape[0], -1), a)

    indices, scores = extract_attention_cache(output.attentions)
    if indices is None or scores is None:
        return None, None, None, None, None
    scores = scores.reshape(indices.shape[0], indices.shape[1], num_rollouts, indices.shape[3], -1)

    return input_ids, indices, scores, output.loss, loss


def init_rng(seed: int) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)


def group_advantages(returns, eps=1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_returns = [torch.zeros_like(returns) for _ in range(dist.get_world_size())]
    dist.all_gather(all_returns, returns)
    all_returns = torch.cat(all_returns, dim=0)

    mean = all_returns.mean(dim=0, keepdim=True)
    std = all_returns.std(dim=0, keepdim=True, unbiased=False) + eps
    return (returns - mean) / std, all_returns, (all_returns - mean) / std


def dist_broadcast_data(data, src=0):
    payload = [data if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total_training_steps", type=int, default=250)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--rollouts_per_step", type=int, default=32)
    parser.add_argument("--epochs_per_step", type=int, default=1)
    parser.add_argument("--max_norm", type=float, default=1.0)
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank()
    seed = 42

    checkpoint_dir = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", rank % torch.cuda.device_count())
    init_rng(seed)
    model, tokenizer = load_model(args.model_name)
    del tokenizer
    if model.device != device:
        model.to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    num_layers = len(model.module.model.layers)

    for layer_idx in range(num_layers):
        with torch.no_grad():
            layer = model.module.model.layers[layer_idx].self_attn
            layer.reference.load_state_dict(layer.judge_model.state_dict())
            for param in layer.reference.parameters():
                param.requires_grad = False

    to_optim = [p for n, p in model.named_parameters() if "judge_model" in n]
    optimizer = optim.Adam(to_optim, lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_training_steps)
    model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    dataset = load_rl_dataset(args.data_name)
    if rank == 0:
        prompt_dataloader = DataLoader(
            dataset,
            batch_size=args.rollouts_per_step,
            sampler=RandomSampler(dataset),
            drop_last=True,
            collate_fn=custom_collate,
        )
        prompt_loader = iter(prompt_dataloader)

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=args.clip_eps, kl_weight=args.kl_weight)

    for k in tqdm(range(args.total_training_steps)):
        objective.kl_weight = 0.0 if k > 100 else args.kl_weight
        if rank == 0:
            try:
                prompt_batch = next(prompt_loader)
            except StopIteration:
                prompt_loader = iter(prompt_dataloader)
                prompt_batch = next(prompt_loader)
        else:
            prompt_batch = None
        prompt_batch = dist_broadcast_data(prompt_batch)

        replay_buffer.clear()
        torch.cuda.empty_cache()
        all_losses = []

        with torch.no_grad():
            for sample in prompt_batch:
                input_ids, loss, loss2, entropy = get_sample_fields(sample, device)
                input_ids, indices, scores, returns, lm_loss = rollout(model.module, input_ids, args.group_size)
                if indices is None:
                    continue

                window, budget = init_model_chunks(model.module, input_ids.shape[-1])
                init_output = model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=True,
                    cache_drop=indices,
                    output_logits=False,
                    reference=True,
                )
                _, init_scores = extract_attention_cache(init_output.attentions)
                if init_scores is None:
                    continue
                init_scores = init_scores.reshape(indices.shape[0], indices.shape[1], args.group_size, indices.shape[3], -1)

                entropy_tail = entropy[window + budget :]
                if entropy_tail.numel() == 0:
                    continue
                top_20_percent_idx = int(len(entropy_tail) * 0.2)
                sorted_indices = torch.argsort(entropy_tail, descending=True)
                other_indices = sorted_indices[top_20_percent_idx:]
                if other_indices.numel() == 0:
                    continue

                loss_low = lm_loss[window + budget :][other_indices]
                loss_low_ori = loss2[window + budget :][other_indices]
                delta = loss_low_ori - loss_low
                new_loss = torch.mean((delta**2) * (delta < -1.5).float())

                advantages, all_returns, _ = group_advantages(-new_loss.unsqueeze(0))
                experience = Experience(
                    input_ids=input_ids,
                    indices=indices,
                    scores=scores,
                    advantages=advantages[:1],
                    returns=returns,
                    loss_ori=loss,
                    init_scores=init_scores,
                )
                replay_buffer.append(experience.to("cpu"))
                all_losses.append(-torch.mean(all_returns))
                torch.cuda.empty_cache()

        if len(replay_buffer) == 0:
            if rank == 0:
                print("No valid samples in this step, skipping.")
            continue

        if rank == 0:
            step_loss = torch.mean(torch.stack(all_losses, dim=0))
            print(step_loss)

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate,
        )

        for param in model.parameters():
            param.requires_grad = False
        for layer_idx in range(num_layers):
            layer = model.module.model.layers[layer_idx].self_attn
            for param in layer.judge_model.parameters():
                param.requires_grad = True
            for param in layer.reference.parameters():
                param.requires_grad = False

        model.train()
        for _ in range(args.epochs_per_step):
            for exps in experience_sampler:
                for exp in exps:
                    exp = exp.to(device)
                    input_ids = exp.input_ids
                    indices = exp.indices
                    scores = exp.scores
                    advantages = exp.advantages
                    init_scores = exp.init_scores

                    init_model_chunks(model.module, input_ids.shape[-1])
                    new_output = model(
                        input_ids=input_ids,
                        use_cache=True,
                        output_attentions=True,
                        cache_drop=indices,
                        output_logits=False,
                    )
                    _, new_scores = extract_attention_cache(new_output.attentions)
                    if new_scores is None:
                        continue
                    new_scores = new_scores.reshape(indices.shape[0], indices.shape[1], args.group_size, indices.shape[3], -1)
                    loss_total = 0.0

                    for layer_idx in range(num_layers):
                        advantages_new = advantages.clone().unsqueeze(-1).unsqueeze(-1)
                        log_scores, _ = sequence_log_probs_from_logits(scores[layer_idx], indices[layer_idx])
                        log_scores_new, _ = sequence_log_probs_from_logits(new_scores[layer_idx], indices[layer_idx])
                        log_scores_init, _ = sequence_log_probs_from_logits(init_scores[layer_idx], indices[layer_idx])
                        loss, _ = objective(
                            log_probs=log_scores_new,
                            log_probs_past=log_scores,
                            log_probs_init=log_scores_init,
                            advantages=advantages_new,
                            indices=indices[layer_idx],
                        )
                        loss_total += loss / max(1, args.rollouts_per_step)
                    loss_total.backward()

            clip_grad_norm_(to_optim, max_norm=args.max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        if checkpoint_dir is not None and args.checkpoint_interval and (k + 1) % args.checkpoint_interval == 0 and rank == 0:
            model.module.save_pretrained(str(checkpoint_dir / f"step_{k}"))

    if checkpoint_dir is not None and rank == 0:
        model.module.save_pretrained(str(checkpoint_dir / "step_final"))

    cleanup_ddp()


if __name__ == "__main__":
    main()
