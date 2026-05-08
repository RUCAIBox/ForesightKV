from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    indices: Optional[torch.Tensor],
) -> torch.Tensor:


    log_ratio =log_probs_ref.float() - log_probs.float()
    
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float=0.01) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_past: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_init: Optional[torch.Tensor]=None,
        indices: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        old_log_probs = log_probs_past
        advantages = advantages.clone()

        log_ratio = (log_probs - old_log_probs).clamp(-10, 10)

        ratio = log_ratio.exp()
        surr1 = ratio * advantages

        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

        loss = -torch.min(surr1, surr2)
        if log_probs_init is not None:
            kl = approx_kl_divergence(log_probs, log_probs_init, indices)
            loss = loss.mean() + self.kl_weight * kl.mean()
            
        else:
            loss =loss.mean()
        return loss, kl.mean()
