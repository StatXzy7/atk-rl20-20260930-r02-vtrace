from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .exact import validate_clipping


@dataclass(frozen=True)
class TorchVTraceOutput:
    vs: torch.Tensor
    advantages: torch.Tensor
    rho: torch.Tensor
    c: torch.Tensor


def _torch_clipped_ratios(
    target_probs: torch.Tensor,
    behavior_probs: torch.Tensor,
    rho_bar: float,
    c_bar: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_clipping(rho_bar, c_bar)
    unsupported = (behavior_probs == 0.0) & (target_probs > 0.0)
    if bool(torch.any(unsupported)):
        raise ValueError("unsupported target action; epsilon correction is invalid")

    target_probs = target_probs.detach().to(dtype=torch.float32)
    behavior_probs = behavior_probs.detach().to(dtype=torch.float32)
    ratios = torch.where(
        behavior_probs > 0.0,
        target_probs / behavior_probs,
        torch.zeros_like(target_probs),
    )
    return torch.clamp(ratios, max=rho_bar), torch.clamp(ratios, max=c_bar)


def vtrace_torch(
    rewards: torch.Tensor,
    behavior_probs: torch.Tensor,
    target_probs: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    discount: float,
    rho_bar: float,
    c_bar: float,
    done: torch.Tensor,
    truncated: torch.Tensor,
    valid: torch.Tensor,
) -> TorchVTraceOutput:
    """Differentiable only through explicitly supplied learner losses' log-probs."""

    if rewards.ndim != 2:
        raise ValueError("batch tensors must have shape [batch, time]")
    batch_size, length = rewards.shape
    values = values.reshape(batch_size, length)
    target_values = values.detach()
    bootstrap_values = bootstrap_values.detach().reshape(batch_size)
    behavior_probs = behavior_probs.detach()
    done = done.to(torch.bool)
    truncated = truncated.to(torch.bool)
    valid = valid.to(torch.bool)

    rho, c = _torch_clipped_ratios(
        target_probs, behavior_probs, rho_bar, c_bar
    )
    rho = torch.where(valid, rho, torch.zeros_like(rho))
    c = torch.where(valid, c, torch.zeros_like(c))

    advantages = torch.zeros_like(target_values)
    for step in reversed(range(length)):
        if step + 1 < length:
            next_valid = valid[:, step + 1]
            next_value = torch.where(
                truncated[:, step],
                bootstrap_values,
                torch.where(next_valid, target_values[:, step + 1], bootstrap_values),
            )
            next_advantage = torch.where(
                next_valid, advantages[:, step + 1], torch.zeros_like(bootstrap_values)
            )
        else:
            next_value = bootstrap_values
            next_advantage = torch.zeros_like(bootstrap_values)
        next_value = torch.where(done[:, step], torch.zeros_like(next_value), next_value)
        next_advantage = torch.where(
            done[:, step], torch.zeros_like(next_advantage), next_advantage
        )
        temporal_difference = rho[:, step] * (
            rewards[:, step] + discount * next_value - target_values[:, step]
        )
        advantages[:, step] = (
            temporal_difference + discount * c[:, step] * next_advantage
        )

    advantages = torch.where(valid, advantages, torch.zeros_like(advantages)).detach()
    vs = (target_values + advantages).detach()
    return TorchVTraceOutput(vs, advantages, rho.detach(), c.detach())


def actor_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    weighted = log_probs * advantages.detach()
    if valid is None:
        return -weighted.mean()
    return -weighted[valid.to(torch.bool)].mean()


def critic_loss(values: torch.Tensor, vs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(values[valid], vs[valid].detach())
