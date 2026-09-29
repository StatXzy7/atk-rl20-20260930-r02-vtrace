from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact import validate_clipping


@dataclass(frozen=True)
class VTraceOutput:
    vs: np.ndarray
    vs_minus_v: np.ndarray
    advantages: np.ndarray
    rho: np.ndarray
    c: np.ndarray


def clipped_ratios(
    target_probs: np.ndarray,
    behavior_probs: np.ndarray,
    rho_bar: float,
    c_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    validate_clipping(rho_bar, c_bar)
    target_probs = np.asarray(target_probs, dtype=np.float64)
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)

    if target_probs.shape != behavior_probs.shape:
        raise ValueError("target and behavior probabilities must have the same shape")
    unsupported = (behavior_probs == 0.0) & (target_probs > 0.0)
    if np.any(unsupported):
        raise ValueError(
            "target assigns positive probability to an unsampled action; "
            "do not stabilize this with epsilon"
        )

    ratios = np.zeros_like(target_probs, dtype=np.float64)
    supported = behavior_probs > 0.0
    ratios[supported] = target_probs[supported] / behavior_probs[supported]
    return np.minimum(rho_bar, ratios), np.minimum(c_bar, ratios)


def vtrace_trajectory(
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    target_probs: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    discount: float,
    rho_bar: float,
    c_bar: float,
    done: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> VTraceOutput:
    """Reference per-trajectory finite-horizon V-trace target."""

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    length = int(rewards.size)
    done = np.zeros(length, dtype=bool) if done is None else np.asarray(done, bool)
    truncated = (
        np.zeros(length, dtype=bool)
        if truncated is None
        else np.asarray(truncated, bool)
    )
    if np.any(done & truncated):
        raise ValueError("a transition cannot terminate and truncate simultaneously")

    rho, c = clipped_ratios(target_probs, behavior_probs, rho_bar, c_bar)
    for array in (rewards, values, done, truncated, rho, c):
        if array.shape != (length,):
            raise ValueError("all trajectory inputs must be one dimensional")

    vs = np.zeros(length, dtype=np.float64)
    advantages = np.zeros(length, dtype=np.float64)
    for step in reversed(range(length)):
        if done[step] or step == length - 1:
            next_value = 0.0 if done[step] else float(bootstrap_value)
            next_advantage = 0.0
        else:
            next_value = float(values[step + 1])
            next_advantage = float(advantages[step + 1])

        temporal_difference = rho[step] * (
            rewards[step] + discount * next_value - values[step]
        )
        advantages[step] = temporal_difference + discount * c[step] * next_advantage
        vs[step] = values[step] + advantages[step]

    return VTraceOutput(vs, advantages, advantages, rho, c)


def vtrace_batch(
    rewards: np.ndarray,
    behavior_probs: np.ndarray,
    target_probs: np.ndarray,
    values: np.ndarray,
    bootstrap_values: np.ndarray,
    discount: float,
    rho_bar: float,
    c_bar: float,
    done: np.ndarray,
    truncated: np.ndarray,
    valid: np.ndarray,
) -> VTraceOutput:
    """Batched recurrence with left-aligned valid steps and explicit padding."""

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float64)
    done = np.asarray(done, bool)
    truncated = np.asarray(truncated, bool)
    valid = np.asarray(valid, bool)

    if rewards.ndim != 2:
        raise ValueError("batch arrays must have shape [batch, time]")
    batch_size, length = rewards.shape
    expected = (batch_size, length)
    for name, array in (
        ("behavior_probs", behavior_probs),
        ("target_probs", target_probs),
        ("values", values),
        ("done", done),
        ("truncated", truncated),
        ("valid", valid),
    ):
        if np.asarray(array).shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if bootstrap_values.shape != (batch_size,):
        raise ValueError("bootstrap_values must have shape [batch]")
    if np.any(done & truncated & valid):
        raise ValueError("a transition cannot terminate and truncate simultaneously")

    rho, c = clipped_ratios(target_probs, behavior_probs, rho_bar, c_bar)
    rho = np.where(valid, rho, 0.0)
    c = np.where(valid, c, 0.0)

    vs = np.zeros((batch_size, length), dtype=np.float64)
    advantages = np.zeros((batch_size, length), dtype=np.float64)
    for step in reversed(range(length)):
        if step + 1 < length:
            next_valid = valid[:, step + 1]
            next_value = np.where(
                truncated[:, step],
                bootstrap_values,
                np.where(next_valid, values[:, step + 1], bootstrap_values),
            )
            next_advantage = np.where(next_valid, advantages[:, step + 1], 0.0)
        else:
            next_value = bootstrap_values
            next_advantage = np.zeros(batch_size, dtype=np.float64)
        next_value = np.where(done[:, step], 0.0, next_value)
        next_advantage = np.where(done[:, step], 0.0, next_advantage)

        temporal_difference = rho[:, step] * (
            rewards[:, step] + discount * next_value - values[:, step]
        )
        advantages[:, step] = (
            temporal_difference + discount * c[:, step] * next_advantage
        )
        vs[:, step] = values[:, step] + advantages[:, step]

    advantages = np.where(valid, advantages, 0.0)
    vs = np.where(valid, vs, 0.0)
    return VTraceOutput(vs, advantages, advantages, rho, c)
