from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VTraceResult:
    targets: np.ndarray
    advantages: np.ndarray
    policy_gradient_advantages: np.ndarray
    rho: np.ndarray
    c: np.ndarray


def importance_ratios(
    target_probs: np.ndarray,
    behavior_probs: np.ndarray,
    *,
    on_zero: str = "error",
) -> np.ndarray:
    """Compute pi(a|s)/mu(a|s) without adding epsilon to mu."""
    target = np.asarray(target_probs, dtype=np.float64)
    behavior = np.asarray(behavior_probs, dtype=np.float64)
    zero = behavior <= 0.0
    if np.any(zero):
        positive_target = target[zero] > 0.0
        if np.any(positive_target):
            if on_zero == "error":
                raise ZeroDivisionError("target action has positive pi but zero mu")
            if on_zero == "nan":
                result = np.full_like(target, np.nan, dtype=np.float64)
                valid = ~zero
                result[valid] = target[valid] / behavior[valid]
                return result
            raise ValueError("on_zero must be 'error' or 'nan'")

    denominator = np.where(behavior > 0.0, behavior, 1.0)
    return np.where(behavior > 0.0, target / denominator, 0.0)


def validate_clips(rho_bar: float, c_bar: float) -> None:
    if not rho_bar > 0.0 or not c_bar > 0.0:
        raise ValueError("rho_bar and c_bar must be positive")
    if c_bar > rho_bar:
        raise ValueError("the configured convention requires 0 < c_bar <= rho_bar")


def _vtrace_core(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_values: np.ndarray,
    rho: np.ndarray,
    c: np.ndarray,
    terminals: np.ndarray,
    truncated: np.ndarray,
    mask: np.ndarray,
    discount: float,
) -> VTraceResult:
    time_steps, batch = rewards.shape
    targets = np.full((time_steps, batch), np.nan, dtype=np.float64)
    next_values_for_td = np.zeros((time_steps, batch), dtype=np.float64)
    has_next_trace = np.zeros((time_steps, batch), dtype=bool)

    for batch_index in range(batch):
        valid_times = np.flatnonzero(mask[:, batch_index])
        for position, time in enumerate(valid_times):
            if terminals[time, batch_index] or truncated[time, batch_index]:
                next_values_for_td[time, batch_index] = bootstrap_values[time, batch_index]
            elif position + 1 < valid_times.size:
                next_time = valid_times[position + 1]
                next_values_for_td[time, batch_index] = values[next_time, batch_index]
                has_next_trace[time, batch_index] = True

    traces = np.zeros((time_steps, batch), dtype=np.float64)
    for batch_index in range(batch):
        for time in reversed(np.flatnonzero(mask[:, batch_index])):
            local_temporal_difference = rho[time, batch_index] * (
                rewards[time, batch_index]
                + discount * next_values_for_td[time, batch_index]
                - values[time, batch_index]
            )
            if has_next_trace[time, batch_index]:
                valid_times = np.flatnonzero(mask[:, batch_index])
                next_time = valid_times[np.searchsorted(valid_times, time) + 1]
                propagated_trace = traces[next_time, batch_index]
            else:
                propagated_trace = 0.0
            traces[time, batch_index] = (
                local_temporal_difference
                + discount * c[time, batch_index] * propagated_trace
            )
            targets[time, batch_index] = values[time, batch_index] + traces[time, batch_index]

    advantages = targets - values
    policy_gradient_advantages = rho * advantages
    return VTraceResult(
        targets=targets,
        advantages=advantages,
        policy_gradient_advantages=policy_gradient_advantages,
        rho=rho,
        c=c,
    )


def vtrace_targets_batch(
    rewards: np.ndarray,
    actions: np.ndarray,
    target_policy_probs: np.ndarray,
    behavior_policy_probs: np.ndarray,
    behavior_probs: np.ndarray,
    values: np.ndarray,
    bootstrap_values: np.ndarray,
    terminals: np.ndarray,
    truncated: np.ndarray,
    mask: np.ndarray,
    states: np.ndarray | None = None,
    next_states: np.ndarray | None = None,
    *,
    discount: float,
    rho_bar: float,
    c_bar: float,
) -> VTraceResult:
    """Padded [T, B] V-trace implementation.

    rho_t clips the current local TD error; c_t multiplies only the next-step
    recursive target. Bootstrap values are used for truncation but zero is used
    after true termination.
    """
    validate_clips(rho_bar, c_bar)
    rewards = np.asarray(rewards, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    target_policy_probs = np.asarray(target_policy_probs, dtype=np.float64)
    behavior_policy_probs = np.asarray(behavior_policy_probs, dtype=np.float64)
    behavior_probs = np.asarray(behavior_probs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float64)
    terminals = np.asarray(terminals, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    if rewards.ndim != 2:
        raise ValueError("all time-major inputs must have shape [T, B]")
    if target_policy_probs.shape[:2] != rewards.shape:
        raise ValueError("target_policy_probs must have shape [T, B, A]")

    time_index, batch_index = np.indices(rewards.shape)
    target_selected = target_policy_probs[time_index, batch_index, actions]
    mu_selected_from_distribution = behavior_policy_probs[time_index, batch_index, actions]
    if np.any(mask & ~np.isclose(mu_selected_from_distribution, behavior_probs)):
        raise ValueError("logged action probability must match logged behavior policy")

    ratio = importance_ratios(target_selected, behavior_probs)
    rho = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)

    return _vtrace_core(
        rewards=rewards,
        values=values,
        bootstrap_values=bootstrap_values,
        rho=rho,
        c=c,
        terminals=terminals,
        truncated=truncated,
        mask=mask,
        discount=discount,
    )


def vtrace_targets_trajectory(
    rewards: np.ndarray,
    actions: np.ndarray,
    target_policy_probs: np.ndarray,
    behavior_policy_probs: np.ndarray,
    behavior_probs: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    terminals: np.ndarray,
    truncated: np.ndarray,
    *,
    discount: float,
    rho_bar: float,
    c_bar: float,
) -> VTraceResult:
    """Per-trajectory reference implementation; arrays are length [T]."""
    length = rewards.shape[0]
    batch_kwargs = dict(
        rewards=np.asarray(rewards)[:, None],
        actions=np.asarray(actions)[:, None],
        target_policy_probs=np.asarray(target_policy_probs)[:, None, :],
        behavior_policy_probs=np.asarray(behavior_policy_probs)[:, None, :],
        behavior_probs=np.asarray(behavior_probs)[:, None],
        values=np.asarray(values)[:, None],
        bootstrap_values=np.full((length, 1), float(bootstrap_value), dtype=np.float64),
        terminals=np.asarray(terminals, dtype=bool)[:, None],
        truncated=np.asarray(truncated, dtype=bool)[:, None],
        mask=np.ones((length, 1), dtype=bool),
    )
    result = vtrace_targets_batch(
        **batch_kwargs,
        discount=discount,
        rho_bar=rho_bar,
        c_bar=c_bar,
    )
    return VTraceResult(
        targets=result.targets[:, 0],
        advantages=result.advantages[:, 0],
        policy_gradient_advantages=result.policy_gradient_advantages[:, 0],
        rho=result.rho[:, 0],
        c=result.c[:, 0],
    )


def vtrace_for_segments(
    segments: list,
    target_policy: np.ndarray,
    state_values: np.ndarray,
    *,
    discount: float,
    rho_bar: float,
    c_bar: float,
) -> list[VTraceResult]:
    target_policy = np.asarray(target_policy, dtype=np.float64)
    state_values = np.asarray(state_values, dtype=np.float64)
    output = []

    for segment in segments:
        values = state_values[segment.states]
        last_next = int(segment.next_states[-1])
        bootstrap_value = 0.0 if segment.terminals[-1] else state_values[last_next]
        output.append(
            vtrace_targets_trajectory(
                rewards=segment.rewards,
                actions=segment.actions,
                target_policy_probs=target_policy[segment.states],
                behavior_policy_probs=segment.behavior_policy_probs,
                behavior_probs=segment.behavior_probs,
                values=values,
                bootstrap_value=bootstrap_value,
                terminals=segment.terminals,
                truncated=segment.truncated,
                discount=discount,
                rho_bar=rho_bar,
                c_bar=c_bar,
            )
        )
    return output
