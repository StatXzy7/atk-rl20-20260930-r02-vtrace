from __future__ import annotations

import numpy as np


def pad_segments(
    segments: list,
    target_policy: np.ndarray,
    state_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pad variable-length trajectories into a [T, B] tensor batch."""
    if not segments:
        raise ValueError("segments must not be empty")
    batch_size = len(segments)
    time_steps = max(segment.length for segment in segments)
    n_actions = target_policy.shape[1]

    def zeros(shape, dtype=np.float64):
        return np.zeros((time_steps, batch_size, *shape), dtype=dtype)

    rewards = zeros(())
    behavior_probs = zeros(())
    values = zeros(())
    bootstrap_values = zeros(())
    states = zeros((), dtype=np.int64)
    actions = zeros((), dtype=np.int64)
    next_states = zeros((), dtype=np.int64)
    target_policy_probs = zeros((n_actions,))
    behavior_policy_probs = zeros((n_actions,))
    terminals = zeros((), dtype=bool)
    truncated = zeros((), dtype=bool)
    mask = zeros((), dtype=bool)

    for batch_index, segment in enumerate(segments):
        length = segment.length
        index = (slice(0, length), batch_index)
        rewards[index] = segment.rewards
        states[index] = segment.states
        actions[index] = segment.actions
        next_states[index] = segment.next_states
        behavior_probs[index] = segment.behavior_probs
        behavior_policy_probs[index] = segment.behavior_policy_probs
        target_policy_probs[index] = target_policy[segment.states]
        values[index] = state_values[segment.states]
        terminals[index] = segment.terminals
        truncated[index] = segment.truncated
        mask[index] = True

        last_next = int(segment.next_states[-1])
        bootstrap_values[length - 1, batch_index] = (
            0.0 if segment.terminals[-1] else state_values[last_next]
        )

    return {
        "rewards": rewards,
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "target_policy_probs": target_policy_probs,
        "behavior_policy_probs": behavior_policy_probs,
        "behavior_probs": behavior_probs,
        "values": values,
        "bootstrap_values": bootstrap_values,
        "terminals": terminals,
        "truncated": truncated,
        "mask": mask,
    }
