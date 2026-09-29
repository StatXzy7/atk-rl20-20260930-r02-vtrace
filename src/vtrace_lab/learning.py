from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import softmax_policy


@dataclass(frozen=True)
class TabularGradients:
    policy_logits: np.ndarray
    values: np.ndarray


def corrected_policy_gradient(
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    policy_logits: np.ndarray,
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """Tabular score-function gradient for sum_t rho_t A_t log pi_theta(a_t|s_t).

    Callers pass ``policy_gradient_advantages`` (rho * (v_trace - V)). Those
    scalars are fixed data; no gradient is propagated through behavior
    probabilities, target probabilities, or importance ratios.
    """
    states = np.asarray(states, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    advantages = np.asarray(advantages, dtype=np.float64)
    probs = softmax_policy(policy_logits)
    gradient = np.zeros((n_states, n_actions), dtype=np.float64)

    for state, action, advantage in zip(states, actions, advantages):
        gradient[state] -= advantage * probs[state]
        gradient[state, action] += advantage
    return gradient


def value_gradient(
    states: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Gradient of 1/2 mean squared error toward detached V-trace targets."""
    gradient = np.zeros(n_states, dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)
    for state, value, target in zip(states, values, targets):
        gradient[state] += value - target
        counts[state] += 1
    visited = counts > 0
    gradient[visited] /= counts[visited]
    return gradient


def train_step(
    segments: list,
    results: list,
    policy_logits: np.ndarray,
    state_values: np.ndarray,
    *,
    policy_learning_rate: float,
    value_learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.concatenate([segment.states for segment in segments])
    actions = np.concatenate([segment.actions for segment in segments])
    advantages = np.concatenate(
        [result.policy_gradient_advantages for result in results]
    )
    targets = np.concatenate([result.targets for result in results])
    values = state_values[states]

    policy_grad = corrected_policy_gradient(
        states=states,
        actions=actions,
        advantages=advantages,
        policy_logits=policy_logits,
        n_states=policy_logits.shape[0],
        n_actions=policy_logits.shape[1],
    )
    value_grad = value_gradient(
        states=states,
        values=values,
        targets=targets,
        n_states=state_values.size,
    )
    updated_logits = policy_logits + policy_learning_rate * policy_grad
    updated_values = state_values - value_learning_rate * value_grad
    return updated_logits, updated_values
