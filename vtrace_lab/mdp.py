from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FiniteMDP:
    """A discounted finite MDP with one explicit absorbing terminal state."""

    transitions: np.ndarray
    rewards: np.ndarray
    discount: float
    terminal_state: int

    def __post_init__(self) -> None:
        transitions = np.asarray(self.transitions, dtype=np.float64)
        rewards = np.asarray(self.rewards, dtype=np.float64)

        if transitions.ndim != 3:
            raise ValueError("transitions must have shape [states, actions, next_states]")
        if rewards.shape != transitions.shape[:2]:
            raise ValueError("rewards must have shape [states, actions]")
        n_states, n_actions, _ = transitions.shape
        if not (0.0 <= self.discount < 1.0):
            raise ValueError("discount must satisfy 0 <= gamma < 1")
        if not 0 <= self.terminal_state < n_states:
            raise ValueError("terminal_state is out of range")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("rewards must be finite")

        nonterminal = self.nonterminal_states
        rows = transitions[nonterminal].reshape(-1, n_states)
        if not np.allclose(rows.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("every nonterminal transition row must sum to one")
        if np.any(rows < -1e-12):
            raise ValueError("transition probabilities must be nonnegative")

        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "rewards", rewards)

    @property
    def n_states(self) -> int:
        return int(self.transitions.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.transitions.shape[1])

    @property
    def nonterminal_states(self) -> np.ndarray:
        return np.asarray(
            [s for s in range(self.n_states) if s != self.terminal_state], dtype=np.int64
        )

    def step(
        self, state: int, action: int, rng: np.random.Generator
    ) -> tuple[int, float, bool]:
        next_state = int(
            rng.choice(self.n_states, p=self.transitions[state, action])
        )
        reward = float(self.rewards[state, action])
        return next_state, reward, next_state == self.terminal_state


def validate_policy(policy: np.ndarray, mdp: FiniteMDP) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    if policy.shape != (mdp.n_states, mdp.n_actions):
        raise ValueError(
            f"policy must have shape {(mdp.n_states, mdp.n_actions)}"
        )
    if np.any(policy < 0.0) or np.any(policy > 1.0):
        raise ValueError("policy probabilities must lie in [0, 1]")
    sums = policy[mdp.nonterminal_states].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-10):
        raise ValueError("every nonterminal policy row must sum to one")
    return policy
