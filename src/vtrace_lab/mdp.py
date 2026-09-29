from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteMDP:
    transition: np.ndarray
    reward: np.ndarray
    initial_state: int = 0
    terminal_states: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        p = np.asarray(self.transition, dtype=np.float64)
        r = np.asarray(self.reward, dtype=np.float64)
        if p.ndim != 3 or p.shape[0] != p.shape[2]:
            raise ValueError("transition must have shape [states, actions, states]")
        if r.shape != p.shape[:2]:
            raise ValueError("reward must have shape [states, actions]")
        if np.any(p < 0.0) or not np.allclose(p.sum(axis=2), 1.0):
            raise ValueError("each transition row must be a probability distribution")
        if not np.all(np.isfinite(r)):
            raise ValueError("rewards must be finite")
        object.__setattr__(self, "transition", p)
        object.__setattr__(self, "reward", r)

    @property
    def n_states(self) -> int:
        return self.transition.shape[0]

    @property
    def n_actions(self) -> int:
        return self.transition.shape[1]

    @property
    def nonterminal_states(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.n_states), self.terminal_states)

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states

    def step(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        next_state = int(rng.choice(self.n_states, p=self.transition[state, action]))
        reward = float(self.reward[state, action])
        return next_state, reward, self.is_terminal(next_state)


def make_lag_mdp(horizon: int = 6, slip: float = 0.20) -> FiniteMDP:
    """Create a small stochastic discounted chain with a real terminal state.

    Action 0 advances; action 1 stays. Both actions have stochastic slips to
    termination, so finite-horizon rollouts include both true termination and
    time-limit truncation.
    """
    if not 0.0 <= slip < 1.0:
        raise ValueError("slip must be in [0, 1)")
    n_nonterminal = max(1, horizon - 2)
    n_states = n_nonterminal + 1
    terminal = n_states - 1
    p = np.zeros((n_states, 2, n_states), dtype=np.float64)
    r = np.zeros((n_states, 2), dtype=np.float64)

    for state in range(n_nonterminal):
        advance = min(state + 1, terminal)
        p[state, 0, advance] += 1.0 - slip
        p[state, 0, state] += slip * 0.5
        p[state, 0, terminal] += slip * 0.5

        p[state, 1, state] += 1.0 - slip
        p[state, 1, advance] += slip * 0.5
        p[state, 1, terminal] += slip * 0.5

        r[state, 0] = 0.45 + 0.05 * state
        r[state, 1] = -0.15 - 0.02 * state

    p[terminal, :, terminal] = 1.0
    return FiniteMDP(p, r, initial_state=0, terminal_states=(terminal,))


def make_one_shot_mdp() -> FiniteMDP:
    """Two-state one-step MDP used for the low-variance/biased counterexample."""
    p = np.zeros((2, 2, 2), dtype=np.float64)
    p[0, :, 1] = 1.0
    p[1, :, 1] = 1.0
    r = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float64)
    return FiniteMDP(p, r, initial_state=0, terminal_states=(1,))
