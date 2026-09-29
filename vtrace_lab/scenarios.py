from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteMDP


@dataclass(frozen=True)
class Scenario:
    name: str
    mdp: FiniteMDP
    mu: np.ndarray
    pi: np.ndarray
    versions: list[np.ndarray]
    initial_state: int


def lag_chain_mdp() -> FiniteMDP:
    states, actions, terminal = 4, 2, 3
    transitions = np.zeros((states, actions, states))
    rewards = np.zeros((states, actions))
    for state in (0, 1):
        transitions[state, :, state + 1] = 1.0
    transitions[2, :, terminal] = 1.0
    rewards[2, 0] = 1.0
    return FiniteMDP(transitions, rewards, 0.9, terminal)


def lag_chain_scenario(rho_bar: float = 2.0) -> Scenario:
    mdp = lag_chain_mdp()
    mu = np.array(
        [
            [0.80, 0.20],
            [0.80, 0.20],
            [0.05, 0.95],
            [1.0, 0.0],
        ]
    )
    middle = np.array(
        [
            [0.875, 0.125],
            [0.875, 0.125],
            [0.50, 0.50],
            [1.0, 0.0],
        ]
    )
    pi = np.array(
        [
            [0.95, 0.05],
            [0.95, 0.05],
            [0.95, 0.05],
            [1.0, 0.0],
        ]
    )
    return Scenario("lag_chain", mdp, mu, pi, [mu, middle, pi], 0)


def support_gap_scenario() -> Scenario:
    mdp = lag_chain_mdp()
    mu = np.array(
        [
            [0.0, 1.0],
            [0.8, 0.2],
            [0.95, 0.05],
            [1.0, 0.0],
        ]
    )
    pi = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.95, 0.05],
            [1.0, 0.0],
        ]
    )
    return Scenario("support_gap", mdp, mu, pi, [mu, pi], 0)


def terminal_bandit_mdp() -> FiniteMDP:
    transitions = np.zeros((2, 2, 2))
    rewards = np.zeros((2, 2))
    transitions[0, :, 1] = 1.0
    rewards[0, 0] = 0.8
    return FiniteMDP(transitions, rewards, 0.99, 1)


def bias_variance_bandit() -> Scenario:
    mdp = terminal_bandit_mdp()
    mu = np.array([[0.2, 0.8], [1.0, 0.0]])
    pi = np.array([[0.8, 0.2], [1.0, 0.0]])
    return Scenario("bias_variance_bandit", mdp, mu, pi, [mu, pi], 0)


def geometric_versions(mu: np.ndarray, pi: np.ndarray, steps: int) -> list[np.ndarray]:
    versions = []
    for alpha in np.linspace(0.0, 1.0, steps):
        versions.append((1.0 - alpha) * mu + alpha * pi)
    return versions


def c_trace_chain(length: int = 6) -> Scenario:
    terminal = length
    states = terminal + 1
    transitions = np.zeros((states, 2, states))
    rewards = np.zeros((states, 2))
    for state in range(length):
        transitions[state, :, state + 1] = 1.0
    rewards[length - 1, :] = 1.0
    mdp = FiniteMDP(transitions, rewards, 0.9, terminal)
    policy = np.zeros((states, 2))
    policy[:length, 0] = 1.0
    policy[terminal, 0] = 1.0
    return Scenario("c_trace_chain", mdp, policy, policy, [policy], 0)
