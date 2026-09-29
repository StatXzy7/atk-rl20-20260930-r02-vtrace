from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteMDP
from .policies import softmax_policy


@dataclass(frozen=True)
class Segment:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminals: np.ndarray
    truncated: np.ndarray
    behavior_probs: np.ndarray
    behavior_policy_probs: np.ndarray
    behavior_version: int
    learner_version_at_sample: int
    length: int


class LaggedPolicyActor:
    """Actor that uses an explicitly versioned learner policy snapshot."""

    def __init__(
        self,
        mdp: FiniteMDP,
        policy_versions: np.ndarray,
        lag: int = 0,
        learner_version: int | None = None,
    ) -> None:
        policies = np.asarray(policy_versions, dtype=np.float64)
        if policies.ndim != 3 or policies.shape[1:] != (mdp.n_states, mdp.n_actions):
            raise ValueError("policy_versions must have shape [versions, states, actions]")
        if not np.allclose(policies.sum(axis=2), 1.0) or np.any(policies < 0.0):
            raise ValueError("every policy version must be a probability distribution")
        if lag < 0:
            raise ValueError("lag must be non-negative")
        self.mdp = mdp
        self.policy_versions = policies
        self.lag = int(lag)
        self.learner_version = (
            len(policies) - 1 if learner_version is None else int(learner_version)
        )

    @property
    def behavior_version(self) -> int:
        return max(0, self.learner_version - self.lag)

    @property
    def behavior_policy(self) -> np.ndarray:
        return self.policy_versions[self.behavior_version]

    @property
    def learner_policy(self) -> np.ndarray:
        return self.policy_versions[self.learner_version]

    def update_learner(self, policy_or_logits: np.ndarray) -> None:
        policy = np.asarray(policy_or_logits, dtype=np.float64)
        if policy.ndim == 2 and policy.shape == (self.mdp.n_states, self.mdp.n_actions):
            if not np.allclose(policy.sum(axis=1), 1.0):
                policy = softmax_policy(policy)
        else:
            policy = softmax_policy(policy)
        self.policy_versions = np.concatenate(
            [self.policy_versions, policy[None, ...]], axis=0
        )
        self.learner_version = len(self.policy_versions) - 1

    def act(self, state: int, rng: np.random.Generator) -> tuple[int, float, np.ndarray]:
        probs = self.behavior_policy[state]
        action = int(rng.choice(self.mdp.n_actions, p=probs))
        return action, float(probs[action]), probs.copy()


def collect_episode(
    mdp: FiniteMDP,
    actor: LaggedPolicyActor,
    rng: np.random.Generator,
    max_steps: int,
) -> Segment:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    states = np.zeros(max_steps, dtype=np.int64)
    actions = np.zeros(max_steps, dtype=np.int64)
    rewards = np.zeros(max_steps, dtype=np.float64)
    next_states = np.zeros(max_steps, dtype=np.int64)
    terminals = np.zeros(max_steps, dtype=bool)
    truncated = np.zeros(max_steps, dtype=bool)
    behavior_probs = np.zeros(max_steps, dtype=np.float64)
    behavior_policy_probs = np.zeros((max_steps, mdp.n_actions), dtype=np.float64)

    state = mdp.initial_state
    behavior_version = actor.behavior_version
    learner_version = actor.learner_version
    steps = 0

    for steps in range(max_steps):
        action, behavior_prob, policy_probs = actor.act(state, rng)
        next_state, reward, terminated = mdp.step(state, action, rng)

        states[steps] = state
        actions[steps] = action
        rewards[steps] = reward
        next_states[steps] = next_state
        terminals[steps] = terminated
        behavior_probs[steps] = behavior_prob
        behavior_policy_probs[steps] = policy_probs

        state = next_state
        if terminated:
            break

    length = steps + 1
    if not terminals[length - 1]:
        truncated[length - 1] = True

    return Segment(
        states=states[:length],
        actions=actions[:length],
        rewards=rewards[:length],
        next_states=next_states[:length],
        terminals=terminals[:length],
        truncated=truncated[:length],
        behavior_probs=behavior_probs[:length],
        behavior_policy_probs=behavior_policy_probs[:length],
        behavior_version=int(behavior_version),
        learner_version_at_sample=int(learner_version),
        length=int(length),
    )


def collect_segments(
    mdp: FiniteMDP,
    actor: LaggedPolicyActor,
    rng: np.random.Generator,
    n_episodes: int,
    max_steps: int,
) -> list[Segment]:
    return [
        collect_episode(mdp, actor, rng, max_steps)
        for _ in range(int(n_episodes))
    ]
