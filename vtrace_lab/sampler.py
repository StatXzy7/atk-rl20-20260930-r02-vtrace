from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import FiniteMDP, validate_policy


@dataclass(frozen=True)
class Segment:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    truncated: np.ndarray
    behavior_probs: np.ndarray
    target_probs_at_sample: np.ndarray
    bootstrap_state: int
    behavior_version: int
    target_version: int
    actual_length: int

    def __len__(self) -> int:
        return int(self.actions.size)


class PolicyLagSampler:
    """Sample with a frozen stale target snapshot; no concurrency is used."""

    def __init__(
        self,
        mdp: FiniteMDP,
        target_versions: list[np.ndarray],
        lag_steps: int = 0,
        initial_state: int | None = None,
    ):
        self.mdp = mdp
        self.target_versions = [validate_policy(p, mdp) for p in target_versions]
        self.lag_steps = int(lag_steps)
        self.initial_state = initial_state
        if self.target_versions and self.lag_steps >= len(self.target_versions):
            raise ValueError("lag_steps must be smaller than the number of versions")

    @property
    def target_version(self) -> int:
        return max(0, len(self.target_versions) - 1 - self.lag_steps)

    @property
    def behavior_policy(self) -> np.ndarray:
        return self.target_versions[self.target_version]

    def sample(
        self,
        length: int,
        rng: np.random.Generator,
        initial_state: int | None = None,
    ) -> Segment:
        if length <= 0:
            raise ValueError("segment length must be positive")

        start = self.initial_state if initial_state is None else initial_state
        if start is None:
            start = int(rng.choice(self.mdp.nonterminal_states))
        if start == self.mdp.terminal_state:
            raise ValueError("segments cannot start in the terminal state")

        states = np.empty(length, dtype=np.int64)
        actions = np.empty(length, dtype=np.int64)
        rewards = np.empty(length, dtype=np.float64)
        done = np.zeros(length, dtype=bool)
        truncated = np.zeros(length, dtype=bool)
        behavior_probs = np.zeros(length, dtype=np.float64)
        target_probs = np.zeros(length, dtype=np.float64)

        state = int(start)
        actual_length = length
        live_version = len(self.target_versions) - 1
        for step in range(length):
            action = int(rng.choice(self.mdp.n_actions, p=self.behavior_policy[state]))
            b_prob = float(self.behavior_policy[state, action])
            pi_prob = float(self.target_versions[live_version][state, action])

            states[step] = state
            actions[step] = action
            behavior_probs[step] = b_prob
            target_probs[step] = pi_prob

            next_state, reward, is_done = self.mdp.step(state, action, rng)
            rewards[step] = reward
            done[step] = is_done
            truncated[step] = (not is_done) and step == length - 1
            bootstrap_state = int(next_state)

            if is_done:
                for extra in range(step + 1, length):
                    states[extra] = self.mdp.terminal_state
                    actions[extra] = 0
                    rewards[extra] = 0.0
                    behavior_probs[extra] = 0.0
                    target_probs[extra] = 0.0
                actual_length = step + 1
                bootstrap_state = self.mdp.terminal_state
                break
            state = next_state

        return Segment(
            states=states,
            actions=actions,
            rewards=rewards,
            done=done,
            truncated=truncated,
            behavior_probs=behavior_probs,
            target_probs_at_sample=target_probs,
            bootstrap_state=bootstrap_state,
            behavior_version=self.target_version,
            target_version=live_version,
            actual_length=actual_length,
        )


def collate_segments(segments: list[Segment]) -> dict[str, np.ndarray]:
    """Left-pad completed trajectories; valid steps are at the start of a row."""

    if not segments:
        raise ValueError("segments cannot be empty")
    batch_size = len(segments)
    length = max(len(segment) for segment in segments)
    actual_length = np.asarray([segment.actual_length for segment in segments], dtype=np.int64)
    float_pad = "rewards", "behavior_probs", "target_probs_at_sample"

    data = {
        "states": np.zeros((batch_size, length), dtype=np.int64),
        "actions": np.zeros((batch_size, length), dtype=np.int64),
        "rewards": np.zeros((batch_size, length), dtype=np.float64),
        "behavior_probs": np.zeros((batch_size, length), dtype=np.float64),
        "target_probs_at_sample": np.zeros((batch_size, length), dtype=np.float64),
        "done": np.zeros((batch_size, length), dtype=bool),
        "truncated": np.zeros((batch_size, length), dtype=bool),
        "bootstrap_state": np.zeros(batch_size, dtype=np.int64),
        "valid": np.zeros((batch_size, length), dtype=bool),
        "behavior_version": np.empty(batch_size, dtype=np.int64),
        "target_version": np.empty(batch_size, dtype=np.int64),
        "actual_length": actual_length,
    }

    for row, segment in enumerate(segments):
        segment_length = segment.actual_length
        end = slice(0, segment_length)
        data["valid"][row, end] = True
        data["states"][row, end] = segment.states[:segment_length]
        data["actions"][row, end] = segment.actions[:segment_length]
        data["done"][row, end] = segment.done[:segment_length]
        data["truncated"][row, end] = segment.truncated[:segment_length]
        data["bootstrap_state"][row] = segment.bootstrap_state
        data["behavior_version"][row] = segment.behavior_version
        data["target_version"][row] = segment.target_version
        for name in float_pad:
            data[name][row, end] = getattr(segment, name)[:segment_length]

    return data


def visited_support_missing(batch: dict[str, np.ndarray]) -> np.ndarray:
    return batch["valid"] & (batch["target_probs_at_sample"] > 0.0) & (
        batch["behavior_probs"] == 0.0
    )
