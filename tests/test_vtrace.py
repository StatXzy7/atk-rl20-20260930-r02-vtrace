import numpy as np
import pytest

from vtrace_lab.batching import pad_segments
from vtrace_lab.exact import support_report
from vtrace_lab.mdp import make_lag_mdp
from vtrace_lab.sampling import LaggedPolicyActor, collect_segments
from vtrace_lab.vtrace import (
    importance_ratios,
    vtrace_targets_batch,
    vtrace_targets_trajectory,
)


def test_c_bar_must_not_exceed_rho_bar():
    with pytest.raises(ValueError, match="c_bar <= rho_bar"):
        vtrace_targets_trajectory(
            np.zeros(1),
            np.zeros(1, dtype=int),
            np.ones((1, 2)) / 2,
            np.ones((1, 2)) / 2,
            np.ones(1) / 2,
            np.zeros(1),
            0.0,
            np.ones(1, dtype=bool),
            np.zeros(1, dtype=bool),
            discount=0.9,
            rho_bar=0.5,
            c_bar=1.0,
        )


def test_importance_ratio_does_not_epsilon_zero_behavior():
    with pytest.raises(ZeroDivisionError):
        importance_ratios(np.array([0.5]), np.array([0.0]))


def test_pi_equals_mu_trajectory_reduces_to_n_step_target():
    rewards = np.array([1.0, -0.5, 0.25])
    actions = np.array([0, 1, 0])
    probs = np.full((3, 2), 0.5)
    behavior_probs = np.full(3, 0.5)
    values = np.array([0.1, -0.2, 0.3])
    terminals = np.array([False, False, True])
    truncated = np.array([False, False, False])
    result = vtrace_targets_trajectory(
        rewards, actions, probs, probs, behavior_probs, values, 0.0,
        terminals, truncated, discount=0.9, rho_bar=np.inf, c_bar=np.inf,
    )
    assert result.rho[0] == 1.0
    assert np.isclose(
        result.targets[0],
        rewards[0] + 0.9 * rewards[1] + 0.9 ** 2 * rewards[2],
    )
    assert np.isclose(result.targets[2], values[2] + rewards[2] - values[2])


def test_rho_and_c_have_separate_recursive_positions():
    rewards = np.array([1.0, 2.0])
    actions = np.zeros(2, dtype=int)
    target = np.tile([[0.8, 0.2]], (2, 1))
    behavior_distribution = np.tile([[0.4, 0.6]], (2, 1))
    behavior_probs = np.full(2, 0.4)
    values = np.zeros(2)
    terminals = np.array([False, True])
    truncated = np.array([False, False])
    result = vtrace_targets_trajectory(
        rewards, actions, target, behavior_distribution, behavior_probs,
        values, 0.0, terminals, truncated, discount=1.0, rho_bar=2.0, c_bar=0.5,
    )
    assert result.rho[0] == 2.0
    assert result.c[0] == 0.5
    assert np.isclose(result.advantages[0], result.targets[0] - values[0])
    assert np.isclose(
        result.policy_gradient_advantages[0],
        result.rho[0] * (result.targets[0] - values[0]),
    )
    assert np.isclose(result.targets[0], 2.0 * 1.0 + 0.5 * result.targets[1])
    assert np.isclose(result.targets[0], 4.0)


def test_batch_padding_matches_per_trajectory_reference():
    mdp = make_lag_mdp(horizon=6)
    mu = np.vstack([np.tile([[0.35, 0.65], [0.45, 0.55]], (2, 1)), [[0.5, 0.5]]])
    pi = np.vstack([np.tile([[0.85, 0.15], [0.75, 0.25]], (2, 1)), [[0.5, 0.5]]])
    actor = LaggedPolicyActor(mdp, np.stack([mu, pi]), lag=1, learner_version=1)
    segments = collect_segments(mdp, actor, np.random.default_rng(4), 16, 8)
    values = np.linspace(0, 1, 5)
    reference = []
    for segment in segments:
        bootstrap = 0.0 if segment.terminals[-1] else values[segment.next_states[-1]]
        reference.append(vtrace_targets_trajectory(
            segment.rewards,
            segment.actions,
            pi[segment.states],
            segment.behavior_policy_probs,
            segment.behavior_probs,
            values[segment.states],
            bootstrap,
            segment.terminals,
            segment.truncated,
            discount=0.95,
            rho_bar=2.0,
            c_bar=1.0,
        ))
    batch_inputs = pad_segments(segments, pi, values)
    batch = vtrace_targets_batch(**batch_inputs, discount=0.95, rho_bar=2.0, c_bar=1.0)
    for index, segment in enumerate(segments):
        length = segment.length
        assert np.allclose(batch.targets[:length, index], reference[index].targets)
    assert np.all(np.isnan(batch.targets[batch_inputs["mask"] == False]))


def test_sampling_records_versions_and_real_probabilities():
    mdp = make_lag_mdp(horizon=6)
    mu = np.ones((5, 2)) / 2
    pi = np.vstack([np.tile([[0.8, 0.2], [0.7, 0.3]], (2, 1)), [[0.5, 0.5]]])
    actor = LaggedPolicyActor(mdp, np.stack([mu, pi]), lag=1, learner_version=1)
    segment = collect_segments(mdp, actor, np.random.default_rng(0), 1, 8)[0]
    assert segment.behavior_version == 0
    assert segment.learner_version_at_sample == 1
    assert np.allclose(
        segment.behavior_probs,
        segment.behavior_policy_probs[np.arange(segment.length), segment.actions],
    )


def test_support_report_flags_missing_action():
    pi = np.array([[0.8, 0.2]])
    mu = np.array([[0.0, 1.0]])
    report = support_report(None, pi, mu)
    assert report["unsupported_actions"][0, 0]
    assert report["unsupported_mass"][0] == 0.8
