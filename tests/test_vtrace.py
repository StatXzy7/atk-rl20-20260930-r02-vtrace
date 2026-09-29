import numpy as np
import pytest
import torch

from vtrace_lab.sampler import PolicyLagSampler, collate_segments
from vtrace_lab.scenarios import lag_chain_scenario, support_gap_scenario
from vtrace_lab.vtrace import vtrace_batch, vtrace_trajectory


def test_reference_terminal_bootstrap():
    rewards = np.array([1.0, 1.0])
    mu = np.array([0.5, 0.5])
    pi = np.array([1.0, 0.5])
    values = np.array([0.0, 0.0])
    done = np.array([True, True])
    output = vtrace_trajectory(
        rewards, mu, pi, values, 0.0, 0.9, 2.0, 1.0, done=done
    )
    np.testing.assert_allclose(output.vs[0], 2.0)
    np.testing.assert_allclose(output.advantages[1], 1.0)


def test_reference_truncation_bootstraps():
    output = vtrace_trajectory(
        rewards=np.array([1.0]),
        behavior_probs=np.array([0.5]),
        target_probs=np.array([0.5]),
        values=np.array([2.0]),
        bootstrap_value=3.0,
        discount=0.9,
        rho_bar=1.0,
        c_bar=1.0,
        truncated=np.array([True]),
    )
    assert output.vs[0] == pytest.approx(1.0 + 0.9 * 3.0)


def test_zero_behavior_prob_with_positive_target_is_rejected_no_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        vtrace_trajectory(
            rewards=np.array([1.0]),
            behavior_probs=np.array([0.0]),
            target_probs=np.array([0.5]),
            values=np.zeros(1),
            bootstrap_value=0.0,
            discount=0.9,
            rho_bar=1.0,
            c_bar=1.0,
        )


def test_batch_matches_reference_with_padding_and_truncation():
    scenario = lag_chain_scenario()
    sampler = PolicyLagSampler(
        scenario.mdp, scenario.versions, lag_steps=2, initial_state=0
    )
    rng = np.random.default_rng(7)
    segments = [sampler.sample(length, rng) for length in (3, 5)]
    batch = collate_segments(segments)
    values_grid = np.arange(batch["rewards"].size, dtype=float).reshape(
        batch["rewards"].shape
    )
    bootstrap = values_grid.max(axis=1) + 10.0
    batched = vtrace_batch(
        batch["rewards"],
        batch["behavior_probs"],
        batch["target_probs_at_sample"],
        values_grid,
        bootstrap,
        scenario.mdp.discount,
        2.0,
        0.75,
        batch["done"],
        batch["truncated"],
        batch["valid"],
    )

    for row, segment in enumerate(segments):
        actual = segment.actual_length
        reference = vtrace_trajectory(
            segment.rewards[:actual],
            segment.behavior_probs[:actual],
            segment.target_probs_at_sample[:actual],
            values_grid[row, :actual],
            bootstrap[row],
            scenario.mdp.discount,
            2.0,
            0.75,
            segment.done[:actual],
            segment.truncated[:actual],
        )
        np.testing.assert_allclose(batched.vs[row, :actual], reference.vs)
        assert np.all(batched.vs[row, actual:] == 0.0)


def test_torch_targets_stop_gradient_but_actor_logprob_gradients_flow():
    from vtrace_lab.learner import actor_loss, critic_loss, vtrace_torch

    rewards = torch.tensor([[1.0]])
    behavior = torch.tensor([[0.2]], requires_grad=True)
    probs = torch.tensor([[0.8, 0.2]])
    log_probs = torch.log(probs).requires_grad_(True)
    values = torch.zeros(1, 1, requires_grad=True)
    bootstrap = torch.zeros(1)
    valid = torch.ones(1, 1, dtype=torch.bool)
    output = vtrace_torch(
        rewards,
        behavior,
        probs.detach().gather(1, torch.zeros(1, 1, dtype=torch.long)),
        values,
        bootstrap,
        0.9,
        2.0,
        1.0,
        torch.zeros(1, 1, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
        valid,
    )
    policy_loss = actor_loss(log_probs[:, :1], output.advantages)
    policy_loss.backward()
    assert behavior.grad is None
    assert log_probs.grad is not None
    assert values.grad is None

    values2 = torch.zeros(1, 1, requires_grad=True)
    critic = critic_loss(values2, output.vs, valid)
    critic.backward()
    assert values2.grad is not None


def test_sampler_records_actual_probabilities_versions_and_truncation():
    scenario = lag_chain_scenario()
    sampler = PolicyLagSampler(
        scenario.mdp, scenario.versions, lag_steps=2, initial_state=0
    )
    segment = sampler.sample(1, np.random.default_rng(1))
    assert segment.behavior_version == 0
    assert segment.target_version == 2
    assert segment.truncated[0]
    assert segment.actual_length == 1
    assert segment.bootstrap_state == 1
    assert segment.behavior_probs[0] == pytest.approx(
        scenario.mu[0, segment.actions[0]]
    )
    assert segment.target_probs_at_sample[0] == pytest.approx(
        scenario.pi[0, segment.actions[0]]
    )


def test_real_terminal_bootstrap_is_zero_and_padding_is_masked():
    scenario = lag_chain_scenario()
    sampler = PolicyLagSampler(
        scenario.mdp, scenario.versions, lag_steps=0, initial_state=0
    )
    segment = sampler.sample(5, np.random.default_rng(0))
    batch = collate_segments([segment])
    assert segment.actual_length == 3
    assert batch["done"][0, 2]
    assert segment.bootstrap_state == scenario.mdp.terminal_state
    assert not batch["valid"][0, 3:].any()


def test_actor_loss_ignores_padding():
    from vtrace_lab.learner import actor_loss

    log_probs = torch.tensor([[1.0, 100.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 100.0]])
    valid = torch.tensor([[True, False]])
    loss = actor_loss(log_probs, advantages, valid)
    loss.backward()
    assert loss.item() == -1.0
    assert log_probs.grad.tolist() == [[-1.0, 0.0]]


def test_extreme_lag_uses_oldest_version_and_keeps_actual_probabilities():
    scenario = lag_chain_scenario()
    lag = len(scenario.versions) - 1
    sampler = PolicyLagSampler(
        scenario.mdp, scenario.versions, lag_steps=lag, initial_state=0
    )
    assert sampler.behavior_policy is scenario.versions[0]
    segment = sampler.sample(1, np.random.default_rng(3))
    assert segment.behavior_version == 0
    assert segment.target_version == len(scenario.versions) - 1
    action = int(segment.actions[0])
    assert segment.behavior_probs[0] == scenario.mu[0, action]
    assert segment.target_probs_at_sample[0] == scenario.pi[0, action]


def test_torch_rejects_support_gap_without_epsilon():
    from vtrace_lab.learner import vtrace_torch

    scenario = support_gap_scenario()
    sampler = PolicyLagSampler(
        scenario.mdp, scenario.versions, lag_steps=1, initial_state=0
    )
    segment = sampler.sample(1, np.random.default_rng(0))
    assert segment.behavior_probs[0] == 1.0
    assert segment.target_probs_at_sample[0] == 0.0
    # The recorded sampled action is itself unsupported by pi. A real pi-positive,
    # mu-zero action cannot be sampled, so support checking is also covered at
    # probability-array boundaries below.
    with pytest.raises(ValueError, match="epsilon"):
        vtrace_torch(
            rewards=torch.zeros(1, 1),
            behavior_probs=torch.zeros(1, 1),
            target_probs=torch.ones(1, 1),
            values=torch.zeros(1, 1),
            bootstrap_values=torch.zeros(1),
            discount=0.9,
            rho_bar=1.0,
            c_bar=1.0,
            done=torch.zeros(1, 1, dtype=torch.bool),
            truncated=torch.ones(1, 1, dtype=torch.bool),
            valid=torch.ones(1, 1, dtype=torch.bool),
        )
