import numpy as np
import pytest

from vtrace_lab.experiments import (
    batch_matches_trajectory,
    c_tradeoff,
    make_scenarios,
    run_policy_lag_training,
    run_scenario,
)


def test_pi_equals_mu_and_unclipped_lag_have_small_finite_sample_error():
    scenarios = {scenario.name: scenario for scenario in make_scenarios()}
    equal = run_scenario(scenarios["pi_equals_mu"], 501, n_episodes=500, max_steps=8)
    unclipped = run_scenario(scenarios["unclipped_lag"], 502, n_episodes=500, max_steps=8)
    assert equal["fixed_point_bias_at_initial"] == 0.0
    assert equal["rmse_to_target_fixed_point"] < 0.08
    assert unclipped["rmse_to_target_fixed_point"] < 0.08
    assert unclipped["transitions"] > 0
    assert equal["true_terminations"] + equal["truncations"] == equal["episodes"]


def test_clipped_counterexample_reduces_variance_with_target_fixed_point_bias():
    unclipped_rows = []
    clipped_rows = []
    scenarios = {scenario.name: scenario for scenario in make_scenarios()}
    unclipped_scenario = type(scenarios["clipped_counterexample"])(
        "counterexample_unclipped",
        scenarios["clipped_counterexample"].behavior,
        scenarios["clipped_counterexample"].target,
        np.inf,
        np.inf,
        True,
    )
    for seed in range(12):
        unclipped = run_scenario(
            unclipped_scenario, 700 + seed, 400, 1
        )
        clipped = run_scenario(
            scenarios["clipped_counterexample"], 800 + seed, 400, 1
        )
        unclipped_rows.append(unclipped["initial_mean"])
        clipped_rows.append(clipped["initial_mean"])
    assert np.var(unclipped_rows, ddof=1) > np.var(clipped_rows, ddof=1)
    assert np.mean(clipped_rows) == pytest.approx(0.0, abs=0.05)
    assert abs(np.mean(unclipped_rows) - 0.57) < 0.10


def test_clipped_counterexample_exact_values_are_different():
    scenario = make_scenarios()[2]
    row = run_scenario(scenario, 1, n_episodes=100, max_steps=1)
    assert np.isclose(row["exact_target"][0], 0.6)
    assert np.isclose(row["exact_clipped"][0], 0.0)
    assert abs(row["fixed_point_bias_at_initial"]) > 0.4


def test_extreme_lag_runs_and_reports_large_ratios():
    scenario = {scenario.name: scenario for scenario in make_scenarios()}["extreme_lag_chain"]
    row = run_scenario(scenario, 900, n_episodes=200, max_steps=8)
    assert row["transitions"] >= row["episodes"]
    assert row["rmse_to_target_fixed_point"] > row["rmse_to_clipped_fixed_point"]


def test_missing_support_is_reported_not_epsilon_corrected():
    scenario = {scenario.name: scenario for scenario in make_scenarios()}["missing_support_one_shot"]
    row = run_scenario(scenario, 901, n_episodes=100, max_steps=1)
    assert not row["full_action_support"]
    assert row["unsupported_mass"][0] == pytest.approx(0.8)


def test_batch_reference_and_c_tradeoff():
    equivalence = batch_matches_trajectory()
    assert equivalence["max_target_gap"] < 1e-12
    rows = c_tradeoff()
    fixed_values = [row["initial_fixed_value"] for row in rows]
    assert np.allclose(fixed_values, fixed_values[0], atol=1e-8)
    assert rows[0]["iterations_to_1e-10"] >= rows[-1]["iterations_to_1e-10"]


def test_policy_lag_training_uses_versioned_snapshots():
    history = run_policy_lag_training(iterations=4, episodes_per_iteration=8)
    assert [row["behavior_version"] for row in history] == [0, 0, 0, 1]
    assert [row["learner_version"] for row in history] == [0, 1, 2, 3]
