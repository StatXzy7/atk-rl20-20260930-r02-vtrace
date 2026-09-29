import numpy as np
import pytest

from vtrace_lab.exact import (
    corrected_policy,
    expected_vtrace_target,
    solve_state_values,
    solve_vtrace_fixed,
    support_missing,
    validate_clipping,
    vtrace_fixed_residual,
)
from vtrace_lab.scenarios import bias_variance_bandit, lag_chain_scenario, support_gap_scenario


def test_exact_on_policy_matches_mu_and_pi():
    scenario = lag_chain_scenario()
    values_pi = solve_state_values(scenario.mu, scenario.mdp)
    values_mu = solve_state_values(scenario.mu, scenario.mdp)
    fixed = solve_vtrace_fixed(
        scenario.mu, scenario.mu, scenario.mdp, 1.0, 1.0
    )
    np.testing.assert_allclose(fixed, values_pi, atol=1e-10)
    np.testing.assert_allclose(fixed, values_mu, atol=1e-10)


def test_unclipped_full_support_recovers_pi():
    scenario = lag_chain_scenario()
    fixed = solve_vtrace_fixed(
        scenario.pi, scenario.mu, scenario.mdp, np.inf, 1.0
    )
    np.testing.assert_allclose(fixed, solve_state_values(scenario.pi, scenario.mdp), atol=1e-12)


def test_finite_rho_fixed_point_is_corrected_policy_not_pi():
    scenario = lag_chain_scenario()
    rho_bar = 2.0
    q = corrected_policy(scenario.pi, scenario.mu, rho_bar)
    fixed = solve_vtrace_fixed(
        scenario.pi, scenario.mu, scenario.mdp, rho_bar, 0.5
    )
    np.testing.assert_allclose(fixed, solve_state_values(q, scenario.mdp), atol=1e-12)
    assert np.linalg.norm(fixed - solve_state_values(scenario.pi, scenario.mdp)) > 0.1


def test_c_changes_horizon_target_but_not_fixed_point():
    scenario = lag_chain_scenario()
    values = solve_vtrace_fixed(
        scenario.pi, scenario.mu, scenario.mdp, 2.0, 1.0
    )
    small = expected_vtrace_target(
        values, scenario.pi, scenario.mu, scenario.mdp, 2.0, 0.25, 2
    )
    large = expected_vtrace_target(
        values, scenario.pi, scenario.mu, scenario.mdp, 2.0, 1.0, 2
    )
    np.testing.assert_allclose(small, values, atol=1e-12)
    np.testing.assert_allclose(large, values, atol=1e-12)

    zeros = np.zeros(scenario.mdp.n_states)
    small_zero = expected_vtrace_target(
        zeros, scenario.pi, scenario.mu, scenario.mdp, 2.0, 0.25, 2
    )
    large_zero = expected_vtrace_target(
        zeros, scenario.pi, scenario.mu, scenario.mdp, 2.0, 1.0, 2
    )
    assert not np.allclose(small_zero, large_zero)


def test_fixed_residual_zero_independent_of_c():
    scenario = lag_chain_scenario()
    fixed = solve_vtrace_fixed(
        scenario.pi, scenario.mu, scenario.mdp, 2.0, 0.25
    )
    residual = vtrace_fixed_residual(
        fixed, scenario.pi, scenario.mu, scenario.mdp, 2.0
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_clipping_order_constraint_and_support_gap():
    with pytest.raises(ValueError):
        validate_clipping(1.0, 0.0)
    with pytest.raises(ValueError):
        validate_clipping(0.5, 1.0)
    scenario = support_gap_scenario()
    assert support_missing(scenario.pi, scenario.mu)[0].any()
    with pytest.raises(ValueError):
        corrected_policy(scenario.pi, scenario.mu, 10.0)


def test_bandit_counterexample_has_lower_variance_but_fixed_bias():
    scenario = bias_variance_bandit()
    v_pi = solve_state_values(scenario.pi, scenario.mdp)[0]
    fixed_clipped = solve_vtrace_fixed(
        scenario.pi, scenario.mu, scenario.mdp, 2.0, 1.0
    )[0]
    fixed_unclipped = solve_vtrace_fixed(
        scenario.pi, scenario.mu, scenario.mdp, np.inf, 1.0
    )[0]
    v_q = solve_state_values(
        corrected_policy(scenario.pi, scenario.mu, 2.0), scenario.mdp
    )[0]

    assert fixed_clipped == pytest.approx(v_q)
    assert fixed_unclipped == pytest.approx(v_pi)
    assert abs(fixed_clipped - v_pi) > 0.05


def test_fixed_base_target_and_repeated_operator_are_different_objects():
    scenario = lag_chain_scenario()
    mdp = scenario.mdp
    fixed = solve_vtrace_fixed(
        scenario.pi, scenario.mu, mdp, 2.0, 2.0
    )

    fixed_base = expected_vtrace_target(
        fixed, scenario.pi, scenario.mu, mdp, 2.0, 2.0, 3
    )
    np.testing.assert_allclose(fixed_base, fixed, atol=1e-12)

    values = np.zeros(mdp.n_states)
    for _ in range(500):
        values = expected_vtrace_target(
            values, scenario.pi, scenario.mu, mdp, 2.0, 2.0, horizon=1
        )
    np.testing.assert_allclose(values, fixed, atol=1e-10)

    one_shot = expected_vtrace_target(
        np.zeros(mdp.n_states),
        scenario.pi,
        scenario.mu,
        mdp,
        2.0,
        2.0,
        horizon=3,
    )
    np.testing.assert_allclose(one_shot[:3], [0.081, 0.09, 0.1], atol=1e-12)
    assert np.linalg.norm(one_shot - fixed) > 0.4
