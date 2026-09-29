import numpy as np

from vtrace_lab.exact import clipped_policy, policy_value, vtrace_expected_operator
from vtrace_lab.mdp import make_lag_mdp, make_one_shot_mdp


def test_linear_policy_value_matches_bellman_equation():
    mdp = make_lag_mdp(horizon=6, slip=0.2)
    pi = np.vstack([np.tile([[0.8, 0.2], [0.7, 0.3]], (2, 1)), [[0.5, 0.5]]])
    v = policy_value(mdp, pi, 0.95)
    p = np.einsum("sa,sat->st", pi, mdp.transition)
    r = np.einsum("sa,sa->s", pi, mdp.reward)
    assert np.allclose(v, r + 0.95 * p @ v)
    assert v[-1] == 0.0


def test_clipped_policy_formula_and_pi_value_when_rho_is_infinite():
    pi = np.array([[0.8, 0.2]])
    mu = np.array([[0.2, 0.8]])
    clipped = clipped_policy(pi, mu, 1.0)
    expected = np.minimum(pi, mu)
    expected /= expected.sum()
    assert np.allclose(clipped, expected)
    assert np.isclose(policy_value(make_one_shot_mdp(), pi, 0.95)[0], 0.6)


def test_expected_operator_fixed_point_is_clipped_policy_value():
    mdp = make_lag_mdp(horizon=6)
    mu = np.vstack([np.tile([[0.35, 0.65], [0.45, 0.55]], (2, 1)), [[0.5, 0.5]]])
    pi = np.vstack([np.tile([[0.85, 0.15], [0.75, 0.25]], (2, 1)), [[0.5, 0.5]]])
    v = np.zeros(mdp.n_states)
    for _ in range(10000):
        nv = vtrace_expected_operator(
            mdp, pi, mu, v, discount=0.95, rho_bar=2.0, c_bar=0.75
        )
        if np.max(np.abs(nv - v)) < 1e-11:
            v = nv
            break
        v = nv
    exact = policy_value(mdp, clipped_policy(pi, mu, 2.0), 0.95)
    assert np.allclose(v, exact, atol=1e-8)


def test_c_does_not_change_fixed_point():
    mdp = make_lag_mdp(horizon=6)
    mu = np.ones((5, 2)) / 2
    pi = np.vstack([np.tile([[0.9, 0.1], [0.8, 0.2]], (2, 1)), [[0.5, 0.5]]])
    values = [
        policy_value(mdp, clipped_policy(pi, mu, 1.0), 0.95),
    ]
    for c_bar in (0.25, 0.5, 1.0):
        v = np.zeros(mdp.n_states)
        for _ in range(10000):
            nv = vtrace_expected_operator(mdp, pi, mu, v, 0.95, 1.0, c_bar)
            if np.max(np.abs(nv - v)) < 1e-11:
                v = nv
                break
            v = nv
        values.append(v)
    assert all(np.allclose(values[0], other, atol=1e-8) for other in values[1:])
