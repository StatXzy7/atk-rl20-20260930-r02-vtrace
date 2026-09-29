from __future__ import annotations

import numpy as np

from .mdp import FiniteMDP


def _matrix(mdp: FiniteMDP, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    policy = np.asarray(policy, dtype=np.float64)
    p_pi = np.einsum("sa,sat->st", policy, mdp.transition)
    r_pi = np.einsum("sa,sa->s", policy, mdp.reward)
    p_pi[list(mdp.terminal_states), :] = 0.0
    r_pi[list(mdp.terminal_states)] = 0.0
    return p_pi, r_pi


def policy_value(
    mdp: FiniteMDP,
    policy: np.ndarray,
    discount: float,
) -> np.ndarray:
    """Solve (I - gamma P^sigma) V = r^sigma exactly."""
    if not 0.0 <= discount < 1.0:
        raise ValueError("discount must be in [0, 1)")
    policy = np.asarray(policy, dtype=np.float64)
    p_sigma, r_sigma = _matrix(mdp, policy)
    values = np.linalg.solve(np.eye(mdp.n_states) - discount * p_sigma, r_sigma)
    values[list(mdp.terminal_states)] = 0.0
    return values


def support_report(
    mdp: FiniteMDP | None,
    target_policy: np.ndarray,
    behavior_policy: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    pi = np.asarray(target_policy, dtype=np.float64)
    mu = np.asarray(behavior_policy, dtype=np.float64)
    unsupported_actions = (pi > 0.0) & (mu <= 0.0)
    supported_mass = np.sum(pi, axis=1, where=mu > 0.0)
    unreachable_states = np.all(mu <= 0.0, axis=1)
    return {
        "unsupported_actions": unsupported_actions,
        "unsupported_mass": np.sum(pi, axis=1, where=unsupported_actions),
        "supported_mass": supported_mass,
        "unreachable_states": unreachable_states,
        "full_action_support": bool(not np.any(unsupported_actions)),
    }


def clipped_policy(
    target_policy: np.ndarray,
    behavior_policy: np.ndarray,
    rho_bar: float,
) -> np.ndarray:
    """Return pi_rho(a|s) proportional to pi(a|s) min(1, rho_bar mu(a|s)/pi(a|s)).

    Under full support this is proportional to min(pi, rho_bar mu). It is a
    normalized probability distribution; no epsilon is added to mu.
    """
    if not rho_bar > 0.0:
        raise ValueError("rho_bar must be positive")
    pi = np.asarray(target_policy, dtype=np.float64)
    mu = np.asarray(behavior_policy, dtype=np.float64)
    if pi.shape != mu.shape:
        raise ValueError("policies must have the same shape")

    weighted = np.minimum(pi, rho_bar * mu)
    unsupported = (pi > 0.0) & (mu <= 0.0)
    weighted = np.where(unsupported, 0.0, weighted)
    denominator = weighted.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0.0):
        states = np.flatnonzero(denominator[:, 0] <= 0.0)
        raise ValueError(f"clipped policy is undefined in states {states.tolist()}")
    return weighted / denominator


def vtrace_expected_operator(
    mdp: FiniteMDP,
    target_policy: np.ndarray,
    behavior_policy: np.ndarray,
    values: np.ndarray,
    discount: float,
    rho_bar: float,
    c_bar: float,
) -> np.ndarray:
    """Apply the asymptotic expected V-trace operator once.

    T V = V + (I - gamma W_c P_mu)^{-1}
          diag(z) (r_pi_rho + gamma P_rho V - V),

    where diagonal W_c contains E_mu[min(c_bar, pi/mu)] in each state.
    """
    if not 0.0 < c_bar <= rho_bar:
        raise ValueError("require 0 < c_bar <= rho_bar")
    pi = np.asarray(target_policy, dtype=np.float64)
    mu = np.asarray(behavior_policy, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    ratio = np.divide(
        pi,
        mu,
        out=np.full_like(pi, np.nan),
        where=mu > 0.0,
    )
    rho = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)

    rho_weight = np.nansum(mu * rho, axis=1)
    if np.any(rho_weight <= 0.0):
        states = np.flatnonzero(rho_weight <= 0.0).tolist()
        raise ValueError(f"V-trace operator has zero supported action mass in states {states}")
    c_weight = np.nansum(mu * c, axis=1)
    c_weight = np.minimum(c_weight, rho_weight)

    p_rho = np.einsum("sa,sat->st", mu * rho, mdp.transition) / rho_weight[:, None]
    p_mu = np.einsum("sa,sat->st", mu, mdp.transition)
    r_rho = np.einsum("sa,sa->s", mu * rho, mdp.reward) / rho_weight

    terminal = list(mdp.terminal_states)
    p_rho[terminal, :] = 0.0
    p_mu[terminal, :] = 0.0
    r_rho[terminal] = 0.0
    c_weight[terminal] = 0.0

    local_bellman_error = rho_weight * (r_rho + discount * p_rho @ v - v)
    a = np.eye(mdp.n_states) - discount * c_weight[:, None] * p_mu
    correction = np.linalg.solve(a, local_bellman_error)
    result = v + correction
    result[terminal] = 0.0
    return result
