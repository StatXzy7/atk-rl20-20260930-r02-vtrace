from __future__ import annotations

import numpy as np

from .mdp import FiniteMDP, validate_policy


def validate_clipping(rho_bar: float, c_bar: float) -> None:
    if not (c_bar > 0.0 and c_bar <= rho_bar):
        raise ValueError("clipping must satisfy 0 < c_bar <= rho_bar")
    if not np.isfinite(rho_bar) and rho_bar != np.inf:
        raise ValueError("rho_bar must be positive finite or np.inf")
    if not np.isfinite(c_bar):
        raise ValueError("c_bar must be positive and finite")


def solve_state_values(policy: np.ndarray, mdp: FiniteMDP) -> np.ndarray:
    """Solve V = R_pi + gamma P_pi V exactly for all nonterminal states."""

    policy = validate_policy(policy, mdp)
    states = mdp.nonterminal_states
    p_next = mdp.transitions[:, :, states]
    p_policy = policy[states]

    reward = np.einsum("sa,sa->s", p_policy, mdp.rewards[states])
    transition = np.einsum("sa,san->sn", p_policy, p_next[states])
    values = np.linalg.solve(np.eye(states.size) - mdp.discount * transition, reward)

    full = np.zeros(mdp.n_states, dtype=np.float64)
    full[states] = values
    return full


def raw_rho_weights(
    pi: np.ndarray, mu: np.ndarray, rho_bar: float
) -> np.ndarray:
    """Return min(pi(a|s), rho_bar mu(a|s)); no epsilon is added anywhere."""

    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    weights = np.zeros_like(pi)
    supported = mu > 0.0
    weights[supported] = np.minimum(pi[supported], rho_bar * mu[supported])
    return weights


def corrected_policy(
    pi: np.ndarray, mu: np.ndarray, rho_bar: float
) -> np.ndarray:
    """Normalized policy induced by finite-horizon rho clipping."""

    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if np.any(pi < 0.0) or np.any(mu < 0.0):
        raise ValueError("policies must be nonnegative")
    weights = raw_rho_weights(pi, mu, rho_bar)
    normalizer = weights.sum(axis=-1, keepdims=True)
    if np.any(normalizer <= 0.0):
        raise ValueError("behavior policy has no probability mass in a state")
    return weights / normalizer


def importance_ratios(pi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Exact pi/mu with infinity for target-supported but behavior-zero actions."""

    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    ratios = np.zeros_like(pi, dtype=np.float64)
    positive_mu = mu > 0.0
    ratios[positive_mu] = pi[positive_mu] / mu[positive_mu]
    ratios[~positive_mu & (pi > 0.0)] = np.inf
    return ratios


def clipped_importance_terms(
    pi: np.ndarray, mu: np.ndarray, rho_bar: float, c_bar: float
) -> tuple[np.ndarray, np.ndarray]:
    validate_clipping(rho_bar, c_bar)
    ratios = importance_ratios(pi, mu)
    rho = np.minimum(rho_bar, ratios)
    c = np.minimum(c_bar, ratios)
    return rho, c


def support_missing(pi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (np.asarray(pi) > 0.0) & (np.asarray(mu) == 0.0)


def solve_vtrace_fixed(
    pi: np.ndarray,
    mu: np.ndarray,
    mdp: FiniteMDP,
    rho_bar: float,
    c_bar: float,
) -> np.ndarray:
    """Solve the expected infinite-horizon V-trace fixed point exactly."""

    validate_clipping(rho_bar, c_bar)
    pi = validate_policy(pi, mdp)
    mu = validate_policy(mu, mdp)
    states = mdp.nonterminal_states
    p_next = mdp.transitions[:, :, states]
    rho, _ = clipped_importance_terms(
        pi[states], mu[states], rho_bar, c_bar
    )
    w_rho = mu[states] * rho

    rho_mass = w_rho.sum(axis=1)
    kernel = np.einsum("sa,san->sn", w_rho, p_next[states])
    matrix = np.diag(rho_mass) - mdp.discount * kernel
    rhs = np.einsum("sa,sa->s", w_rho, mdp.rewards[states])

    fixed = np.zeros(mdp.n_states, dtype=np.float64)
    fixed[states] = np.linalg.solve(matrix, rhs)
    return fixed


def vtrace_fixed_residual(
    values: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    mdp: FiniteMDP,
    rho_bar: float,
) -> np.ndarray:
    """Expected unnormalized one-step V-trace residual at each state."""

    values = np.asarray(values, dtype=np.float64)
    states = mdp.nonterminal_states
    next_values = mdp.transitions[:, :, states] @ values[states]
    action_residual = (
        mdp.rewards
        + mdp.discount * next_values
        - values[:, None]
    )
    weights = raw_rho_weights(pi, mu, rho_bar)
    return np.einsum("sa,sa->s", weights[states], action_residual[states])


def expected_vtrace_target(
    values: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    mdp: FiniteMDP,
    rho_bar: float,
    c_bar: float,
    horizon: int,
) -> np.ndarray:
    """
    Expected V-trace target under trajectories sampled from mu.

    A horizon of zero denotes the sampling cutoff and bootstraps from ``values``.
    True terminal transitions naturally contribute no next-state value.
    """

    validate_clipping(rho_bar, c_bar)
    pi = validate_policy(pi, mdp)
    mu = validate_policy(mu, mdp)
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")

    states = mdp.nonterminal_states
    base = np.zeros(mdp.n_states, dtype=np.float64)
    base[states] = np.asarray(values, dtype=np.float64)[states]
    target = base.copy()

    ratios = importance_ratios(pi[states], mu[states])
    rho = np.minimum(rho_bar, ratios)
    c = np.minimum(c_bar, ratios)
    p_next = mdp.transitions[:, :, states]

    action_values = (
        mdp.rewards[states]
        + mdp.discount * np.einsum("san,n->sa", p_next[states], base[states])
    )
    for _ in range(horizon):
        delta = np.einsum(
            "sa,sa->s",
            mu[states] * rho,
            action_values - base[states, None],
        )
        future_difference = np.einsum(
            "sa,san,n->s",
            mu[states] * c,
            p_next[states],
            target[states] - base[states],
        )
        updated = base.copy()
        updated[states] = base[states] + delta + mdp.discount * future_difference
        target = updated
    return target


def one_step_terminal_target_distribution(
    pi: np.ndarray,
    mu: np.ndarray,
    rewards: np.ndarray,
    rho_bar: float,
    reference_value: float,
) -> tuple[np.ndarray, float, float]:
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    ratios = importance_ratios(pi, mu)
    rho = np.minimum(rho_bar, ratios)
    targets = reference_value + rho * (rewards - reference_value)
    mean = float(np.dot(mu, targets))
    variance = float(np.dot(mu, (targets - mean) ** 2))
    return targets, mean, variance
