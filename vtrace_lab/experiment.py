from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .exact import (
    corrected_policy,
    expected_vtrace_target,
    one_step_terminal_target_distribution,
    raw_rho_weights,
    solve_state_values,
    solve_vtrace_fixed,
    support_missing,
)
from .sampler import PolicyLagSampler, collate_segments
from .scenarios import (
    bias_variance_bandit,
    c_trace_chain,
    lag_chain_scenario,
    support_gap_scenario,
)
from .vtrace import vtrace_batch


RHO_BAR = 2.0
C_BAR = 0.5


def exact_case() -> dict:
    scenario = lag_chain_scenario()
    mdp = scenario.mdp
    values_mu = solve_state_values(scenario.mu, mdp)
    values_pi = solve_state_values(scenario.pi, mdp)
    q = corrected_policy(scenario.pi, scenario.mu, RHO_BAR)
    values_q = solve_state_values(q, mdp)
    fixed_finite = solve_vtrace_fixed(
        scenario.pi, scenario.mu, mdp, RHO_BAR, C_BAR
    )
    fixed_unclipped = solve_vtrace_fixed(
        scenario.pi, scenario.mu, mdp, np.inf, 1.0
    )
    return {
        "state_0_values": {
            "mu": float(values_mu[0]),
            "pi": float(values_pi[0]),
            "rho_clipped_corrected_policy": float(values_q[0]),
            "vtrace_fixed_rho_2_c_0.5": float(fixed_finite[0]),
            "vtrace_fixed_rho_inf_c_1": float(fixed_unclipped[0]),
        },
        "fixed_point_bias_vq_minus_vpi_state_0": float(
            fixed_finite[0] - values_pi[0]
        ),
        "corrected_policy_state_2": q[2].tolist(),
    }


def support_gap_case() -> dict:
    scenario = support_gap_scenario()
    values_pi = solve_state_values(scenario.pi, scenario.mdp)
    values_mu = solve_state_values(scenario.mu, scenario.mdp)
    raw = raw_rho_weights(scenario.pi, scenario.mu, RHO_BAR)
    gap_actions = np.argwhere(support_missing(scenario.pi, scenario.mu)).tolist()
    singular = False
    try:
        solve_vtrace_fixed(
            scenario.pi, scenario.mu, scenario.mdp, RHO_BAR, C_BAR
        )
    except np.linalg.LinAlgError:
        singular = True
    return {
        "mu_value_state_0": float(values_mu[0]),
        "pi_value_state_0": float(values_pi[0]),
        "raw_clipped_weights_state_0": raw[0].tolist(),
        "missing_state_actions": gap_actions,
        "normalized_corrected_policy_exists": False,
        "exact_fixed_linear_system_singular": singular,
        "interpretation": (
            "The unsupported target action is never observed; finite IS weights "
            "cannot identify it, and adding epsilon would invent a probability."
        ),
    }


def finite_sample_case(seeds: int, segment_counts: list[int]) -> dict:
    scenario = lag_chain_scenario()
    mdp = scenario.mdp
    values_q = solve_vtrace_fixed(
        scenario.pi, scenario.mu, mdp, RHO_BAR, C_BAR
    )
    sampler = PolicyLagSampler(
        mdp, scenario.versions, lag_steps=len(scenario.versions) - 1, initial_state=None
    )
    results = []
    for horizon in (1, 3):
        for count in segment_counts:
            seed_rmse = []
            seed_bias = []
            actual_transitions = []
            for seed in range(seeds):
                rng = np.random.default_rng(10_000 + seed)
                segments = [
                    sampler.sample(
                        horizon,
                        rng,
                        initial_state=int(rng.choice(mdp.nonterminal_states)),
                    )
                    for _ in range(count)
                ]
                batch = collate_segments(segments)
                bootstrap = values_q[batch["bootstrap_state"]]
                values = values_q[batch["states"]]
                output = vtrace_batch(
                    batch["rewards"],
                    batch["behavior_probs"],
                    batch["target_probs_at_sample"],
                    values,
                    bootstrap,
                    mdp.discount,
                    RHO_BAR,
                    C_BAR,
                    batch["done"],
                    batch["truncated"],
                    batch["valid"],
                )
                estimates = {}
                for state in mdp.nonterminal_states:
                    mask = batch["valid"] & (batch["states"] == state)
                    if np.any(mask):
                        estimates[state] = float(np.mean(output.vs[mask]))
                states = np.asarray(sorted(estimates), dtype=np.int64)
                errors = np.asarray([estimates[int(s)] - values_q[int(s)] for s in states])
                seed_rmse.append(float(np.sqrt(np.mean(errors**2))))
                seed_bias.append(float(np.mean(errors)))
                actual_transitions.append(int(batch["valid"].sum()))
            results.append(
                {
                    "horizon": horizon,
                    "requested_segments": count,
                    "actual_transitions_mean": float(np.mean(actual_transitions)),
                    "rmse_mean": float(np.mean(seed_rmse)),
                    "rmse_std": float(np.std(seed_rmse)),
                    "bias_mean": float(np.mean(seed_bias)),
                    "bias_std": float(np.std(seed_bias)),
                    "seeds": seeds,
                }
            )
    return {"bootstrap": "exact V_rho_c", "results": results}


def bias_variance_case() -> dict:
    scenario = bias_variance_bandit()
    mdp = scenario.mdp
    values_pi = solve_state_values(scenario.pi, mdp)[0]
    values_mu = solve_state_values(scenario.mu, mdp)[0]
    fixed_clipped = solve_vtrace_fixed(
        scenario.pi, scenario.mu, mdp, RHO_BAR, C_BAR
    )[0]
    q = corrected_policy(scenario.pi, scenario.mu, RHO_BAR)
    values_q = solve_state_values(q, mdp)[0]

    at_zero = {}
    at_fixed = {}
    for label, rho, reference in (
        ("rho_2", RHO_BAR, fixed_clipped),
        ("rho_inf", np.inf, values_pi),
    ):
        targets_zero, mean_zero, variance_zero = one_step_terminal_target_distribution(
            scenario.pi[0], scenario.mu[0], mdp.rewards[0], rho, 0.0
        )
        _, mean_fixed, variance_fixed = one_step_terminal_target_distribution(
            scenario.pi[0], scenario.mu[0], mdp.rewards[0], rho, reference
        )
        at_zero[label] = {
            "mean": mean_zero,
            "variance": variance_zero,
            "mse_to_Vpi": float(
                np.dot(
                    scenario.mu[0],
                    (targets_zero - values_pi) ** 2,
                )
            ),
        }
        at_fixed[label] = {
            "mean": mean_fixed,
            "variance": variance_fixed,
            "reference": float(reference),
        }

    return {
        "Vmu": values_mu,
        "Vpi": values_pi,
        "V_clipped_fixed": values_q,
        "fixed_bias_to_pi": values_q - values_pi,
        "one_step_from_zero": at_zero,
        "one_step_at_own_fixed_reference": at_fixed,
        "conclusion": (
            "rho_bar=2 lowers one-step target variance versus rho_bar=inf, "
            "but its fixed point is V_q, not V_pi."
        ),
    }


def c_trace_case(iterations: int = 8) -> dict:
    scenario = c_trace_chain(length=6)
    mdp = scenario.mdp
    fixed = solve_state_values(scenario.pi, mdp)
    output = {}
    for c_bar in (0.25, 1.0):
        values = np.zeros(mdp.n_states)
        errors = []
        horizon = mdp.n_states - 1
        for _ in range(iterations):
            values = expected_vtrace_target(
                values,
                scenario.pi,
                scenario.mu,
                mdp,
                rho_bar=1.0,
                c_bar=c_bar,
                horizon=horizon,
            )
            errors.append(float(np.linalg.norm(values[mdp.nonterminal_states] - fixed[mdp.nonterminal_states])))
        output[f"c_{c_bar:g}"] = {
            "initial_state_target_after_first_batch": None,
            "errors_after_each_batch": errors,
            "fixed_state_0": float(fixed[0]),
        }
        first = expected_vtrace_target(
            np.zeros(mdp.n_states),
            scenario.pi,
            scenario.mu,
            mdp,
            1.0,
            c_bar,
            horizon,
        )
        output[f"c_{c_bar:g}"]["initial_state_target_after_first_batch"] = float(first[0])
    return output


def run(seeds: int) -> dict:
    return {
        "configuration": {
            "seeds": seeds,
            "rho_bar": RHO_BAR,
            "c_bar": C_BAR,
            "rule": "0 < c_bar <= rho_bar",
        },
        "exact_values": exact_case(),
        "finite_sample_error": finite_sample_case(
            seeds=seeds, segment_counts=[64, 256, 1024]
        ),
        "bias_variance_counterexample": bias_variance_case(),
        "support_gap": support_gap_case(),
        "c_trace_speed_and_fixed_point": c_trace_case(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V-trace policy-lag experiments")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/experiment.json"))
    args = parser.parse_args()
    results = run(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
