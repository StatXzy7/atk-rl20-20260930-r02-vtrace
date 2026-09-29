from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .batching import pad_segments
from .exact import clipped_policy, policy_value, support_report, vtrace_expected_operator
from .learning import train_step
from .mdp import make_lag_mdp, make_one_shot_mdp
from .policies import softmax_policy
from .sampling import LaggedPolicyActor, collect_segments
from .vtrace import vtrace_for_segments, vtrace_targets_batch

DISCOUNT = 0.95


@dataclass(frozen=True)
class Scenario:
    name: str
    behavior: np.ndarray
    target: np.ndarray
    rho_bar: float
    c_bar: float
    one_shot: bool = False


def chain_policies():
    uniform = np.ones((5, 2)) / 2.0
    lagged = np.vstack([
        np.tile(np.array([[0.35, 0.65], [0.45, 0.55]]), (2, 1)),
        uniform[-1:],
    ])
    target = np.vstack([
        np.tile(np.array([[0.85, 0.15], [0.75, 0.25]]), (2, 1)),
        uniform[-1:],
    ])
    extreme = np.vstack([
        np.tile(np.array([[0.97, 0.03], [0.95, 0.05]]), (2, 1)),
        uniform[-1:],
    ])
    missing_behavior = uniform.copy()
    missing_behavior[0] = np.array([0.0, 1.0])
    return uniform, lagged, target, extreme, missing_behavior


def make_scenarios():
    _, lagged, target, extreme, _ = chain_policies()
    one_mu = np.array([[0.20, 0.80], [0.50, 0.50]])
    one_pi = np.array([[0.80, 0.20], [0.50, 0.50]])
    return [
        Scenario("pi_equals_mu", lagged, lagged, np.inf, np.inf),
        Scenario("unclipped_lag", lagged, target, np.inf, np.inf),
        Scenario("clipped_counterexample", one_mu, one_pi, 1.0, 1.0, True),
        Scenario("extreme_lag_chain", extreme, target, 2.0, 1.0),
        Scenario("missing_support_one_shot", np.array([[0.0, 1.0], [0.5, 0.5]]), one_pi, 1.0, 1.0, True),
    ]


def visits(states, n_states):
    counts = np.zeros(n_states, dtype=np.int64)
    np.add.at(counts, states, 1)
    return counts


def rmse_at_visited(estimate, exact, states):
    visited = np.unique(states)
    return float(np.sqrt(np.mean((estimate[visited] - exact[visited]) ** 2)))


def average_state_targets(segments, results, n_states):
    sums = np.zeros(n_states)
    counts = np.zeros(n_states)
    for segment, result in zip(segments, results):
        np.add.at(sums, segment.states, result.targets)
        np.add.at(counts, segment.states, 1)
    mean = np.full(n_states, np.nan)
    visited = counts > 0
    mean[visited] = sums[visited] / counts[visited]
    return mean, counts


def run_scenario(scenario, seed, n_episodes=200, max_steps=8, bootstrap_mode="fixed_point"):
    mdp = make_one_shot_mdp() if scenario.one_shot else make_lag_mdp(horizon=6, slip=0.20)
    rng = np.random.default_rng(seed)
    actor = LaggedPolicyActor(
        mdp,
        np.stack([scenario.behavior, scenario.target]),
        lag=1,
        learner_version=1,
    )
    segments = collect_segments(mdp, actor, rng, n_episodes, max_steps)

    exact_target = policy_value(mdp, scenario.target, DISCOUNT)
    exact_behavior = policy_value(mdp, scenario.behavior, DISCOUNT)
    exact_clipped = None
    support = support_report(mdp, scenario.target, scenario.behavior)
    if not scenario.one_shot or support["supported_mass"][0] > 0:
        clipped = clipped_policy(scenario.target, scenario.behavior, scenario.rho_bar)
        exact_clipped = policy_value(mdp, clipped, DISCOUNT)

    if bootstrap_mode == "zero":
        state_values = np.zeros(mdp.n_states)
    elif bootstrap_mode == "fixed_point":
        if np.isinf(scenario.rho_bar):
            state_values = exact_target
        else:
            state_values = exact_clipped
    else:
        raise ValueError("bootstrap_mode must be 'zero' or 'fixed_point'")

    results = vtrace_for_segments(
        segments,
        scenario.target,
        state_values,
        discount=DISCOUNT,
        rho_bar=scenario.rho_bar,
        c_bar=scenario.c_bar,
    )
    estimates, counts = average_state_targets(segments, results, mdp.n_states)
    states = np.concatenate([segment.states for segment in segments])

    initial_targets = np.array(
        [result.targets[0] for segment, result in zip(segments, results) if segment.states[0] == 0]
    )
    return {
        "scenario": scenario.name,
        "seed": int(seed),
        "episodes": len(segments),
        "transitions": int(sum(segment.length for segment in segments)),
        "true_terminations": int(sum(np.any(segment.terminals) for segment in segments)),
        "truncations": int(sum(np.any(segment.truncated) for segment in segments)),
        "visits": counts.tolist(),
        "estimate": estimates.tolist(),
        "initial_mean": float(np.mean(initial_targets)),
        "initial_variance": float(np.var(initial_targets, ddof=1)),
        "rmse_to_target_fixed_point": None
        if exact_target is None
        else rmse_at_visited(estimates, exact_target, states),
        "rmse_to_clipped_fixed_point": None
        if exact_clipped is None
        else rmse_at_visited(estimates, exact_clipped, states),
        "exact_target": exact_target.tolist(),
        "exact_behavior": exact_behavior.tolist(),
        "exact_clipped": None if exact_clipped is None else exact_clipped.tolist(),
        "fixed_point_bias_at_initial": None
        if exact_clipped is None
        else float(exact_clipped[0] - exact_target[0]),
        "full_action_support": bool(support["full_action_support"]),
        "unsupported_mass": support["unsupported_mass"].tolist(),
        "state_values": state_values.tolist(),
    }


def batch_matches_trajectory(seed=7):
    mdp = make_lag_mdp(horizon=6)
    _, lagged, target, _, _ = chain_policies()
    actor = LaggedPolicyActor(mdp, np.stack([lagged, target]), lag=1, learner_version=1)
    segments = collect_segments(mdp, actor, np.random.default_rng(seed), 12, 8)
    values = policy_value(mdp, target, DISCOUNT)
    trajectory_results = vtrace_for_segments(
        segments, target, values, discount=DISCOUNT, rho_bar=2.0, c_bar=1.0
    )
    batch = pad_segments(segments, target, values)
    batch_result = vtrace_targets_batch(
        **batch, discount=DISCOUNT, rho_bar=2.0, c_bar=1.0
    )
    max_target_gap = 0.0
    max_advantage_gap = 0.0
    for index, segment in enumerate(segments):
        length = segment.length
        max_target_gap = max(
            max_target_gap,
            float(np.max(np.abs(batch_result.targets[:length, index] - trajectory_results[index].targets))),
        )
        max_advantage_gap = max(
            max_advantage_gap,
            float(np.max(np.abs(batch_result.advantages[:length, index] - trajectory_results[index].advantages))),
        )
    return {
        "max_target_gap": max_target_gap,
        "max_advantage_gap": max_advantage_gap,
    }


def c_tradeoff():
    mdp = make_lag_mdp(horizon=6, slip=0.20)
    _, lagged, target, _, _ = chain_policies()
    rows = []
    for c_bar in [0.01, 0.25, 0.5, 1.0]:
        value = np.zeros(mdp.n_states)
        for iteration in range(1, 100000):
            previous = value.copy()
            value = vtrace_expected_operator(
                mdp,
                target,
                lagged,
                value,
                discount=DISCOUNT,
                rho_bar=2.0,
                c_bar=c_bar,
            )
            delta = float(np.max(np.abs(value - previous)))
            if delta < 1e-10:
                break
        rows.append({
            "c_bar": c_bar,
            "iterations_to_1e-10": iteration,
            "initial_fixed_value": float(value[0]),
        })
    exact = policy_value(mdp, clipped_policy(target, lagged, 2.0), DISCOUNT)
    for row in rows:
        row["fixed_point_gap_at_initial"] = abs(row["initial_fixed_value"] - exact[0])
    return rows


def aggregate_seed_runs(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    summary = {}
    for name, group in grouped.items():
        initial = np.array([row["initial_mean"] for row in group])
        summary[name] = {
            "seeds": len(group),
            "episodes_per_seed": group[0]["episodes"],
            "total_transitions": int(sum(row["transitions"] for row in group)),
            "true_terminations": int(sum(row["true_terminations"] for row in group)),
            "truncations": int(sum(row["truncations"] for row in group)),
            "initial_estimate_mean": float(np.mean(initial)),
            "initial_estimate_variance_across_seeds": float(np.var(initial, ddof=1)),
            "mean_rmse_to_target_fixed_point": float(
                np.mean([row["rmse_to_target_fixed_point"] for row in group])
            ),
            "mean_rmse_to_clipped_fixed_point": (
                None
                if not any(row["rmse_to_clipped_fixed_point"] is not None for row in group)
                else float(np.mean([
                    row["rmse_to_clipped_fixed_point"]
                    for row in group
                    if row["rmse_to_clipped_fixed_point"] is not None
                ]))
            ),
            "target_initial_value": group[0]["exact_target"][0],
            "behavior_initial_value": group[0]["exact_behavior"][0],
            "clipped_initial_value": None
            if group[0]["exact_clipped"] is None
            else group[0]["exact_clipped"][0],
            "fixed_point_bias_at_initial": group[0]["fixed_point_bias_at_initial"],
            "full_action_support": group[0]["full_action_support"],
            "unsupported_mass": group[0]["unsupported_mass"],
    }
    return summary


def run_policy_lag_training(seed=3, iterations=12, episodes_per_iteration=24, lag=2):
    mdp = make_lag_mdp(horizon=6)
    rng = np.random.default_rng(seed)
    logits = np.zeros((mdp.n_states, mdp.n_actions))
    values = np.zeros(mdp.n_states)
    versions = []
    history = []

    for iteration in range(iterations):
        current = softmax_policy(logits)
        versions.append(current)
        actor = LaggedPolicyActor(
            mdp,
            np.stack(versions),
            lag=min(lag, len(versions) - 1),
            learner_version=len(versions) - 1,
        )
        segments = collect_segments(mdp, actor, rng, episodes_per_iteration, 8)
        results = vtrace_for_segments(
            segments,
            current,
            values,
            discount=DISCOUNT,
            rho_bar=2.0,
            c_bar=1.0,
        )
        exact_initial = policy_value(mdp, current, DISCOUNT)[0]
        history.append({
            "iteration": iteration,
            "learner_version": actor.learner_version,
            "behavior_version": actor.behavior_version,
            "exact_learner_initial_value": float(exact_initial),
        })
        logits, values = train_step(
            segments,
            results,
            logits,
            values,
            policy_learning_rate=0.05,
            value_learning_rate=0.25,
        )
    return history


def c_finite_sample_variance(seeds=10):
    mdp = make_lag_mdp(horizon=6, slip=0.20)
    _, lagged, target, _, _ = chain_policies()
    actor = LaggedPolicyActor(mdp, np.stack([lagged, target]), lag=1, learner_version=1)
    values = policy_value(mdp, clipped_policy(target, lagged, 2.0), DISCOUNT)

    segments_by_seed = []
    for seed in range(seeds):
        segments_by_seed.append(
            collect_segments(mdp, actor, np.random.default_rng(100 + seed), 200, 8)
        )

    rows = []
    for c_bar in [0.01, 0.25, 0.5, 1.0]:
        seed_means = []
        all_targets = []
        total_transitions = 0
        for segments in segments_by_seed:
            results = vtrace_for_segments(
                segments,
                target,
                values,
                discount=DISCOUNT,
                rho_bar=2.0,
                c_bar=c_bar,
            )
            targets = np.array([result.targets[0] for result in results])
            seed_means.append(float(targets.mean()))
            all_targets.extend(targets.tolist())
            total_transitions += sum(segment.length for segment in segments)
        rows.append({
            "c_bar": c_bar,
            "rho_bar": 2.0,
            "seeds": seeds,
            "episodes_per_seed": 200,
            "total_transitions": int(total_transitions),
            "initial_estimate_mean": float(np.mean(seed_means)),
            "initial_estimate_variance_across_seeds": float(np.var(seed_means, ddof=1)),
            "initial_estimate_variance_within_seeds": float(np.var(all_targets, ddof=1)),
            "clipped_initial_value": float(values[0]),
            "target_initial_value": float(policy_value(mdp, target, DISCOUNT)[0]),
        })
    return rows


def run_experiment(seeds=12, episodes_per_seed=200, output_dir="results"):
    seed_rows = []
    for scenario in make_scenarios():
        for seed in range(seeds):
            seed_rows.append(
                run_scenario(
                    scenario,
                    seed=1000 + seed,
                    n_episodes=episodes_per_seed,
                    max_steps=8,
                    bootstrap_mode="fixed_point",
                )
            )

    counterexample_seeds = []
    for seed in range(seeds):
        unclipped = Scenario(
            "counterexample_unclipped",
            np.array([[0.20, 0.80], [0.5, 0.5]]),
            np.array([[0.80, 0.20], [0.5, 0.5]]),
            np.inf,
            np.inf,
            True,
        )
        clipped = make_scenarios()[2]
        counterexample_seeds.append(run_scenario(unclipped, 2000 + seed, episodes_per_seed, 1, "fixed_point"))
        counterexample_seeds.append(run_scenario(clipped, 3000 + seed, episodes_per_seed, 1, "fixed_point"))

    def json_safe(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        return value

    result = {
        "discount": DISCOUNT,
        "scenario_summary": aggregate_seed_runs(seed_rows),
        "counterexample_summary": aggregate_seed_runs(counterexample_seeds),
        "c_fixed_point_and_convergence": c_tradeoff(),
        "c_finite_sample_variance": c_finite_sample_variance(seeds=min(seeds, 10)),
        "batch_reference_equivalence": batch_matches_trajectory(),
        "policy_lag_training": run_policy_lag_training(),
        "seed_rows": seed_rows + counterexample_seeds,
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "vtrace_experiment.json"
    json_path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return result, json_path
