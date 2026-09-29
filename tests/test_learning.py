import numpy as np

from vtrace_lab.learning import corrected_policy_gradient, value_gradient


def test_corrected_policy_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(3, 2))
    states = np.array([0, 0, 2])
    actions = np.array([1, 0, 1])
    advantages = np.array([0.7, -0.4, 1.2])
    gradient = corrected_policy_gradient(states, actions, advantages, logits, 3, 2)

    epsilon = 1e-6
    numerical = np.zeros_like(logits)

    def objective(candidate):
        shifted = candidate - candidate.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        return sum(
            advantage * np.log(probs[state, action])
            for state, action, advantage in zip(states, actions, advantages)
        )

    for state in range(3):
        for action in range(2):
            plus = logits.copy()
            minus = logits.copy()
            plus[state, action] += epsilon
            minus[state, action] -= epsilon
            numerical[state, action] = (objective(plus) - objective(minus)) / (2 * epsilon)

    assert np.allclose(gradient, numerical, atol=1e-6)


def test_value_gradient_points_away_from_detached_target():
    values = np.array([1.0, 3.0])
    targets = np.array([1.5, 2.0])
    states = np.array([0, 1])
    gradient = value_gradient(states, values, targets, 2)
    assert np.allclose(gradient, [-0.5, 1.0])
