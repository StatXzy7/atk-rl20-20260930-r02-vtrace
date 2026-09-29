import numpy as np

from vtrace_lab.vtrace import vtrace_targets_batch


def test_batch_distinguishes_true_termination_from_truncation_and_padding():
    rewards = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 0.0],
    ])
    actions = np.zeros((2, 3), dtype=int)
    target = np.tile(np.ones((2, 3, 2)) / 2.0, (1, 1, 1))
    behavior = target.copy()
    behavior_probs = np.full((2, 3), 0.5)
    values = np.zeros((2, 3))
    bootstrap = np.zeros((2, 3))
    bootstrap[1, 1] = 5.0
    terminals = np.array([
        [False, False, False],
        [True, False, False],
    ])
    truncated = np.array([
        [False, False, False],
        [False, True, False],
    ])
    mask = np.array([
        [True, True, False],
        [True, True, False],
    ])
    result = vtrace_targets_batch(
        rewards,
        actions,
        target,
        behavior,
        behavior_probs,
        values,
        bootstrap,
        terminals,
        truncated,
        mask,
        discount=0.9,
        rho_bar=np.inf,
        c_bar=np.inf,
    )
    assert np.isclose(result.targets[0, 0], 1.0 + 0.9 * 2.0)
    assert np.isclose(result.targets[1, 0], 2.0)
    assert np.isclose(
        result.targets[0, 1],
        1.0 + 0.9 * 2.0 + 0.9 ** 2 * 5.0,
    )
    assert np.isclose(result.targets[1, 1], 2.0 + 0.9 * 5.0)
    assert np.isnan(result.targets[0, 2])
    assert np.isnan(result.targets[1, 2])
