from __future__ import annotations

import numpy as np


def softmax_policy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True)


def deterministic_logits(policy: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(policy, dtype=np.float64) + 1e-300) * 100.0
