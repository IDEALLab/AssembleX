"""Unit tests for the decoupled WeightOptimizer (online preference learning).

Runs standalone (`python test_weight_optimizer.py`) since this environment has
no pytest; the test_* functions are also pytest-compatible. Imports only the
pure-numpy optimizer — no ASAPx / simulation dependencies.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ASAPx"))
from plan_sequence.planner.weight_optimizer import WeightOptimizer


def _make(epsilon=0.1):
    return WeightOptimizer(
        n_features=3,
        feature_names=["a", "b", "c"],
        epsilon=epsilon,
        lower_is_better=True,
    )


def test_update_makes_star_strictly_cheaper():
    """After an update, the oracle's pick n_star must have strictly lower cost
    than the model's pick n_hat (it is preferred under the new weights)."""
    opt = _make()
    phi_hat = np.array([0.0, 0.0, 1.0])  # model's (wrong) pick
    phi_star = np.array([1.0, 0.0, 0.0])  # human's pick
    opt.update(phi_hat, phi_star)
    cost_hat = float(opt.weights @ phi_hat)
    cost_star = float(opt.weights @ phi_star)
    assert cost_star < cost_hat, (cost_star, cost_hat)


def test_no_update_when_already_correct():
    """Identical features (model already picked n_star) → no weight change,
    eta=0, and no division-by-zero."""
    opt = _make()
    before = opt.weights.copy()
    phi = np.array([0.3, 0.6, 0.1])
    opt.update(phi, phi)
    assert np.allclose(opt.weights, before)
    assert opt.n_updates == 0


def test_weights_stay_normalized():
    """Weight vector remains L2-normalized after a sequence of updates."""
    opt = _make()
    rng = np.random.default_rng(0)
    for _ in range(50):
        phi_hat = rng.random(3)
        phi_star = rng.random(3)
        opt.update(phi_hat, phi_star)
        assert abs(np.linalg.norm(opt.weights) - 1.0) < 1e-9


def test_should_query_margin_gate():
    """should_query is True only when the top-2 cost gap is below epsilon, and
    False when there are fewer than 2 candidates."""
    opt = _make(epsilon=0.1)
    # Force known weights so costs are predictable: cost = w·phi.
    opt.weights = opt._normalize(np.array([1.0, 0.0, 0.0]))
    w0 = float(opt.weights[0])

    # Two candidates with a LARGE cost gap (>> epsilon) → no query.
    big_gap = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert opt.should_query(big_gap) is False

    # Two candidates with a TINY cost gap (< epsilon) → query.
    tiny = (opt.epsilon * 0.5) / w0
    small_gap = np.array([[0.0, 0.0, 0.0], [tiny, 0.0, 0.0]])
    assert opt.should_query(small_gap) is True

    # Single candidate → never query.
    assert opt.should_query(np.array([[1.0, 2.0, 3.0]])) is False


def test_save_and_load_roundtrip():
    """Weights + metadata persist across save/load (new optimizer picks them up)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.json")
        opt = WeightOptimizer(
            n_features=3, feature_names=["a", "b", "c"], epsilon=0.2, save_path=path
        )
        opt.update(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        saved = opt.weights.copy()
        n_up = opt.n_updates

        opt2 = WeightOptimizer(
            n_features=3, feature_names=["a", "b", "c"], epsilon=0.2, save_path=path
        )
        assert np.allclose(opt2.weights, saved)
        assert opt2.n_updates == n_up


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
