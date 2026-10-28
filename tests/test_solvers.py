"""pytest -q   (runs in ~10 s)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gne_formation.game import make_scenario, ring_with_chords          # noqa: E402
from gne_formation.solvers import (centralized_vgne, distributed_pfb,   # noqa: E402
                                   init_state, kkt_residual)

STEPS = dict(alpha=0.05, tau=0.1, nu=0.1, c=3.0)


def _setup(N=6, seed=0):
    game, P0 = make_scenario(N, seed=seed)
    game.set_hyperplanes(P0, d_min=1.0)
    return game, P0


def test_game_is_strongly_monotone_and_not_potential():
    game, _ = _setup()
    assert game.strong_monotonicity() > 0
    J = game.jacobian()
    assert np.linalg.norm(J - J.T) > 1e-3        # non-symmetric => not a potential game


def test_reference_satisfies_kkt_and_has_active_constraints():
    game, P0 = _setup()
    X, lam, res = centralized_vgne(game, P0)
    assert res < 1e-9
    assert game.violation(X) < 1e-9
    assert np.sum(lam > 1e-6) >= 1               # coupling actually matters


def test_hyperplanes_imply_pairwise_separation():
    game, P0 = _setup()
    X, _, _ = centralized_vgne(game, P0)
    N = game.N
    d = min(np.linalg.norm(X[i] - X[j]) for i in range(N) for j in range(i + 1, N))
    assert d >= 1.0 - 1e-8


def test_distributed_matches_centralized():
    game, P0 = _setup()
    Xs, ls, _ = centralized_vgne(game, P0)
    S = distributed_pfb(game, ring_with_chords(6), init_state(game, P0), 4000, **STEPS)
    assert np.linalg.norm(S.X - Xs) / np.linalg.norm(Xs) < 1e-6
    assert np.abs(S.lam - ls).max() < 1e-4       # every local multiplier copy -> lambda*
    assert kkt_residual(game, S.X, S.lam.mean(0)) < 1e-5
