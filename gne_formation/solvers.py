"""Equilibrium-seeking algorithms.

* `centralized_vgne` - full-information reference solver for the variational GNE (v-GNE),
  via an extragradient method on the KKT operator, with an explicit KKT certificate.
* `distributed_pfb`  - fully distributed, partial-decision-information preconditioned
  forward-backward GNE seeking in the style of Pavel (IEEE TAC 2020) and Bianchi,
  Belgioioso & Grammatico (Automatica 2022). Each agent only talks to its neighbours on a
  communication graph, keeps an *estimate* of every other agent's decision, a local copy
  lambda_i of the shared multiplier and an auxiliary variable z_i that drives the
  multiplier copies to consensus.
* Setting `oracle="zeroth_order"` replaces the model-based partial gradient with a
  two-point random-direction estimate built only from (noisy) evaluations of the agent's
  own cost: a data-driven, model-free variant (cf. Krilasevic & Grammatico, Automatica 2021).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .game import FormationGame


# ----------------------------------------------------------------------------- reference
def kkt_residual(game: FormationGame, X: np.ndarray, lam: np.ndarray) -> float:
    """Natural-map residual of the v-GNE KKT system (0 iff (X, lam) is a v-GNE pair)."""
    F = game.pseudo_gradient(X)
    G = F + (game.A.T @ lam).reshape(game.N, 2) if game.m else F
    Xp = np.stack([game.proj_box(i, X[i] - G[i]) for i in range(game.N)])
    r = np.linalg.norm(X - Xp)
    if game.m:
        lp = np.maximum(lam + game.A @ X.reshape(-1) - game.b, 0.0)
        r = max(r, np.linalg.norm(lam - lp))
    return float(r)


def centralized_vgne(game: FormationGame, X0: np.ndarray, iters: int = 200000,
                     tol: float = 1e-11) -> tuple[np.ndarray, np.ndarray, float]:
    """Korpelevich extragradient on T(x, lam) = (F(x) + A^T lam, b - A x) + normal cones."""
    N, m = game.N, game.m
    L = game.lipschitz() + (np.linalg.norm(game.A, 2) if m else 0.0)
    gam = 0.9 / L
    X, lam = X0.copy(), np.zeros(m)

    def op(X, lam):
        gx = game.pseudo_gradient(X)
        if m:
            gx = gx + (game.A.T @ lam).reshape(N, 2)
            gl = game.b - game.A @ X.reshape(-1)
        else:
            gl = np.zeros(0)
        return gx, gl

    def proj(X, lam):
        return np.stack([game.proj_box(i, X[i]) for i in range(N)]), np.maximum(lam, 0.0)

    res = np.inf
    for k in range(iters):
        gx, gl = op(X, lam)
        Xh, lh = proj(X - gam * gx, lam - gam * gl)
        gx, gl = op(Xh, lh)
        X, lam = proj(X - gam * gx, lam - gam * gl)
        if k % 200 == 0:
            res = kkt_residual(game, X, lam)
            if res < tol:
                break
    return X, lam, kkt_residual(game, X, lam)


# ----------------------------------------------------------------------------- distributed
@dataclass
class PFBState:
    Xest: np.ndarray   # (N, N, 2): Xest[i, j] = agent i's estimate of x_j (Xest[i, i] = x_i)
    lam: np.ndarray    # (N, m): local multiplier copies
    z: np.ndarray      # (N, m): auxiliary consensus variables

    @property
    def X(self) -> np.ndarray:
        return np.stack([self.Xest[i, i] for i in range(self.Xest.shape[0])])


def init_state(game: FormationGame, X0: np.ndarray) -> PFBState:
    N, m = game.N, game.m
    return PFBState(Xest=np.tile(X0[None], (N, 1, 1)).copy(),
                    lam=np.zeros((N, m)), z=np.zeros((N, m)))


def _zeroth_order_grad(game, i, Xi_view, delta, noise, n_dirs, rng):
    """Two-point random-direction estimate of nabla_{x_i} J_i using only cost values."""
    g = np.zeros(2)
    for _ in range(n_dirs):
        th = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        Xp, Xm = Xi_view.copy(), Xi_view.copy()
        Xp[i] += delta * u
        Xm[i] -= delta * u
        Jp = game.cost(i, Xp) + noise * rng.standard_normal()
        Jm = game.cost(i, Xm) + noise * rng.standard_normal()
        g += 2.0 * (Jp - Jm) / (2.0 * delta) * u      # dimension n_i = 2
    return g / n_dirs


def distributed_pfb(game: FormationGame, W: np.ndarray, state: PFBState, iters: int,
                    alpha: float, tau: float, nu: float, c: float,
                    oracle: str = "model", delta: float = 0.05, noise: float = 0.0,
                    n_dirs: int = 1, seed: int = 0, record=None):
    """Run `iters` synchronous rounds; each round = one exchange with graph neighbours.

    Agent i update (only neighbour information in the Laplacian terms):
      x_i      <- P_Omega_i[ x_i - alpha (grad_i J_i(Xest_i) + A_i^T lam_i + c sum_j w_ij (x_i - Xest_j[i])) ]
      Xest_i[-i] <- Xest_i[-i] - alpha c sum_j w_ij (Xest_i[-i] - Xest_j[-i])
      z_i      <- z_i + nu sum_j w_ij (lam_i - lam_j)
      lam_i    <- P_{>=0}[ lam_i + tau ( A_i (2 x_i^+ - x_i) - b_i
                           - sum_j w_ij ((2 z_i^+ - z_i) - (2 z_j^+ - z_j))
                           - sum_j w_ij (lam_i - lam_j) ) ]
    """
    N, m = game.N, game.m
    Lap = np.diag(W.sum(1)) - W
    bl = game.b_local() if m else np.zeros((N, 0))
    Ab = [game.A_block(i) for i in range(N)] if m else None
    rng = np.random.default_rng(seed)
    S = state
    for k in range(iters):
        Xflat = S.Xest.reshape(N, -1)
        LX = (Lap @ Xflat).reshape(N, N, 2)             # consensus disagreement on estimates
        Xold = S.X
        Xnew_est = S.Xest - alpha * c * LX              # estimate update (i's own row fixed below)
        for i in range(N):
            if oracle == "model":
                gi = game.grad_i(i, S.Xest[i])
            else:
                gi = _zeroth_order_grad(game, i, S.Xest[i], delta, noise, n_dirs, rng)
            if m:
                gi = gi + Ab[i].T @ S.lam[i]
            xi = S.Xest[i, i] - alpha * (gi + c * LX[i, i])
            Xnew_est[i, i] = game.proj_box(i, xi)
        Xnew = np.stack([Xnew_est[i, i] for i in range(N)])
        if m:
            z_new = S.z + nu * (Lap @ S.lam)
            zz = 2 * z_new - S.z
            Ax = np.stack([Ab[i] @ (2 * Xnew[i] - Xold[i]) for i in range(N)])
            lam_new = np.maximum(S.lam + tau * (Ax - bl - Lap @ zz - Lap @ S.lam), 0.0)
            S.z, S.lam = z_new, lam_new
        S.Xest = Xnew_est
        if record is not None:
            record(k, S)
    return S
