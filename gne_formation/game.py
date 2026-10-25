"""Multi-robot formation game with shared (convexified) collision-avoidance constraints.

Each robot i chooses a planar position x_i in R^2 inside a local box Omega_i and minimises

    J_i(x_i, x_-i) = a_i/2 ||x_i - g_i||^2 + sum_{j in F_i} b_ij/2 ||x_i - x_j - d_ij||^2

i.e. "get to my own goal" plus "keep my formation offsets to my formation neighbours".
The weights b_ij are NOT symmetric (b_ij != b_ji), so the pseudo-gradient has a
non-symmetric Jacobian: this is a genuine (non-potential) game, not an optimisation
problem in disguise.

Shared coupling constraints are separating hyperplanes, one per robot pair (i, j):

    n_ij^T (x_j - x_i) >= d_min,      n_ij = (p_j - p_i) / ||p_j - p_i||,

built from reference positions p. Since n_ij is a unit vector, any x satisfying the
hyperplane also satisfies ||x_j - x_i|| >= d_min, so this is a conservative *convex
inner approximation* of the (non-convex) collision-avoidance constraint. Stacked, the
coupling is A x <= b with A = [A_1, ..., A_N] column-separable, which is exactly the
generalized Nash equilibrium problem (GNEP) class treated by operator-splitting methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import numpy as np


@dataclass
class FormationGame:
    goals: np.ndarray              # (N, 2) individual goals g_i
    a: np.ndarray                  # (N,) goal weights a_i > 0
    B: np.ndarray                  # (N, N) formation weights b_ij >= 0 (non-symmetric)
    D: np.ndarray                  # (N, N, 2) desired offsets d_ij = desired x_i - x_j
    lo: np.ndarray                 # (N, 2) lower corner of Omega_i
    hi: np.ndarray                 # (N, 2) upper corner of Omega_i
    A: np.ndarray = field(default=None)   # (m, 2N) coupling matrix
    b: np.ndarray = field(default=None)   # (m,)
    pairs: list = field(default_factory=list)

    @property
    def N(self) -> int:
        return self.goals.shape[0]

    @property
    def m(self) -> int:
        return 0 if self.A is None else self.A.shape[0]

    # ---------- costs and pseudo-gradient ----------
    def cost(self, i: int, X: np.ndarray) -> float:
        """J_i evaluated at the full profile X (N, 2) -- used as a black-box oracle."""
        c = 0.5 * self.a[i] * np.sum((X[i] - self.goals[i]) ** 2)
        for j in range(self.N):
            if self.B[i, j] > 0:
                c += 0.5 * self.B[i, j] * np.sum((X[i] - X[j] - self.D[i, j]) ** 2)
        return float(c)

    def grad_i(self, i: int, X: np.ndarray) -> np.ndarray:
        """nabla_{x_i} J_i(x_i, x_-i) evaluated at the profile X (N, 2)."""
        g = self.a[i] * (X[i] - self.goals[i])
        for j in range(self.N):
            if self.B[i, j] > 0:
                g = g + self.B[i, j] * (X[i] - X[j] - self.D[i, j])
        return g

    def pseudo_gradient(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.grad_i(i, X) for i in range(self.N)])

    def jacobian(self) -> np.ndarray:
        """Constant Jacobian of the (affine) pseudo-gradient, shape (2N, 2N)."""
        N = self.N
        Q = np.zeros((N, N))
        for i in range(N):
            Q[i, i] = self.a[i] + self.B[i].sum()
            for j in range(N):
                if j != i:
                    Q[i, j] = -self.B[i, j]
        return np.kron(Q, np.eye(2))

    def strong_monotonicity(self) -> float:
        J = self.jacobian()
        return float(np.linalg.eigvalsh(0.5 * (J + J.T)).min())

    def lipschitz(self) -> float:
        return float(np.linalg.norm(self.jacobian(), 2))

    # ---------- sets and constraints ----------
    def proj_box(self, i: int, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lo[i], self.hi[i])

    def A_block(self, i: int) -> np.ndarray:
        return self.A[:, 2 * i: 2 * i + 2]

    def b_local(self) -> np.ndarray:
        """Split b among agents: b = sum_i b_i (each row shared by its two robots)."""
        bl = np.zeros((self.N, self.m))
        for r, (i, j) in enumerate(self.pairs):
            bl[i, r] = 0.5 * self.b[r]
            bl[j, r] = 0.5 * self.b[r]
        return bl

    def violation(self, X: np.ndarray) -> float:
        if self.m == 0:
            return 0.0
        return float(np.maximum(self.A @ X.reshape(-1) - self.b, 0.0).max())

    def set_hyperplanes(self, P: np.ndarray, d_min: float, radius: float = np.inf) -> None:
        """Build pairwise separating hyperplanes from reference positions P (N, 2).

        Only pairs closer than `radius` get a constraint (others cannot collide within
        one planning step). If P itself is d_min-separated, then P is feasible.
        """
        rows, rhs, pairs = [], [], []
        for i, j in itertools.combinations(range(self.N), 2):
            dist = np.linalg.norm(P[j] - P[i])
            if dist > radius:
                continue
            n = (P[j] - P[i]) / dist
            row = np.zeros(2 * self.N)
            # n^T (x_j - x_i) >= d_min   <=>   n^T x_i - n^T x_j <= -d_min
            row[2 * i: 2 * i + 2] = n
            row[2 * j: 2 * j + 2] = -n
            rows.append(row)
            rhs.append(-d_min)
            pairs.append((i, j))
        self.A = np.array(rows) if rows else np.zeros((0, 2 * self.N))
        self.b = np.array(rhs)
        self.pairs = pairs


def ring_with_chords(N: int, chord: int = 2) -> np.ndarray:
    """Undirected communication graph: ring plus chords i -- i+chord. Returns adjacency."""
    W = np.zeros((N, N))
    for i in range(N):
        for s in (1, chord):
            j = (i + s) % N
            if j != i:
                W[i, j] = W[j, i] = 1.0
    return W


def make_scenario(N: int = 6, seed: int = 0, workspace: float = 10.0) -> tuple[FormationGame, np.ndarray]:
    """Selfish robots contending for the same target, with a desired ring formation.

    Every robot's individual goal is the SAME target point (so they compete for it);
    formation terms ask for a regular polygon around it, each robot coupled to its two
    polygon neighbours with asymmetric weights. Returns (game, start positions).
    """
    rng = np.random.default_rng(seed)
    target = np.array([7.5, 5.0])
    goals = np.tile(target, (N, 1))
    a = rng.uniform(0.8, 1.2, N)
    r_form = 1.2 + 0.15 * N
    ang = 2 * np.pi * np.arange(N) / N
    slots = target + r_form * np.c_[np.cos(ang), np.sin(ang)]
    B = np.zeros((N, N))
    D = np.zeros((N, N, 2))
    for i in range(N):
        for j in ((i - 1) % N, (i + 1) % N):
            B[i, j] = rng.uniform(0.3, 1.0)         # asymmetric: b_ij != b_ji
            D[i, j] = slots[i] - slots[j]
    # start: scattered on the left of the workspace, pairwise >= 1.5 apart
    starts = []
    while len(starts) < N:
        p = np.array([rng.uniform(0.5, 3.0), rng.uniform(0.5, workspace - 0.5)])
        if all(np.linalg.norm(p - q) >= 1.5 for q in starts):
            starts.append(p)
    starts = np.array(starts)
    lo = np.zeros((N, 2))
    hi = np.full((N, 2), workspace)
    game = FormationGame(goals=goals, a=a, B=B, D=D, lo=lo, hi=hi)
    return game, starts
