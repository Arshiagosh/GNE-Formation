"""Reproduce every figure and number in the README.

    python experiments/run_experiments.py         # ~2-3 minutes on a laptop

E1  Static GNE (N=6): distributed partial-information pFB vs. the centralized v-GNE,
    model-based and zeroth-order (cost-measurement-only) oracles.
E2  Scaling: rounds needed to reach 1e-6 relative accuracy for N = 4 ... 12 robots.
E3  Receding-horizon game-theoretic planner: robots re-solve the GNE every step with
    hyperplanes rebuilt around their current positions; recursive feasibility keeps
    every pair >= d_min apart.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gne_formation.game import make_scenario, ring_with_chords          # noqa: E402
from gne_formation.solvers import (centralized_vgne, init_state,        # noqa: E402
                                   distributed_pfb, kkt_residual)

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# categorical palette (fixed order) + recessive ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
STEPS = dict(alpha=0.05, tau=0.1, nu=0.1, c=3.0)
D_MIN = 1.0
results: dict = {}


# ------------------------------------------------------------------------------- E1
def e1():
    game, P0 = make_scenario(6, seed=0)
    game.set_hyperplanes(P0, d_min=D_MIN)
    Xs, ls, res = centralized_vgne(game, P0)
    W = ring_with_chords(6)
    scale = np.linalg.norm(Xs)
    runs = {
        "model-based gradient": dict(oracle="model"),
        "zeroth-order, 4 probes/round, σ=0.01": dict(oracle="zeroth_order", n_dirs=4, noise=0.01),
        "zeroth-order, 32 probes/round, σ=0.01": dict(oracle="zeroth_order", n_dirs=32, noise=0.01),
    }
    iters = 4000
    curves = {}
    for name, kw in runs.items():
        err, vio = [], []
        def rec(k, S):
            err.append(np.linalg.norm(S.X - Xs) / scale)
            vio.append(game.violation(S.X))
        S = distributed_pfb(game, W, init_state(game, P0), iters, **STEPS, record=rec, **kw)
        curves[name] = (np.array(err), np.array(vio))
        results.setdefault("E1", {})[name] = {
            "final_rel_error": float(err[-1]),
            "final_max_violation": float(vio[-1]),
            "final_lambda_gap": float(np.abs(S.lam - ls).max()),
        }
    results["E1"]["reference"] = {
        "kkt_residual": res, "active_constraints": int(np.sum(ls > 1e-6)),
        "num_constraints": game.m, "strong_monotonicity": game.strong_monotonicity(),
        "lipschitz": game.lipschitz(),
        "jacobian_asymmetry": float(np.linalg.norm(game.jacobian() - game.jacobian().T)),
    }

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for (name, (err, vio)), col in zip(curves.items(), SERIES):
        axs[0].semilogy(err, color=col, label=name)
        # running max over 50 rounds: shows the envelope instead of noise spikes to zero
        env = np.array([vio[max(0, k - 49):k + 1].max() for k in range(len(vio))])
        axs[1].semilogy(np.maximum(env, 1e-12), color=col, label=name)
    axs[0].set(xlabel="communication round k", ylabel="‖x_k − x*‖ / ‖x*‖",
               title="Distance to the v-GNE")
    axs[1].set(xlabel="communication round k", ylabel="max (Ax − b)₊  (50-round envelope)",
               title="Shared-constraint violation")
    axs[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("E1 · 6 robots, 15 coupling constraints, ring+chord graph", x=0.01, y=1.04,
                 ha="left", fontsize=11, color=INK)
    fig.savefig(os.path.join(FIG, "e1_convergence.png"))
    plt.close(fig)

    # equilibrium picture
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for i, col in enumerate(SERIES[:6]):
        ax.plot(*P0[i], "o", color=col, mfc="white", ms=8, mew=2)
        ax.plot(*Xs[i], "o", color=col, ms=9, mec="white", mew=2, label=f"robot {i+1}")
        ax.add_patch(plt.Circle(Xs[i], D_MIN / 2, fill=False, ec=col, lw=1, ls="--"))
    ax.plot(*game.goals[0], "x", color=INK, ms=10, mew=2.5)
    ax.annotate("shared goal", game.goals[0], xytext=(-62, 14), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.set_aspect("equal")
    ax.set(xlim=(0, 10), ylim=(0, 10), title="Start (hollow) → v-GNE (filled)")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.savefig(os.path.join(FIG, "e1_equilibrium.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------- E2
def e2():
    Ns, rounds = [4, 6, 8, 10, 12], []
    for N in Ns:
        game, P0 = make_scenario(N, seed=1)
        game.set_hyperplanes(P0, d_min=D_MIN)
        Xs, _, _ = centralized_vgne(game, P0)
        W = ring_with_chords(N)
        scale = np.linalg.norm(Xs)
        hit = {"k": None}
        def rec(k, S):
            if hit["k"] is None and np.linalg.norm(S.X - Xs) / scale < 1e-6:
                hit["k"] = k
        distributed_pfb(game, W, init_state(game, P0), 20000, **STEPS, record=rec)
        rounds.append(hit["k"])
        results.setdefault("E2", {})[N] = {"rounds_to_1e-6": hit["k"], "constraints": game.m}
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(Ns, rounds, "-o", color=SERIES[0], ms=8, mec="white", mew=2)
    for n, r in zip(Ns, rounds):
        ax.annotate(f"{r}", (n, r), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.set(xlabel="number of robots N", ylabel="rounds to 10⁻⁶ rel. error",
           title="E2 · Scaling (ring+chord graph)")
    ax.set_ylim(0, max(rounds) * 1.2)
    fig.savefig(os.path.join(FIG, "e2_scaling.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------- E3
def e3(steps: int = 40, vmax_step: float = 0.35, inner: int = 2500, margin: float = 0.1):
    N = 6
    game, P = make_scenario(N, seed=0)
    W = ring_with_chords(N)
    traj, dmins, kkts, viols = [P.copy()], [], [], []
    for t in range(steps):
        game.lo = np.maximum(P - vmax_step, 0.0)
        game.hi = np.minimum(P + vmax_step, 10.0)
        game.set_hyperplanes(P, d_min=D_MIN + margin, radius=4.0)
        S = distributed_pfb(game, W, init_state(game, P), inner, **STEPS)
        Xn = S.X
        kkts.append(kkt_residual(game, Xn, S.lam.mean(0)) if game.m else 0.0)
        viols.append(game.violation(Xn))
        P = Xn
        traj.append(P.copy())
        dmins.append(min(np.linalg.norm(P[i] - P[j]) for i in range(N) for j in range(i + 1, N)))
    traj = np.array(traj)
    results["E3"] = {"steps": steps, "min_pairwise_distance": float(min(dmins)),
                     "d_min": D_MIN, "planning_margin": margin, "inner_rounds": inner,
                     "max_kkt_residual": float(max(kkts)), "max_planned_violation": float(max(viols)),
                     "final_dist_to_goal_mean": float(np.linalg.norm(P - game.goals, axis=1).mean())}

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    for i, col in enumerate(SERIES[:N]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=col, lw=2, label=f"robot {i+1}")
        ax.plot(*traj[0, i], "o", color=col, mfc="white", ms=7, mew=2)
        ax.plot(*traj[-1, i], "o", color=col, ms=8, mec="white", mew=2)
    ax.plot(*game.goals[0], "x", color=INK, ms=10, mew=2.5)
    ax.set_aspect("equal")
    ax.set(xlim=(0, 10), ylim=(0, 10), title="Receding-horizon GNE planner")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax = axs[1]
    ax.plot(np.arange(1, steps + 1), dmins, color=SERIES[0], label="closest pair")
    ax.axhline(D_MIN, color=INK2, lw=1, ls="--")
    ax.annotate("d_min", (steps, D_MIN), xytext=(-4, -12), textcoords="offset points",
                ha="right", fontsize=8, color=INK2)
    ax.set(xlabel="planning step", ylabel="min pairwise distance", title="Safety margin",
           ylim=(0, max(dmins) * 1.1))
    fig.suptitle(f"E3 · each step = {inner} neighbour-only rounds, hyperplanes rebuilt at current positions",
                 x=0.01, y=1.04, ha="left", fontsize=10, color=INK)
    fig.savefig(os.path.join(FIG, "e3_receding_horizon.png"))
    plt.close(fig)


if __name__ == "__main__":
    e1(); print("E1 done")
    e2(); print("E2 done")
    e3(); print("E3 done")
    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))
