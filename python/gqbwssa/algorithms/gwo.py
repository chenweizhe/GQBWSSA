
from __future__ import annotations

import numpy as np

from ..utils import bounds, sigmoid_binarize


def gwo(N, max_iter, lb, ub, dim, fobj, rng: np.random.Generator, log=None):
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    positions = (rng.random((N, dim)) > 0.5).astype(float)
    fitness = fobj.batch(positions)

    idx = np.argsort(fitness, kind="stable")
    alpha_pos, alpha_score = positions[idx[0]].copy(), fitness[idx[0]]
    beta_pos, beta_score = positions[idx[1]].copy(), fitness[idx[1]]
    delta_pos, delta_score = positions[idx[2]].copy(), fitness[idx[2]]

    curve = np.zeros(max_iter)
    for t in range(1, max_iter + 1):
        a = 2 - t * (2 / max_iter)

        r1, r2 = rng.random((N, dim)), rng.random((N, dim))
        A1, C1 = 2 * a * r1 - a, 2 * r2
        X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions)

        r1, r2 = rng.random((N, dim)), rng.random((N, dim))
        A2, C2 = 5 * a * r1 - a, 5 * r2  
        X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions)

        r1, r2 = rng.random((N, dim)), rng.random((N, dim))
        A3, C3 = 2 * a * r1 - a, 2 * r2
        X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions)

        epositions = bounds((X1 + X2 + X3) / 3, lb, ub)

        positions = np.vstack([sigmoid_binarize(e, rng) for e in epositions])
        fitness = fobj.batch(positions)
        for i in range(N):
            f = fitness[i]
            if f < alpha_score:
                alpha_score, alpha_pos = f, positions[i].copy()
            if alpha_score < f < beta_score:
                beta_score, beta_pos = f, positions[i].copy()
            if f > alpha_score and f > beta_score and f < delta_score:
                delta_score, delta_pos = f, positions[i].copy()

        curve[t - 1] = alpha_score
        if log:
            log(t, alpha_score)

    return alpha_pos, curve, fitness.copy()
