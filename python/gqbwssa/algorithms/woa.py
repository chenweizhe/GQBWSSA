
from __future__ import annotations

import numpy as np

from ..utils import bounds, sigmoid_binarize


def woa(N, max_iter, lb, ub, dim, fobj, rng: np.random.Generator, log=None):
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    positions = (rng.random((N, dim)) > 0.5).astype(float)
    fitness = fobj.batch(positions)
    idx = np.argsort(fitness, kind="stable")
    leader_pos, leader_score = positions[idx[0]].copy(), fitness[idx[0]]

    curve = np.zeros(max_iter)
    epositions = np.zeros((N, dim))
    for t in range(1, max_iter + 1):
        a = 2 - t * (2 / max_iter)
        a2 = -1 + t * (-1 / max_iter)

        for i in range(N):
            r1, r2 = rng.random(), rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = (a2 - 1) * rng.random() + 1

            p = rng.random(dim)
            if abs(A) >= 1:
                
                rand_idx = rng.integers(0, N, size=dim)
                x_rand = positions[rand_idx, np.arange(dim)]
                shrink = x_rand - A * np.abs(C * x_rand - positions[i])
            else:
               
                shrink = leader_pos - A * np.abs(C * leader_pos - positions[i])
           
            spiral = np.abs(leader_pos - positions[i]) * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos
            epositions[i] = np.where(p < 0.5, shrink, spiral)

        epositions = bounds(epositions, lb, ub)
        positions = np.vstack([sigmoid_binarize(e, rng) for e in epositions])
        fitness = fobj.batch(positions)
        for i in range(N):
            if fitness[i] < leader_score:
                leader_score, leader_pos = fitness[i], positions[i].copy()

        curve[t - 1] = leader_score
        if log:
            log(t, leader_score)

    return leader_pos, curve, fitness.copy()
