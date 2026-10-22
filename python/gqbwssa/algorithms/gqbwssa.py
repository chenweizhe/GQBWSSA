from __future__ import annotations

import numpy as np

from ..utils import q_g_initialization


def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def gqbwssa(N, max_iter, lb, ub, dim, fobj, rng: np.random.Generator, log=None):
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    pos, fit = q_g_initialization(N, dim, ub, lb, fobj, rng)
    curve = np.zeros(max_iter)

    order = np.argsort(fit, kind="stable")
    xposbest = pos[order[0]].copy()
    fvalbest = fit[order[0]]

    mine, maxe = 0.4, 0.9
    beta1, beta2 = 0.4, 0.9

    for t in range(1, max_iter + 1):
        blocation = int(np.argmin(fit))
        best = fit[blocation]
        if t == 1 or best < fvalbest:
            fvalbest = best
            xposbest = pos[blocation].copy()

        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        w = maxe + t * (maxe - mine) / max_iter
        beta = beta2 - t * (beta2 - beta1) / max_iter

        for i in range(N):
            if i + 1 <= N / 2:
                
                c2 = rng.random(dim)
                c3 = rng.random(dim)
                step = c1 * ((ub - lb) * c2 + lb)
                pos[i] = np.where(c3 < 0.5, w * xposbest + step, w * xposbest - step)
                pos[i] = np.minimum(np.maximum(pos[i], lb), ub)
            else:
               
                for _ in range(dim):
                    point1 = pos[i - 1].copy()
                    point2 = pos[i].copy()
                    mmean = pos[i].mean()
                    if rng.random() > 0.1:
                        pos[i] = (point2 + point1) / 2
                    else:
                        pos[i] = xposbest - beta * (pos[i] - 2 * mmean)
                    pos[i] = np.minimum(np.maximum(pos[i], lb), ub)

        
        for i in range(N):
            pos[i] = np.minimum(np.maximum(pos[i], lb), ub)
            fit[i] = fobj(pos[i])
            blocation = int(np.argmin(fit))
            best = fit[blocation]
            if best < fvalbest:
                fvalbest = best
                xposbest = pos[blocation].copy()

        xposbest[1:] = _matlab_round(xposbest[1:])
        curve[t - 1] = fvalbest
        if log:
            log(t, fvalbest)

    best_pos = (xposbest > 0).astype(float)
    return best_pos, curve, fit.copy()
