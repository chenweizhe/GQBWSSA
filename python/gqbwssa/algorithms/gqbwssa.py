"""GQBWSSA"""

import time

import numpy as np

from ..utils import bounds, matlab_round, q_g_initialization
from .common import MultiRunRecorder


def gqbwssa(N, iters, lb, ub, dim, fobj, multi_num, fitness_obj, rng):
    rec = MultiRunRecorder()

    for ii in range(multi_num):
        fitness_obj.clear_cache()
        t0 = time.perf_counter()
        pos, fit = q_g_initialization(N, dim, ub, lb, fobj, rng)
        curve = np.zeros(iters)

        order = np.argsort(fit, kind="stable")
        xposbest = pos[order[0]].copy()
        fvalbest = fit[order[0]]

        for t in range(1, iters + 1):
            blocation = int(np.argmin(fit))
            best = fit[blocation]
            if t == 1 or best < fvalbest:
                fvalbest = best
                xposbest = pos[blocation].copy()

            print(f"  [GQBWSSA] run {ii + 1}/{multi_num}  iter {t}/{iters}  best={fvalbest:.6f}", flush=True)

            c1 = 2 * np.exp(-(4 * t / iters) ** 2)
            mine, maxe = 0.4, 0.9
            w = maxe + t * (maxe - mine) / iters
            beta1, beta2 = 0.4, 0.9
            beta = beta2 - t * (beta2 - beta1) / iters

            for i in range(N):
                i1 = i + 1
                if i1 <= N / 2:
                    c2 = rng.random(dim)
                    c3 = rng.random(dim)
                    step = c1 * ((ub - lb) * c2 + lb)
                    pos[i] = np.where(c3 < 0.5, w * xposbest + step, w * xposbest - step)
                    pos[i] = bounds(pos[i], lb, ub)
                elif N / 2 < i1 < N + 1:
                    for _ in range(dim):
                        point1 = pos[i - 1].copy()
                        point2 = pos[i].copy()
                        mmean = pos[i].mean()
                        if rng.random() > 0.1:
                            pos[i] = (point2 + point1) / 2
                        else:
                            pos[i] = xposbest - beta * (pos[i] - 2 * mmean)
                        pos[i] = bounds(pos[i], lb, ub)

            for i in range(N):
                pos[i] = bounds(pos[i], lb, ub)
                fit[i] = fobj(pos[i])
                blocation = int(np.argmin(fit))
                if fit[blocation] < fvalbest:
                    fvalbest = fit[blocation]
                    xposbest = pos[blocation].copy()

            xposbest[1:] = matlab_round(xposbest[1:])
            curve[t - 1] = fvalbest

        xposbest = (xposbest > 0).astype(float)
        rec.add(fitness_obj, xposbest, curve, t0)

    print("GQBWSSA finished", flush=True)
    return rec.result()
