
from __future__ import annotations

import numpy as np

from ..utils import bounds, initialization, levy, sigmoid_binarize


def hho(N, max_iter, lb, ub, dim, fobj, rng: np.random.Generator, log=None):
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    rabbit_location = np.zeros(dim)
    rabbit_energy = np.inf

    X = initialization(N, ub, lb, dim, rng)
    curve = np.zeros(max_iter)
    fitness00 = np.zeros(N)

    t = 0
    while t < max_iter:
        X = bounds(X, lb, ub)
        X = np.vstack([sigmoid_binarize(row, rng) for row in X])
        fitness00 = fobj.batch(X)
        for i in range(N):
            if fitness00[i] < rabbit_energy:
                rabbit_energy = fitness00[i]
                rabbit_location = X[i].copy()

        E1 = 2 * (1 - t / max_iter)
        for i in range(N):
            escaping_energy = E1 * (2 * rng.random() - 1)

            if abs(escaping_energy) >= 1:
                
                q = rng.random()
                x_rand = X[int(np.floor(N * rng.random()))].copy()
                if q < 0.5:
                    X[i] = x_rand - rng.random() * np.abs(x_rand - 2 * rng.random() * X[i])
                else:
                    X[i] = (rabbit_location - X.mean(axis=0)) - rng.random() * ((ub - lb) * rng.random() + lb)
            else:
               
                sp = rng.random()
                e = escaping_energy
                if sp >= 0.5 and abs(e) < 0.5:
                    
                    X[i] = rabbit_location - e * np.abs(rabbit_location - X[i])
                elif sp >= 0.5 and abs(e) >= 0.5:
                    
                    jump = 2 * (1 - rng.random())
                    X[i] = (rabbit_location - X[i]) - e * np.abs(jump * rabbit_location - X[i])
                else:
                    
                    jump = 2 * (1 - rng.random())
                    ref = X[i] if abs(e) >= 0.5 else X.mean(axis=0)
                    base = rabbit_location - e * np.abs(jump * rabbit_location - ref)
                    X1 = sigmoid_binarize(base, rng)
                    if fobj(X1) < fobj(X[i]):
                        X[i] = X1
                    else:
                        X2 = sigmoid_binarize(base + rng.random(dim) * levy(dim, rng), rng)
                        if fobj(X2) < fobj(X[i]):
                            X[i] = X2

        t += 1
        curve[t - 1] = rabbit_energy
        if log:
            log(t, rabbit_energy)

    return rabbit_location, curve, fitness00.copy()
