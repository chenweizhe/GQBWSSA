
import time

import numpy as np

from ..utils import binarize, initialization, levy
from .common import MultiRunRecorder


def hho(popsize, iters, lb, ub, dim, fobj, multi_num, fitness_obj, rng):
    rec = MultiRunRecorder()

    for ii in range(multi_num):
        fitness_obj.clear_cache()
        t0 = time.perf_counter()
        rabbit_location = np.zeros(dim)
        rabbit_energy = np.inf

        X = initialization(popsize, ub, lb, dim, rng)
        curve = np.zeros(iters)
        fitness00 = np.zeros(popsize)

        for t in range(iters):
            print(f"  [HHO] run {ii + 1}/{multi_num}  iter {t + 1}/{iters}  best={rabbit_energy:.6f}", flush=True)
            for i in range(popsize):
                X[i] = np.clip(X[i], lb, ub)
                X[i] = binarize(X[i], rng)
                fitness = fobj(X[i])
                fitness00[i] = fitness
                if fitness < rabbit_energy:
                    rabbit_energy = fitness
                    rabbit_location = X[i].copy()

            E1 = 2 * (1 - t / iters)
            for i in range(popsize):
                E0 = 2 * rng.random() - 1
                esc = E1 * E0

                if abs(esc) >= 1:
                    q = rng.random()
                    x_rand = X[int(np.floor(popsize * rng.random()))].copy()
                    if q < 0.5:
                        X[i] = x_rand - rng.random() * np.abs(x_rand - 2 * rng.random() * X[i])
                    else:
                        X[i] = (rabbit_location - X.mean(axis=0)) - rng.random() * ((ub - lb) * rng.random() + lb)
                    continue

                sp = rng.random()
                if sp >= 0.5 and abs(esc) < 0.5:
                    X[i] = rabbit_location - esc * np.abs(rabbit_location - X[i])

                if sp >= 0.5 and abs(esc) >= 0.5:
                    jump = 2 * (1 - rng.random())
                    X[i] = (rabbit_location - X[i]) - esc * np.abs(jump * rabbit_location - X[i])

                if sp < 0.5:
                    jump = 2 * (1 - rng.random())
                    ref = X[i] if abs(esc) >= 0.5 else X.mean(axis=0)
                    base = rabbit_location - esc * np.abs(jump * rabbit_location - ref)
                    X1 = binarize(base, rng)
                    f1 = fobj(X1)
                    fx = fobj(X[i])
                    if f1 < fx:
                        X[i] = X1
                    else:
                        X2 = binarize(base + rng.random(dim) * levy(dim, rng), rng)
                        if fobj(X2) < fx:
                            X[i] = X2

            curve[t] = rabbit_energy

        rec.add(fitness_obj, rabbit_location, curve, t0)

    print("HHO finished", flush=True)
    return rec.result()
