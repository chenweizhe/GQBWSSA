
import time

import numpy as np

from ..utils import binarize, boundary_check
from .common import MultiRunRecorder


def gwo(popsize, maxgen, lb, ub, dim, fobj, multi_num, fitness_obj, rng):
    rec = MultiRunRecorder()

    for ii in range(multi_num):
        fitness_obj.clear_cache()
        t0 = time.perf_counter()
        curve = np.zeros(maxgen)

        positions = np.zeros((popsize, dim))
        fitness = np.zeros(popsize)
        for i in range(popsize):
            positions[i] = (rng.random(dim) > 0.5).astype(float)
            fitness[i] = fobj(positions[i])

        order = np.argsort(fitness, kind="stable")
        alpha_pos, alpha_score = positions[order[0]].copy(), fitness[order[0]]
        beta_pos, beta_score = positions[order[1]].copy(), fitness[order[1]]
        delta_pos, delta_score = positions[order[2]].copy(), fitness[order[2]]

        epositions = np.zeros((popsize, dim))
        for t in range(1, maxgen + 1):
            print(f"  [GWO] run {ii + 1}/{multi_num}  iter {t}/{maxgen}  best={alpha_score:.6f}", flush=True)
            a = 2 - t * (2 / maxgen)
            for i in range(popsize):
                r = rng.random((6, dim))
                A1, C1 = 2 * a * r[0] - a, 2 * r[1]
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])
                A2, C2 = 5 * a * r[2] - a, 5 * r[3]
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])
                A3, C3 = 2 * a * r[4] - a, 2 * r[5]
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])
                epositions[i] = boundary_check((X1 + X2 + X3) / 3, lb, ub)

            for i in range(popsize):
                positions[i] = binarize(epositions[i], rng)
                fitness[i] = fobj(positions[i])
                if fitness[i] < alpha_score:
                    alpha_score, alpha_pos = fitness[i], positions[i].copy()
                if alpha_score < fitness[i] < beta_score:
                    beta_score, beta_pos = fitness[i], positions[i].copy()
                if fitness[i] > alpha_score and fitness[i] > beta_score and fitness[i] < delta_score:
                    delta_score, delta_pos = fitness[i], positions[i].copy()

            curve[t - 1] = alpha_score

        rec.add(fitness_obj, alpha_pos, curve, t0)

    print("GWO finished", flush=True)
    return rec.result()
