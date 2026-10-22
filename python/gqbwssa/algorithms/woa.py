
import time

import numpy as np

from ..utils import binarize
from .common import MultiRunRecorder


def woa(popsize, maxgen, lb, ub, dim, fobj, multi_num, fitness_obj, rng):
    lb = lb * np.ones(dim)
    ub = ub * np.ones(dim)
    rec = MultiRunRecorder()

    for ii in range(multi_num):
        fitness_obj.clear_cache()
        t0 = time.perf_counter()

        fitness = np.zeros(popsize)
        positions = np.zeros((popsize, dim))
        for i in range(popsize):
            positions[i] = (rng.random(dim) > 0.5).astype(float)
            fitness[i] = fobj(positions[i])
        order = np.argsort(fitness, kind="stable")
        leader_pos = positions[order[0]].copy()
        leader_score = fitness[order[0]]

        curve = np.zeros(maxgen)
        epositions = np.zeros((popsize, dim))
        for t in range(1, maxgen + 1):
            print(f"  [WOA] run {ii + 1}/{multi_num}  iter {t}/{maxgen}  best={leader_score:.6f}", flush=True)
            a = 2 - t * (2 / maxgen)
            a2 = -1 + t * (-1 / maxgen)

            for i in range(popsize):
                r1, r2 = rng.random(), rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * rng.random() + 1

                p = rng.random(dim)
                if abs(A) >= 1:
                    idx = rng.integers(0, popsize, size=dim)
                    x_rand = positions[idx, np.arange(dim)]
                    shrink = x_rand - A * np.abs(C * x_rand - positions[i])
                else:
                    shrink = leader_pos - A * np.abs(C * leader_pos - positions[i])
                spiral = np.abs(leader_pos - positions[i]) * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos
                epositions[i] = np.where(p < 0.5, shrink, spiral)

            for i in range(popsize):
                epositions[i] = np.clip(epositions[i], lb, ub)
                positions[i] = binarize(epositions[i], rng)
                fitness[i] = fobj(positions[i])
                if fitness[i] < leader_score:
                    leader_score = fitness[i]
                    leader_pos = positions[i].copy()

            curve[t - 1] = leader_score

        rec.add(fitness_obj, leader_pos, curve, t0)

    print("WOA finished", flush=True)
    return rec.result()
