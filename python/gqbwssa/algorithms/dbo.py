
import time

import numpy as np

from ..utils import binarize, bounds
from .common import MultiRunRecorder


def dbo(popsize, max_iteration, lb, ub, dim, fobj, multi_num, fitness_obj, rng):
    rec = MultiRunRecorder()

    for ii in range(multi_num):
        fitness_obj.clear_cache()
        t0 = time.perf_counter()
        rol_num = int(round(0.25 * popsize))
        bab_num = int(round(0.25 * popsize))
        egg_num = int(round(0.25 * popsize))

        x = np.zeros((popsize, dim))
        fit = np.zeros(popsize)
        for i in range(popsize):
            x[i] = (rng.random(dim) > 0.5).astype(float)
            fit[i] = fobj(x[i])
        p_fit = fit.copy()
        pX = x.copy()
        XX = pX.copy()
        best_i = int(np.argmin(fit))
        f_min = fit[best_i]
        best_x = x[best_i].copy()

        ex = np.zeros((popsize, dim))
        curve = np.zeros(max_iteration)
        for t in range(1, max_iteration + 1):
            print(f"  [DBO] run {ii + 1}/{multi_num}  iter {t}/{max_iteration}  best={f_min:.6f}", flush=True)
            worse = x[int(np.argmax(fit))].copy()

            r2 = rng.random()
            for i in range(rol_num):
                if r2 < 0.9:
                    a = 1 if rng.random() > 0.1 else -1
                    ex[i] = pX[i] + 0.3 * np.abs(pX[i] - worse) + a * 0.1 * XX[i]
                else:
                    aaa = int(rng.integers(1, 181))
                    theta = aaa * np.pi / 180
                    ex[i] = pX[i] + np.tan(theta) * np.abs(pX[i] - XX[i])
                ex[i] = bounds(ex[i], lb, ub)
                x[i] = binarize(ex[i], rng)
                fit[i] = fobj(x[i])

            best_xx = x[int(np.argmin(fit))].copy()

            R = 1 - t / max_iteration
            xnew1 = bounds(best_xx * (1 - R), lb, ub)
            xnew2 = bounds(best_xx * (1 + R), lb, ub)
            xnew11 = bounds(best_x * (1 - R), lb, ub)
            xnew22 = bounds(best_x * (1 + R), lb, ub)

            for i in range(rol_num, rol_num + bab_num):
                ex[i] = best_xx + rng.random(dim) * (pX[i] - xnew1) + rng.random(dim) * (pX[i] - xnew2)
                ex[i] = bounds(ex[i], xnew1, xnew2)
                x[i] = binarize(ex[i], rng)
                fit[i] = fobj(x[i])

            for i in range(rol_num + bab_num, rol_num + bab_num + egg_num):
                ex[i] = pX[i] + rng.standard_normal() * (pX[i] - xnew11) + rng.random(dim) * (pX[i] - xnew22)
                ex[i] = bounds(ex[i], xnew1, xnew2)
                x[i] = binarize(ex[i], rng)
                fit[i] = fobj(x[i])

            for j in range(rol_num + bab_num + egg_num, popsize):
                ex[j] = best_x + rng.standard_normal(dim) * (np.abs(pX[j] - best_xx) + np.abs(pX[j] - best_x)) / 2
                ex[j] = bounds(ex[j], xnew1, xnew2)
                x[j] = binarize(ex[j], rng)
                fit[j] = fobj(x[j])

            XX = pX.copy()
            for i in range(popsize):
                if fit[i] < p_fit[i]:
                    p_fit[i] = fit[i]
                    pX[i] = x[i]
                if p_fit[i] < f_min:
                    f_min = p_fit[i]
                    best_x = pX[i].copy()

            curve[t - 1] = f_min

        rec.add(fitness_obj, best_x, curve, t0)

    print("DBO finished", flush=True)
    return rec.result()
