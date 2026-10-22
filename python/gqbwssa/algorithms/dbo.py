from __future__ import annotations

import numpy as np

from ..utils import bounds, sigmoid_binarize


def dbo(N, max_iter, lb, ub, dim, fobj, rng: np.random.Generator, log=None):
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    rol_num = int(round(0.25 * N))  
    bab_num = int(round(0.25 * N))  
    egg_num = int(round(0.25 * N))  

    x = (rng.random((N, dim)) > 0.5).astype(float)
    fit = fobj.batch(x)
    p_fit = fit.copy()
    p_x = x.copy()
    xx = p_x.copy()
    best_i = int(np.argmin(fit))
    f_min = fit[best_i]
    best_x = x[best_i].copy()

    ex = np.zeros((N, dim))
    curve = np.zeros(max_iter)
    for t in range(1, max_iter + 1):
        worse = x[int(np.argmax(fit))].copy()

        
        r2 = rng.random()
        for i in range(rol_num):
            if r2 < 0.9:
                rng.random()  
                a = 1 if rng.random() > 0.1 else -1
                ex[i] = p_x[i] + 0.3 * np.abs(p_x[i] - worse) + a * 0.1 * xx[i]
            else:
                aaa = int(rng.integers(1, 181))
                theta = aaa * np.pi / 180
                ex[i] = p_x[i] + np.tan(theta) * np.abs(p_x[i] - xx[i])
            ex[i] = bounds(ex[i], lb, ub)
            x[i] = sigmoid_binarize(ex[i], rng)
        fit[:rol_num] = fobj.batch(x[:rol_num])

        best_ii = int(np.argmin(fit))
        best_xx = x[best_ii].copy()

        R = 1 - t / max_iter
        xnew1 = bounds(best_xx * (1 - R), lb, ub)
        xnew2 = bounds(best_xx * (1 + R), lb, ub)
        xnew11 = bounds(best_x * (1 - R), lb, ub)
        xnew22 = bounds(best_x * (1 + R), lb, ub)

        s = rol_num
        
        for i in range(s, s + bab_num):
            ex[i] = best_xx + rng.random(dim) * (p_x[i] - xnew1) + rng.random(dim) * (p_x[i] - xnew2)
            ex[i] = bounds(ex[i], xnew1, xnew2)
            x[i] = sigmoid_binarize(ex[i], rng)
       
        for i in range(s + bab_num, s + bab_num + egg_num):
            ex[i] = p_x[i] + rng.standard_normal() * (p_x[i] - xnew11) + rng.random(dim) * (p_x[i] - xnew22)
            ex[i] = bounds(ex[i], xnew1, xnew2)
            x[i] = sigmoid_binarize(ex[i], rng)
        
        for j in range(s + bab_num + egg_num, N):
            ex[j] = best_x + rng.standard_normal(dim) * (np.abs(p_x[j] - best_xx) + np.abs(p_x[j] - best_x)) / 2
            ex[j] = bounds(ex[j], xnew1, xnew2)
            x[j] = sigmoid_binarize(ex[j], rng)
        fit[s:] = fobj.batch(x[s:])

        
        xx = p_x.copy()
        for i in range(N):
            if fit[i] < p_fit[i]:
                p_fit[i] = fit[i]
                p_x[i] = x[i]
            if p_fit[i] < f_min:
                f_min = p_fit[i]
                best_x = p_x[i].copy()

        curve[t - 1] = f_min
        if log:
            log(t, f_min)

    return best_x, curve, fit.copy()
