
from __future__ import annotations

import time

import numpy as np

from .algorithms import ALGORITHMS
from .fitness import RFFitness


def run_algorithm(name: str, popsize: int, max_iter: int, lb, ub, dim: int,
                  fobj: RFFitness, multi_num: int, rng: np.random.Generator,
                  verbose: bool = True):
    
    algo = ALGORITHMS[name]
    acc = np.zeros(multi_num)
    ratio = np.zeros(multi_num)
    run_time = np.zeros(multi_num)
    conve = np.zeros((multi_num, max_iter))
    pos = np.zeros((multi_num, dim))
    n_calls = np.zeros(multi_num, dtype=int)
    n_train = np.zeros(multi_num, dtype=int)

    for ii in range(multi_num):
        
        fobj.clear_cache()
        fobj.reset_stats()
        t0 = time.perf_counter()

        def log(t, best, _ii=ii):
            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"[{name}] run {_ii + 1}/{multi_num}  iter {t:3d}/{max_iter}  "
                      f"best_fitness={best:.6f}  rf_trained={fobj.n_train}  "
                      f"elapsed={elapsed:.1f}s", flush=True)

        best_pos, curve, final_fit = algo(popsize, max_iter, lb, ub, dim, fobj, rng, log=log)

        
        _, a, r = fobj.full(best_pos, use_cache=False)
        run_time[ii] = time.perf_counter() - t0
        acc[ii], ratio[ii] = a, r
        conve[ii] = curve
        pos[ii] = best_pos
        n_calls[ii], n_train[ii] = fobj.n_calls, fobj.n_train

    return {
        "ind": np.vstack([acc, ratio, run_time]),
        "conve": conve,
        "pos": pos,
        "n_calls": n_calls,
        "n_train": n_train,
    }
