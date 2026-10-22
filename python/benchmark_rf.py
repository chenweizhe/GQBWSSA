

import argparse
import os
import sys
import time

if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

import numpy as np

from gqbwssa.data import load_dataset
from gqbwssa.fitness import FitnessRF

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "qifenlei_ct_quan.csv"))
    p.add_argument("--no-cpu", action="store_true", help="不跑 sklearn 对照")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    t = time.perf_counter()
    xtr, xva, ytr, yva = load_dataset(args.data, rng)
    print(f"Data loaded in {time.perf_counter() - t:.1f}s  train {xtr.shape}  test {xva.shape}")
    y_np = ytr.get() if hasattr(ytr, "get") else np.asarray(ytr)
    classes, counts = np.unique(y_np, return_counts=True)
    print("Class counts:", dict(zip(classes.astype(int).tolist(), counts.tolist())))

    t = time.perf_counter()
    gpu = FitnessRF(xtr, ytr, xva, yva, backend="gpu", seed=0, cache=False)
    print(f"GPU prepare {time.perf_counter() - t:.1f}s  engine={gpu.engine}")
    backends = [("gpu", gpu)]
    if not args.no_cpu:
        backends.append(("cpu", FitnessRF(xtr, ytr, xva, yva, backend="cpu", seed=0, cache=False)))

    dim = xtr.shape[1]
    masks = [np.ones(dim), (rng.random(dim) > 0.5).astype(float), (rng.random(dim) > 0.8).astype(float)]
    gpu.evaluate(masks[0])
    for mk in masks:
        for name, f in backends:
            times, accs = [], []
            for _ in range(args.repeat):
                t = time.perf_counter()
                _, acc, _ = f.evaluate(mk)
                times.append(time.perf_counter() - t)
                accs.append(acc)
            print(f"{name}  n_features={int(mk.sum()):2d}  ACC={np.mean(accs):.5f}±{np.std(accs):.5f}  "
                  f"mean_time={np.mean(times):.2f}s")


if __name__ == "__main__":
    main()
