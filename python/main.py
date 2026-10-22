from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from gqbwssa.algorithms import ALGORITHMS  # noqa: E402
from gqbwssa.data import load_dataset  # noqa: E402
from gqbwssa.fitness import RFFitness  # noqa: E402
from gqbwssa.results import plot_convergence, save_summary, summarize  # noqa: E402
from gqbwssa.runner import run_algorithm  # noqa: E402

_CONSOLE_COLUMNS = {
    "序号": "id",
    "算法": "algorithm",
    "平均准确度": "mean_accuracy",
    "平均特征选择率": "mean_selection_ratio",
    "平均运行时间(s)": "mean_time_s",
    "最佳适应度": "best_fitness",
    "最差适应度值": "worst_fitness",
    "标准差": "std",
    "平均适应度": "mean_fitness",
    "最佳子集特征数": "best_subset_size",
}


def parse_args():
    p = argparse.ArgumentParser(description="GPU")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "ciciot.csv"))
    p.add_argument("--algos", nargs="+", default=["GWO", "WOA", "DBO", "HHO", "GQBWSSA"],
                   choices=list(ALGORITHMS))
    p.add_argument("--multi-num", type=int, default=5)
    p.add_argument("--popsize", type=int, default=20)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--lb", type=float, default=-6.0)
    p.add_argument("--ub", type=float, default=6.0)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    g = p.add_argument_group("rf/fn")
    g.add_argument("--backend", choices=["cuml", "sklearn"], default="cuml")
    g.add_argument("--ntree", type=int, default=20)
    g.add_argument("--mtry", type=int, default=2)
    g.add_argument("--max-depth", type=int, default=16)
    g.add_argument("--n-bins", type=int, default=128)
    g.add_argument("--n-streams", type=int, default=10)
    g.add_argument("--alpha", type=float, default=0.99)
    g.add_argument("--beta", type=float, default=0.01)
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--exact-ratio", action="store_true")

    o = p.add_argument_group("输出")
    o.add_argument("--out", default=None)
    o.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_name = os.path.basename(args.data)
    tag = os.path.splitext(dataset_name)[0]
    out_dir = args.out or os.path.join(ROOT, "results", f"{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    use_gpu = args.backend == "cuml"

    t0 = time.perf_counter()
    data = load_dataset(args.data, train_ratio=args.train_ratio, use_gpu=use_gpu, rng=rng,
                        max_rows=args.max_rows)
    device = "GPU" if use_gpu else "CPU"
    print(f"data loaded: {dataset_name}  train {tuple(data.xtrain.shape)}  "
          f"test {tuple(data.xvalid.shape)}  features {data.dim}  device {device}  "
          f"elapsed {time.perf_counter() - t0:.1f}s")

    max_depth = None if (args.backend == "sklearn" and args.max_depth == 0) else args.max_depth
    fobj = RFFitness(data, rng, alpha=args.alpha, beta=args.beta, ntree=args.ntree,
                     mtry=args.mtry, max_depth=max_depth, n_bins=args.n_bins,
                     n_streams=args.n_streams, backend=args.backend,
                     use_cache=not args.no_cache, matlab_ratio=not args.exact_ratio)
    dim = data.dim
    lb = args.lb * np.ones(dim)
    ub = args.ub * np.ones(dim)

    results = {}
    for name in args.algos:
        print(f"\n================ start {name} ================")
        results[name] = run_algorithm(name, args.popsize, args.max_iter, lb, ub, dim, fobj,
                                      args.multi_num, rng, verbose=not args.quiet)
        print(f"================ {name} done ================")

    df = summarize(results, data.feature_names, dataset_name)
    xlsx = save_summary(df, results, out_dir, tag)
    pngs = plot_convergence(results, out_dir, tag, dataset_name)

    show = df.drop(columns=["最佳特征子集(0/1)", "最佳特征子集(名称)", "数据集"]).rename(
        columns=_CONSOLE_COLUMNS)
    import pandas as pd
    with pd.option_context("display.max_columns", 20, "display.width", 220):
        print("\n" + show.to_string(index=False))
    print(f"\nsaved to: {out_dir}\n  {xlsx}\n  " + "\n  ".join(pngs))
    print(f"total elapsed {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
