
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    xtrain: object  # cupy.ndarray（GPU）或 numpy.ndarray（CPU）
    xvalid: object
    ytrain: object
    yvalid: object
    feature_names: list
    on_gpu: bool

    @property
    def dim(self) -> int:
        return int(self.xtrain.shape[1])


def _minmax_fit_apply(xp, xtrain, xvalid):
    
    xmin = xtrain.min(axis=0)
    xmax = xtrain.max(axis=0)
    rng = xmax - xmin
    rng = xp.where(rng == 0, 1, rng)
    return (xtrain - xmin) / rng, (xvalid - xmin) / rng


def load_dataset(path: str, train_ratio: float = 0.7, use_gpu: bool = True,
                 rng: np.random.Generator | None = None, max_rows: int | None = None,
                 label_col: str | None = None) -> Dataset:
    
    rng = rng or np.random.default_rng()

    if use_gpu:
        import cudf
        import cupy as xp
        df = cudf.read_csv(path)
    else:
        import pandas as pd
        xp = np
        df = pd.read_csv(path)

    if max_rows is not None and max_rows < len(df):
        df = df.iloc[:max_rows]

    label_col = label_col or df.columns[-1]
    feature_names = [c for c in df.columns if c != label_col]

    x = df[feature_names].astype("float32").values
    y = df[label_col].astype("int32").values
    if use_gpu:
        x, y = xp.asarray(x), xp.asarray(y)

    n = x.shape[0]
    perm = xp.asarray(rng.permutation(n))
    train_num = int(np.floor(n * train_ratio))
    tr, te = perm[:train_num], perm[train_num:]

    xtrain, xvalid = _minmax_fit_apply(xp, x[tr], x[te])
    return Dataset(
        xtrain=xp.ascontiguousarray(xtrain, dtype=xp.float32),
        xvalid=xp.ascontiguousarray(xvalid, dtype=xp.float32),
        ytrain=xp.ascontiguousarray(y[tr]),
        yvalid=xp.ascontiguousarray(y[te]),
        feature_names=feature_names,
        on_gpu=use_gpu,
    )
