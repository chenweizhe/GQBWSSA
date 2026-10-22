import numpy as np


def read_csv(path):
    
    try:
        import cudf
        import cupy as cp

        data = cudf.read_csv(path).astype("float64").to_cupy()
        return data, cp
    except ImportError:
        import pandas as pd

        return pd.read_csv(path).to_numpy(dtype=np.float64), np


def _mapminmax(xp, xtrain, xvalid, ymin=0.0, ymax=1.0):
    
    xmin = xtrain.min(axis=0)
    xmax = xtrain.max(axis=0)
    span = xmax - xmin
    gain = xp.where(span > 0, (ymax - ymin) / xp.where(span > 0, span, 1.0), 1.0)
    return (xtrain - xmin) * gain + ymin, (xvalid - xmin) * gain + ymin


def load_dataset(path, rng, train_ratio=0.7):
    
    res, xp = read_csv(path)
    r = rng.permutation(res.shape[0])
    if xp is not np:
        r = xp.asarray(r)
    train_num = int(np.floor(len(r) * train_ratio))
    train_data = res[r[:train_num]]
    test_data = res[r[train_num:]]

    xtrain, ytrain = train_data[:, :-1], train_data[:, -1]
    xvalid, yvalid = test_data[:, :-1], test_data[:, -1]
    xtrain, xvalid = _mapminmax(xp, xtrain, xvalid, 0.0, 1.0)

    dtype = xp.float32
    xtrain = xp.ascontiguousarray(xtrain, dtype=dtype)
    xvalid = xp.ascontiguousarray(xvalid, dtype=dtype)
    ytrain = xp.ascontiguousarray(ytrain, dtype=xp.int32)
    yvalid = xp.ascontiguousarray(yvalid, dtype=xp.int32)
    del res, train_data, test_data
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    return xtrain, xvalid, ytrain, yvalid
