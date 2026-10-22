
from __future__ import annotations

import time
import warnings

import numpy as np

from .data import Dataset


class RFFitness:
    def __init__(self, data: Dataset, rng: np.random.Generator, *, alpha: float = 0.99,
                 beta: float = 0.01, ntree: int = 20, mtry: int = 2, max_depth: int | None = 16,
                 n_bins: int = 128, n_streams: int = 10, backend: str = "cuml",
                 use_cache: bool = True, matlab_ratio: bool = True, n_jobs: int = -1):
        self.data = data
        self.rng = rng
        self.alpha = alpha
        self.beta = beta
        self.ntree = ntree
        self.mtry = mtry
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.n_streams = n_streams
        self.backend = backend
        self.use_cache = use_cache
        self.n_jobs = n_jobs
        self.dim = data.dim
       
        self.ratio_den = self.dim + 1 if matlab_ratio else self.dim

        if backend == "cuml":
            if not data.on_gpu:
                raise ValueError("cuml 后端需要以 GPU 模式加载数据")
            import cupy as cp
            from cuml.ensemble import RandomForestClassifier
            self._xp = cp
            self._RF = RandomForestClassifier
        elif backend == "sklearn":
            from sklearn.ensemble import RandomForestClassifier
            self._xp = np
            self._RF = RandomForestClassifier
            if data.on_gpu:
                import cupy as cp
                self.data = Dataset(cp.asnumpy(data.xtrain), cp.asnumpy(data.xvalid),
                                    cp.asnumpy(data.ytrain), cp.asnumpy(data.yvalid),
                                    data.feature_names, False)
        else:
            raise ValueError(f"未知后端: {backend}")

        self._cache: dict[bytes, tuple[float, float, float]] = {}
        self.n_calls = 0
        self.n_train = 0
        self.train_time = 0.0

    def reset_stats(self):
        self.n_calls = 0
        self.n_train = 0
        self.train_time = 0.0

    def clear_cache(self):
        self._cache.clear()

    def _mask(self, x) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(-1)[: self.dim] > 0.5

    def _train_eval(self, mask: np.ndarray) -> float:
       
        xp = self._xp
        cols = xp.asarray(np.flatnonzero(mask))
        k = int(mask.sum())
        mtry = min(self.mtry, k)
        seed = int(self.rng.integers(0, 2**31 - 1))
        d = self.data
        xtr = d.xtrain[:, cols]
        xva = d.xvalid[:, cols]

        t0 = time.perf_counter()
        if self.backend == "cuml":
            model = self._RF(n_estimators=self.ntree, max_features=mtry,
                             max_depth=self.max_depth if self.max_depth is not None else 32,
                             n_bins=self.n_bins, n_streams=self.n_streams,
                             random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(xtr, d.ytrain)
                pred = model.predict(xva)
            acc = float((xp.asarray(pred) == d.yvalid).mean())
        else:
            model = self._RF(n_estimators=self.ntree, max_features=mtry,
                             max_depth=self.max_depth, n_jobs=self.n_jobs,
                             random_state=seed)
            model.fit(xtr, d.ytrain)
            acc = float((model.predict(xva) == d.yvalid).mean())
        self.train_time += time.perf_counter() - t0
        self.n_train += 1
        return acc

    def full(self, x, use_cache: bool | None = None):
        
        self.n_calls += 1
        mask = self._mask(x)
        k = int(mask.sum())
        if k == 0:
            return (np.inf, np.inf, np.inf)

        use_cache = self.use_cache if use_cache is None else use_cache
        key = mask.tobytes()
        if use_cache and key in self._cache:
            return self._cache[key]

        acc = self._train_eval(mask)
        ratio = k / self.ratio_den
        fit = self.alpha * (1 - acc) + self.beta * ratio
        res = (fit, acc, ratio)
        if self.use_cache:
            self._cache[key] = res
        return res

    def __call__(self, x) -> float:
       
        return self.full(x)[0]

    def batch(self, pop) -> np.ndarray:
       
        return np.array([self(p) for p in np.atleast_2d(pop)], dtype=float)
