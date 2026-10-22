"""适应度函数（对应 Fitness_RF.m）。
"""

import time

import numpy as np


def _as_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class FitnessRF:
    def __init__(self, xtrain, ytrain, xvalid, yvalid, backend="gpu", ntree=20, mtry=2,
                 alpha=0.99, beta=0.01, n_bins=128, max_depth=16, n_streams=10,
                 seed=None, cache=True, verbose=False):
        self.dim = int(xtrain.shape[1])
        self.alpha = alpha
        self.beta = beta
        self.backend = backend
        self.ntree = ntree
        self.mtry = mtry
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.n_streams = n_streams
        self.seed = seed
        self.cache = cache
        self.verbose = verbose
        self.n_calls = 0
        self.n_trains = 0
        self.total_time = 0.0
        self._cache = {}
       
        self._rf_rng = np.random.default_rng(seed)

        if backend == "gpu":
            import cupy as cp

            self._cp = cp
            self.xtrain = cp.ascontiguousarray(xtrain, dtype=cp.float32)
            self.xvalid = cp.ascontiguousarray(xvalid, dtype=cp.float32)
            self.ytrain = cp.ascontiguousarray(ytrain, dtype=cp.int32)
            self.yvalid = cp.ascontiguousarray(yvalid, dtype=cp.int32)
            try:
                from cuml.ensemble import RandomForestClassifier

                self._RF = RandomForestClassifier
                self.engine = "cuml"
            except ImportError:
                from .rf_gpu import CupyRandomForest

                self.engine = "cupy"
                
                self.cache = False
                self.model = CupyRandomForest(ntree=ntree, mtry=mtry, n_bins=n_bins, seed=seed)
                y_np = _as_numpy(self.ytrain)
                self.model.prepare(_as_numpy(self.xtrain), y_np, _as_numpy(self.xvalid))
                raw = _as_numpy(self.yvalid).ravel()
                codes = np.searchsorted(self.model.classes, raw)
                known = np.isin(raw, self.model.classes)
                self._y_code = cp.asarray(np.where(known, codes, -1), dtype=cp.int32)
                cp.get_default_memory_pool().free_all_blocks()
        elif backend == "cpu":
            self.engine = "sklearn"
            self.xtrain = np.ascontiguousarray(_as_numpy(xtrain), dtype=np.float32)
            self.xvalid = np.ascontiguousarray(_as_numpy(xvalid), dtype=np.float32)
            self.ytrain = np.asarray(_as_numpy(ytrain)).ravel()
            self.yvalid = np.asarray(_as_numpy(yvalid)).ravel()
        else:
            raise ValueError(f"未知 backend: {backend}，可选 'gpu' 或 'cpu'")

    def clear_cache(self):
       
        self._cache.clear()

    def _accuracy(self, feat_idx, state):
        if self.engine == "cupy":
            pred = self.model.fit_predict(feat_idx)
            return float((pred == self._y_code).mean())

        k = len(feat_idx)
        if self.engine == "cuml":
            cp = self._cp
            cols = cp.asarray(feat_idx)
            model = self._RF(
                n_estimators=self.ntree,
                max_features=min(self.mtry, k),
                max_depth=self.max_depth,
                n_bins=self.n_bins,
                n_streams=self.n_streams,
                random_state=state,
            )
            model.fit(self.xtrain[:, cols], self.ytrain)
            pred = model.predict(self.xvalid[:, cols])
            return float((cp.asarray(pred) == self.yvalid).mean())

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(
            n_estimators=self.ntree, max_features=min(self.mtry, k),
            max_depth=self.max_depth, bootstrap=True, n_jobs=-1, random_state=state)
        clf.fit(self.xtrain[:, feat_idx], self.ytrain)
        return float(np.mean(clf.predict(self.xvalid[:, feat_idx]) == self.yvalid))

    def evaluate(self, x, use_cache=True):
        
        mask = np.asarray(x, dtype=float).ravel() > 0.5
        n_sel = int(mask.sum())
        if n_sel == 0:
            return np.inf, np.inf, np.inf

        
        state = None if self.engine == "cupy" else int(self._rf_rng.integers(2**31 - 1))
        key = mask.tobytes()
        if use_cache and self.cache and key in self._cache:
            self.n_calls += 1
            fit, acc, ratio = self._cache[key]
            if self.verbose:
                print(f"    [fitness #{self.n_calls}] n_features={n_sel:2d}  ACC={acc:.5f}  cache", flush=True)
            return fit, acc, ratio

        t0 = time.perf_counter()
        acc = self._accuracy(np.nonzero(mask)[0], state)
        dt = time.perf_counter() - t0
        self.n_calls += 1
        self.n_trains += 1
        self.total_time += dt
        ratio = n_sel / (self.dim + 1)
        fit = self.alpha * (1 - acc) + self.beta * ratio
        if self.cache:
            self._cache[key] = (fit, acc, ratio)
        if self.verbose:
            print(f"    [fitness #{self.n_calls}] n_features={n_sel:2d}  ACC={acc:.5f}  {dt:.2f}s", flush=True)
        return fit, acc, ratio

    def __call__(self, x):
        return self.evaluate(x)[0]
