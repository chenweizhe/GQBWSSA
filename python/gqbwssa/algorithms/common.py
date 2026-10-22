import time

import numpy as np


class MultiRunRecorder:
    

    def __init__(self):
        self.acc, self.ratio, self.time, self.curve, self.best = [], [], [], [], []

    def add(self, fitness_obj, best_pos, curve, t0):
        
        _, acc, ratio = fitness_obj.evaluate(best_pos, use_cache=False)
        self.acc.append(acc)
        self.ratio.append(ratio)
        self.time.append(time.perf_counter() - t0)
        self.curve.append(np.asarray(curve, dtype=float))
        self.best.append((np.asarray(best_pos, dtype=float) > 0.5).astype(float))

    def result(self):
        res1 = np.vstack([self.acc, self.ratio, self.time])
        return res1, np.vstack(self.curve), np.vstack(self.best)
