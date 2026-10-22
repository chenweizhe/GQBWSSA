
from __future__ import annotations

import math

import numpy as np


def bounds(s, lb, ub):
    
    s = np.array(s, dtype=float, copy=True)
    lb = np.broadcast_to(lb, s.shape)
    ub = np.broadcast_to(ub, s.shape)
    i = s < lb
    s[i] = lb[i]
    j = s > ub
    s[j] = ub[j]
    return s


def sigmoid_binarize(x, rng: np.random.Generator):
    
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        prob = 1.0 / (1.0 + np.exp(x))
    return (prob > rng.random(x.shape)).astype(float)


def levy(d: int, rng: np.random.Generator):
    
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.standard_normal(d) * sigma
    v = rng.standard_normal(d)
    return u / np.abs(v) ** (1 / beta)


def initialization(pop: int, ub, lb, dim: int, rng: np.random.Generator):
   
    ub = np.broadcast_to(ub, (dim,))
    lb = np.broadcast_to(lb, (dim,))
    return (ub - lb) * rng.random((pop, dim)) + lb


def _primes(n: int):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    return np.flatnonzero(sieve)


def goodnode(m: int, n: int):
    
    tmp1 = np.arange(1, m + 1)[:, None] * np.ones((1, n))
    ind = np.arange(1, n + 1)
    prime1 = _primes(100 * n)
    p = prime1[prime1 >= (2 * n + 3)][0]
    tmp2 = 2 * np.cos((2 * np.pi * ind) / p)
    gd = tmp1 * tmp2[None, :]
    return np.mod(gd, 1)


def goodnode_initial(pop: int, ub, lb, dim: int, rng: np.random.Generator):
    
    ub = np.broadcast_to(ub, (dim,))
    lb = np.broadcast_to(lb, (dim,))
    x = (ub - lb) * goodnode(pop, dim) + lb
    for i in range(pop):
        x[i] = sigmoid_binarize(x[i], rng)
    return x


def q_g_initialization(pop: int, dim: int, ub, lb, fobj, rng: np.random.Generator):
    
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    x = goodnode_initial(pop, ub, lb, dim, rng)

    
    theta = np.arcsin(2 * (x - lb) / (ub - lb) - 1)
    x2 = 0.5 * (1 + np.cos(theta)) * (ub - lb) + lb
    for i in range(pop):
        x2[i] = bounds(x2[i], lb, ub)
        x2[i] = sigmoid_binarize(x2[i], rng)

    fit = fobj.batch(x)
    fit2 = fobj.batch(x2)
    better = fit2 < fit
    x[better] = x2[better]
    fit = np.where(better, fit2, fit)
    return x, fit
