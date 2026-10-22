

import math

import numpy as np


def bounds(s, lb, ub):
    
    s = np.where(s < lb, lb, s)
    return np.where(s > ub, ub, s)


boundary_check = bounds


def binarize(x, rng):
    
    with np.errstate(over="ignore"):
        prob = 1.0 / (1.0 + np.exp(x))
    return (prob > rng.random(np.shape(x))).astype(float)


def matlab_round(x):
    
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def levy(d, rng):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal(d) * sigma
    v = rng.standard_normal(d)
    return u / np.abs(v) ** (1 / beta)


def initialization(pop, ub, lb, dim, rng):
    return (ub - lb) * rng.random((pop, dim)) + lb


def _primes(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0]


def goodnode(M, N):
    
    tmp1 = np.arange(1, M + 1)[:, None] * np.ones((1, N))
    ind = np.arange(1, N + 1)
    prime1 = _primes(100 * N)
    p = prime1[prime1 >= (2 * N + 3)][0]
    tmp2 = 2 * np.cos(2 * np.pi * ind / p)
    return np.mod(tmp1 * tmp2[None, :], 1)


def goodnode_initial(pop, ub, lb, dim, rng):
    gd = goodnode(pop, dim)
    X = (ub - lb) * gd + lb
    return binarize(X, rng)


def q_g_initialization(pop, dim, ub, lb, fobj, rng):
    
    x = goodnode_initial(pop, ub, lb, dim, rng)
    fit = np.ones(pop)
    for i in range(pop):
        fit[i] = fobj(x[i])
        theta = np.arcsin(2 * (x[i] - lb) / (ub - lb) - 1)
        x2 = 0.5 * (1 + np.cos(theta)) * (ub - lb) + lb
        x2 = bounds(x2, lb, ub)
        x2 = binarize(x2, rng)
        fit2 = fobj(x2)
        if fit2 < fit[i]:
            x[i] = x2
            fit[i] = fit2
    return x, fit
