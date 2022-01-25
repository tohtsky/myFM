"""
A simple example for ordred probit regression,
taken from the "MCMCoprobit" document of MCMCpack:
https://rdrr.io/cran/MCMCpack/man/MCMCoprobit.html
"""

import numpy as np

from myfm import MyFMOrderedProbit

N_DATA = 100

rns = np.random.RandomState(42)
X = rns.randn(N_DATA, 2)
z = 1 + X[:, 0] * 0.1 - X[:, 1] * 0.5 + rns.randn(N_DATA)

y = z.copy()
y[z < 0] = 0
y[(z >= 0) & (z < 1)] = 1
y[(z >= 1) & (z < 1.5)] = 2
y[z >= 1.5] = 3

# Faster than MCMCoprobit by 40x, in my environment.
fm = MyFMOrderedProbit(0, random_seed=42).fit(
    X,
    y,
    n_iter=11000,
    n_kept_samples=10000,
)

c0 = np.asfarray([s.cutpoints[0] for s in fm.predictor_.samples])
print(c0.mean(axis=0))
