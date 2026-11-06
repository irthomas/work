# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:06:40 2026

@author: iant

BASELINE FIT
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam=250.0, p=0.95, niter=10):

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z
