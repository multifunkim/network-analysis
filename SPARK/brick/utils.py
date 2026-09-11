#!/usr/bin/env python3
import numpy as np
import math

def description_length(x, D, a):
    """
    Per‐voxel MDL: data‐fit + sparse‐code penalty (in bits).
    x : (T,)       time‐series
    D : (T, K)     dictionary
    a : (K,)       sparse code
    """
    resid = x - D @ a
    T = x.size
    sigma2 = np.mean(resid**2)
    sigma2 = max(sigma2, 1e-12)
    # data term (in bits)
    L_data = 0.5 * T * math.log2(2 * math.pi * sigma2) \
             + np.sum(resid**2) / (2 * sigma2) / math.log(2)
    # sparse penalty: k*(log2(K)+16)
    k = np.count_nonzero(a)
    if k == 0:
        return np.inf
    K = D.shape[1]
    L_sparse = k * (math.log2(K) + 16)
    return L_data + L_sparse
