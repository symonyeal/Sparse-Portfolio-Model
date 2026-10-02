"""F2: mean-absolute-deviation selection over X(K), a MILP solved by HiGHS.

Konno and Yamazaki (1991), eqs. (3.2)-(3.6), with X(K) in place of 0 <= x_j <= u_j:

    min (1/S) sum_s d_s   s.t.  d_s + a_s'w >= 0,  d_s - a_s'w >= 0,  mu'w >= rho,  (w, z) in X(K)

where a_s = r_s - mu and mu is the sample mean (their eq. (3.2)). At an optimum d_s = |a_s'w|.
The source's epigraph form replaces the split d+_s - d-_s = y_s of the specification; the two
are equivalent and this one is the paper's.
"""

import numpy as np
from scipy import sparse

from pf import crt, rw

# Legend
# r      : scenario returns (S, N); S scenarios, N assets
# mu     : sample mean of r (Konno-Yamazaki eq. (3.2))
# a      : centred returns a_s = r_s - mu
# d      : epigraph variables d_s >= |a_s'w|
# K, e, dl, sn : cardinality limit, floors, ceilings, cardinality sense (pf/rw.py)
# rho    : required mean return (None: no return row)
# tl, gp : time limit (s) and requested relative gap


def mad(r, K, rho=None, e=0.0, dl=1.0, sn="<=", tl=None, gp=0.0):
    """Solve F2; returns w, z, the recomputed deviation and the certificate."""
    r = np.asarray(r, float)
    S, N = r.shape
    mu = r.mean(axis=0)
    a = r - mu
    B = rw.pad(rw.X(N, K, e, dl, sn), S)
    if rho is not None:
        B = rw.stack(B, rw.pad(rw.R(mu, rho, ">="), S))
    I = sparse.identity(S, format="csr")
    Z = sparse.csr_matrix((S, N))
    D = sparse.vstack([sparse.hstack([a, Z, I]), sparse.hstack([-a, Z, I])], format="csr")
    A = sparse.vstack([B["A"], D], format="csr")
    lo = np.r_[B["lo"], np.zeros(2 * S)]
    hi = np.r_[B["hi"], np.full(2 * S, np.inf)]
    c = np.r_[np.zeros(2 * N), np.full(S, 1.0 / S)]
    x, c0 = crt.mi(c, A, lo, hi, np.r_[B["lb"], np.zeros(S)], np.r_[B["ub"], np.full(S, np.inf)],
                   np.r_[B["it"], np.zeros(S)], tl=tl, gp=gp)
    if x is None:
        return {"w": None, "z": None, "f": np.nan, "c": c0}
    w = x[:N]
    return {"w": w, "z": np.round(x[N:2 * N]), "f": dev(r, w), "c": c0}


def dev(r, w):
    """(1/S) sum_s |(r_s - mu)'w|, recomputed from the returns at w."""
    r = np.asarray(r, float)
    return float(np.mean(np.abs((r - r.mean(axis=0)) @ w)))
