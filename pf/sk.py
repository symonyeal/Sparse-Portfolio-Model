"""F5: mean-variance-skewness selection over X(K), a nonconvex MINLP solved by SCIP.

Harvey, Liechty, Liechty and Mueller (2010), eqs. (3)-(6): utility linear in the first three
moments, taken with plug-in sample moments (their "parameter utility"):

    max mu'w - (g/2) w'Sw + (k/6) (1/S) sum_s y_s^3,   y_s = (r_s - mu)'w,   (w, z) in X(K)

(1/S) sum_s y_s^3 is the sample third central moment of r'w, w'M3(w kron w) in Jondeau and
Rockinger (2006). The default k = g(g + 1) is the ratio U'''(1)/U'(1) for CRRA utility with
relative risk aversion g, at unit wealth; Jondeau and Rockinger expand about the expected
wealth 1 + mu'w instead, so the two agree only to first order in mu'w. The cubic makes the
objective nonconvex; only SCIP's spatial branch-and-bound dual bound certifies it.

`grid` evaluates the objective on a lattice of every support of size <= K, the independent
check of the dual bound for small instances.
"""

import itertools
import numpy as np

from pf import crt, rw

# Legend
# r      : scenario returns (S, N)
# mu, Sg : sample mean and covariance (1/S) of r, unless Sg is supplied (e.g. shrinkage)
# g, k   : variance aversion gamma and skewness preference kappa
# y      : centred portfolio returns y_s = (r_s - mu)'w
# q      : objective value; K, e, dl, sn : as in pf/rw.py
# h      : lattice step of `grid` (weights are multiples of h)
# sg     : objective scale for SCIP, 1 / mean(diag Sg)


def kap(g):
    """CRRA third-to-first derivative ratio at unit wealth: g (g + 1)."""
    return g * (g + 1.0)


def mom(r):
    """Sample mean and covariance with divisor S."""
    r = rw.rr(r)
    mu = r.mean(axis=0)
    a = r - mu
    return mu, a.T @ a / r.shape[0]


def obj(w, r, g, k, mu=None, Sg=None):
    """mu'w - (g/2) w'Sw + (k/6) mean(y^3) at w."""
    r = rw.rr(r)
    m0, S0 = mom(r)
    mu = m0 if mu is None else mu
    Sg = S0 if Sg is None else Sg
    w = np.asarray(w, float)
    y = (r - r.mean(axis=0)) @ w
    return float(mu @ w - 0.5 * g * w @ Sg @ w + k / 6.0 * np.mean(y ** 3))


def f5(r, K, g, k=None, e=0.0, dl=1.0, sn="<=", Sg=None, tl=None, gp=0.0, ft=1e-9):
    """Solve F5 with SCIP; returns w, z, the recomputed objective and the certificate (max)."""
    from pyscipopt import Model, quicksum
    r = rw.rr(r)
    crt.limits(tl, gp)
    S, N = r.shape
    k = kap(g) if k is None else k
    mu, S0 = mom(r)
    _, Sg = rw.ms(mu, S0 if Sg is None else Sg)
    if not np.isfinite(g) or not np.isfinite(k):
        raise ValueError("g and k must be finite")
    a = r - mu
    sg = rw.scale(Sg)
    B = rw.X(N, K, e, dl, sn)
    m = Model()
    m.hideOutput()
    m.setParam("numerics/feastol", ft)
    m.setParam("limits/gap", gp)
    m.setParam("limits/absgap", 0.0)
    w = [m.addVar(f"w{i}", lb=0.0, ub=float(B["ub"][i])) for i in range(N)]
    z = [m.addVar(f"z{i}", vtype="B") for i in range(N)]
    rw.scip(m, B, w + z)
    lo, hi = np.minimum(a, 0).sum(axis=1), np.maximum(a, 0).sum(axis=1)   # y_s over 0 <= w <= 1
    y = [m.addVar(f"y{s}", lb=float(lo[s]), ub=float(hi[s])) for s in range(S)]
    for s in range(S):
        m.addCons(y[s] == quicksum(float(a[s, i]) * w[i] for i in range(N)))
    q = m.addVar("q", lb=None)
    ex = (quicksum(sg * mu[i] * w[i] for i in range(N))
          - 0.5 * g * sg * quicksum(Sg[i, j] * w[i] * w[j] for i in range(N) for j in range(N))
          + (k * sg / (6.0 * S)) * quicksum(y[s] ** 3 for s in range(S)))
    m.addCons(q <= ex)
    m.setObjective(q, "maximize")
    c = crt.un(crt.so(m, "max", tl), sg)
    if m.getNSols() == 0:
        return {"w": None, "z": None, "f": np.nan, "c": c}
    s0 = m.getBestSol()
    W = np.array([m.getSolVal(s0, x) for x in w])
    Z = np.round([m.getSolVal(s0, x) for x in z])
    return {"w": W, "z": Z, "f": obj(W, r, g, k, mu, Sg), "c": c}


def grid(r, K, g, k=None, e=0.0, dl=1.0, h=0.01, Sg=None, sn="<="):
    """Best objective over the lattice in X(K), with step h (N <= 12, K <= 3)."""
    r = rw.rr(r)
    N = r.shape[1]
    N, K = rw.nk(N, K, sn)
    if N > 12 or K > 3:
        raise ValueError("the grid check is for N <= 12 and K <= 3")
    k = kap(g) if k is None else k
    mu, S0 = mom(r)
    _, Sg = rw.ms(mu, S0 if Sg is None else Sg)
    if not np.isfinite(g) or not np.isfinite(k):
        raise ValueError("g and k must be finite")
    e, dl = rw.ed(N, e, dl)
    a = r - mu
    if not np.isfinite(h) or not 0 < h <= 1 or not np.isclose(1 / h, round(1 / h),
                                                           rtol=0, atol=1e-10):
        raise ValueError("h must be the reciprocal of a positive integer")
    n = round(1 / h)
    best, arg = -np.inf, None
    sizes = range(1, K + 1) if sn == "<=" else (K,)
    for m in sizes:
        counts = range(1, n) if sn == "<=" else range(n + 1)
        c = [x for x in itertools.product(counts, repeat=m - 1)
             if sum(x) < n or (sn == "==" and sum(x) == n)]
        c = np.array(c, dtype=float).reshape(len(c), m - 1)
        V = np.c_[c, n - c.sum(axis=1)] * h                # lattice points of the m-simplex
        for sup in itertools.combinations(range(N), m):
            j = list(sup)
            W = V[np.all((V >= e[j] - 1e-12) & (V <= dl[j] + 1e-12), axis=1)]
            if not W.size:
                continue
            y = a[:, j] @ W.T
            f = (W @ mu[j] - 0.5 * g * np.einsum("pi,ij,pj->p", W, Sg[np.ix_(j, j)], W)
                 + k / 6.0 * np.mean(y ** 3, axis=0))
            p = int(np.argmax(f))
            if f[p] > best:
                best, arg = float(f[p]), np.zeros(N)
                arg[j] = W[p]
    return best, arg
