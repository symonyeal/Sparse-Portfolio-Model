"""F1 by the extended cutting-plane method: a HiGHS MILP master over linearisations of w'Sw.

Westerlund and Pettersson (1995), eqs. (9)-(15). The master minimises lam v - (1 - lam) mu'w
over X(K), the return row, and the cuts

    v >= f(w_k) + grad f(w_k)'(w - w_k) = 2 w_k'S w - w_k'S w_k,    f(w) = w'Sw,

each added at the master solution w_k it cuts off. Every master is a relaxation of F1, so its
dual bound (scipy milp `mip_dual_bound`) is a lower bound on F1, and the master values increase
monotonically (their eq. (15)). Every w_k lies in X(K), so its exact objective is an upper bound.

This route shares only X(K) (pf/rw.py) with pf/mv.py; it uses no SCIP and no quadratic solver.
"""

import time

import numpy as np
from scipy import sparse

from pf import crt, mv, rw

# Legend
# mu, S, K, lam, e, d, sn, rho, rs : as in pf/mv.py
# x      : master columns [w (N), z (N), v]
# P      : cut points w_k
# lb, ub : lower bound (best master dual bound) and upper bound (least exact objective at a w_k)
# tol    : stop when ub - lb <= tol * max(|ub|, ga)
# it     : iteration limit; tl : total time limit (s)
# sg     : objective scale 1 / mean(diag S), as in pf/mv.py
# pt     : bound path, (iteration, lb, ub) in original units


def ecp(mu, S, K, lam, e=0.0, d=1.0, sn="<=", rho=None, rs="==", tol=1e-8, it=2000, tl=None,
        ga=crt.GA):
    """Solve F1 by extended cutting planes; returns w, z, f, the certificate and the bound path.

    The tail converges slowly: on port1 with K = 10 and lam = 1, 2,000 masters left a relative
    bracket of 1.1e-6. A limit returns status `limit` with both bounds.
    """
    mu, S = rw.ms(mu, S)
    crt.limits(tl)
    if not 0 <= lam <= 1:
        raise ValueError("lam must lie in [0, 1]")
    if not np.isfinite(tol) or tol < 0 or not np.isfinite(ga) or ga <= 0:
        raise ValueError("tol must be finite and nonnegative; ga must be finite and positive")
    if not np.isfinite(it) or int(it) != it or it < 0:
        raise ValueError("it must be a nonnegative integer")
    N = mu.size
    sg = rw.scale(S)
    Ss, ms = sg * S, sg * mu
    B = rw.pad(rw.X(N, K, e, d, sn), 1)
    if rho is not None:
        B = rw.stack(B, rw.pad(rw.R(mu, rho, rs), 1))
    c = np.r_[-(1 - lam) * ms, np.zeros(N), lam]
    l = np.r_[B["lb"], 0.0]
    u = np.r_[B["ub"], np.inf]
    itg = np.r_[B["it"], 0]
    P = [np.full(N, 1.0 / N)]                 # any point of the simplex gives a valid first cut
    lb, ub, best, pt, st = -np.inf, np.inf, None, [], "limit"
    t0 = time.perf_counter()
    while len(pt) < it:
        G = np.array([np.r_[-2 * Ss @ w, np.zeros(N), 1.0] for w in P])   # v - 2 w_k'Ss w
        A = sparse.vstack([B["A"], sparse.csr_matrix(G)], format="csr")
        lo = np.r_[B["lo"], [-(w @ Ss @ w) for w in P]]
        hi = np.r_[B["hi"], np.full(len(P), np.inf)]
        r = None if tl is None else max(tl - (time.perf_counter() - t0), 0.0)
        x, cm = crt.mi(c, A, lo, hi, l, u, itg, tl=r, gp=0.0)
        lb = max(lb, cm["bd"])
        if x is None:
            st = cm["st"]
            break
        w = x[:N]
        f = sg * mv.obj(w, mu, S, lam)
        if f < ub:
            ub, best = f, (w.copy(), np.round(x[N:2 * N]))
        pt.append((len(pt) + 1, lb / sg, ub / sg))
        if cm["st"] == "opt" and ub - lb <= tol * max(abs(ub), ga):
            st = "opt"
            break
        if cm["st"] != "opt" or (tl is not None and time.perf_counter() - t0 >= tl):
            st = "limit" if cm["st"] in ("opt", "limit") else cm["st"]
            break
        P.append(w.copy())
    cert = crt.un(crt.C(st, "min", ub if best is not None else None,
                        lb if np.isfinite(lb) else None, t=time.perf_counter() - t0,
                        it=len(pt), bs="ecp master mip_dual_bound"), sg)
    if best is None:
        return {"w": None, "z": None, "f": np.nan, "c": cert, "pt": pt}
    return {"w": best[0], "z": best[1], "f": mv.obj(best[0], mu, S, lam), "c": cert, "pt": pt}

