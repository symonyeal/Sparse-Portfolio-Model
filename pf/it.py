"""F4: regression-based index tracking over X(K), restricted to zero transaction costs.

Canakgoz and Beasley (2009), eqs. (1)-(11) and (16)-(26). Each stock's log return is regressed
on the index log return with an intercept, r_it = a_i + b_i R_t + e_it, and the portfolio's
coefficients are a = sum_i w_i a_i, b = sum_i w_i b_i. The model is lexicographic:

    stage 1   min D   s.t.  D >= a,  D >= -a,                               (w, z) in X(K)
    stage 2   min E   s.t.  E >= b - 1,  E >= 1 - b,  |a - a*| <= ea,        (w, z) in X(K)
    stage 3   min sum_i G_i  (transaction cost), with a and b held

Restriction, set by the owner on 2 October 2026: the budget is the shared fully invested one,
so c = f_is = f_ib = 0, G = 0, w = xV/C, and stage 3 is constant and omitted. The source's
cost rows (3)-(6) are therefore absent, not replaced. The source holds 1'z = K; ea is a project
setting (the paper says only that eps is small).
"""

import numpy as np
from scipy import sparse

from pf import crt, rw

# Legend
# r, Ri  : stock log returns (T, N) and index log returns (T)
# a, b   : OLS intercepts and slopes of each stock on the index (with intercept)
# D, E   : stage objectives |a'w| and |b'w - 1|
# ea     : tolerance replacing the stage-2 equality a = a* (Canakgoz-Beasley eq. (23))
# f1, f2 : recomputed objectives at the respective stage incumbents
# a1     : signed intercept fixed after stage 1; st : overall lexicographic solve status
# K, e, dl, sn, tl, gp : as in pf/mad.py; sn defaults to the source's 1'z = K
# te     : tracking error, the standard deviation of r'w - Ri over the window

EA = 1e-9


def ols(r, Ri):
    """Intercepts and slopes of each column of r on Ri by least squares with an intercept."""
    r, Ri = np.asarray(r, float), np.asarray(Ri, float)
    if (Ri.ndim != 1 or r.ndim not in (1, 2) or Ri.size < 2
            or r.shape[0] != Ri.size or (r.ndim == 2 and r.shape[1] < 1)):
        raise ValueError("returns require matching observations and a one-dimensional index")
    if not (np.isfinite(r).all() and np.isfinite(Ri).all()):
        raise ValueError("stock and index returns must be finite")
    if np.ptp(Ri) == 0:
        raise ValueError("constant index returns: the slope is undefined")
    X = np.c_[np.ones(Ri.size), Ri]
    c, _, rank, _ = np.linalg.lstsq(X, r, rcond=None)
    if rank != 2:
        raise ValueError("index returns do not identify an intercept and slope")
    return c[0], c[1]


def it(r, Ri, K, e=0.01, dl=1.0, sn="==", ea=EA, tl=None, gp=0.0):
    """Solve stages 1 and 2; returns w, z, (a, b) at w, and both stage certificates."""
    r, Ri = np.asarray(r, float), np.asarray(Ri, float)
    if r.ndim != 2:
        raise ValueError("stock returns must be a two-dimensional observations-by-assets array")
    if not np.isfinite(ea) or ea < 0:
        raise ValueError("ea must be finite and nonnegative")
    a, b = ols(r, Ri)
    N = a.size
    B = rw.pad(rw.X(N, K, e, dl, sn), 1)
    l, u, itg = np.r_[B["lb"], 0.0], np.r_[B["ub"], np.inf], np.r_[B["it"], 0]
    c = np.r_[np.zeros(2 * N), 1.0]

    def rows(q, t):                          # D >= q'w - t and D >= t - q'w
        return sparse.csr_matrix(np.vstack([np.r_[-q, np.zeros(N), 1.0],
                                            np.r_[q, np.zeros(N), 1.0]])), np.array([-t, t])

    G1, l1 = rows(a, 0.0)
    x1, c1 = crt.mi(c, sparse.vstack([B["A"], G1]), np.r_[B["lo"], l1],
                    np.r_[B["hi"], np.full(2, np.inf)], l, u, itg, tl=tl, gp=gp)
    out = {"w": None, "z": None, "c1": c1, "c2": None, "st": c1["st"],
           "a": None, "b": None, "a1": None, "f1": None, "f2": None, "ea": float(ea)}
    if x1 is None:
        return out

    def checked(x, cert, stage):
        w, z = x[:N], x[N:2 * N]
        if rw.ok(w, z, K, e, dl, sn):
            raise ValueError(f"F4 stage {stage} incumbent violates X(K)")
        ai, bi = (float(v) for v in ols(r @ w, Ri))
        actual = abs(ai) if stage == 1 else abs(bi - 1.0)
        if (not np.isfinite(cert["f"]) or not np.isclose(x[-1], cert["f"], atol=1e-8)
                or cert["f"] < actual - 1e-8
                or (cert["st"] == "opt"
                    and not np.isclose(actual, cert["f"], atol=1e-8, rtol=1e-7))):
            raise ValueError(f"F4 stage {stage} objective disagrees with independent regression")
        return {"w": w, "z": np.round(z), "a": ai, "b": bi}, actual

    vals, f1 = checked(x1, c1, 1)
    s = vals["a"]
    out |= vals | {"a1": s, "f1": f1}
    if c1["st"] != "opt":
        return out
    G2, l2 = rows(b, 1.0)
    H = sparse.csr_matrix(np.r_[a, np.zeros(N), 0.0][None, :])
    x2, c2 = crt.mi(c, sparse.vstack([B["A"], G2, H]),
                    np.r_[B["lo"], l2, s - ea], np.r_[B["hi"], np.full(2, np.inf), s + ea],
                    l, u, itg, tl=tl, gp=gp)
    out |= {"c2": c2, "st": c2["st"]}
    if x2 is None:
        return out
    vals, f2 = checked(x2, c2, 2)
    if abs(vals["a"] - s) > ea + 1e-8:
        raise ValueError("F4 stage 2 violates the fixed stage-1 intercept")
    return out | vals | {"f2": f2}


def te(r, Ri, w):
    """Tracking error: standard deviation of r'w - Ri (population, 1/T)."""
    return float(np.std(np.asarray(r, float) @ w - np.asarray(Ri, float)))
