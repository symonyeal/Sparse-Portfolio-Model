"""F1 and F1p: cardinality-constrained mean-variance selection, solved by SCIP.

F1  (Chang et al., 2000, eq. (16)): min lam w'Sw - (1 - lam) mu'w over X(K); with lam = 1 and
    a return row mu'w = rho it is their eqs. (8)-(13).
F1p (Frangioni and Gentile, 2006; Gunluk and Linderoth, 2010, eqs. (17)-(23)): S = Q + diag(pi),
    pi_i w_i^2 replaced by pi_i t_i with w_i^2 <= t_i z_i. Same optimum, tighter relaxation.

The quadratic enters through an epigraph v >= w'Sw, because SCIP takes a linear objective.

Gunluk and Linderoth take pi = (l_min - 0.001) 1 with l_min the least eigenvalue of S. At the
scale of weekly returns l_min is near 1e-4, so the absolute 0.001 makes pi negative; here
pi = th l_min with th = 0.999 keeps Q positive definite at any scale.

`qp` returns a point of the continuous frontier (K = N, e = 0) with a KKT certificate built from
the active set; it is the exact reference for gate G1.
"""

import time

import numpy as np

from pf import crt, rw

# Legend
# mu, S  : mean vector and covariance matrix
# K      : cardinality limit; e, d : floors and ceilings; sn : cardinality sense
# lam    : risk-return weight of Chang et al. (2000), in [0, 1]
# rho, rs: required return and the sense of its row (None: no return row)
# p      : True for the perspective formulation F1p
# Q, pi  : perspective split S = Q + diag(pi), Q positive definite, pi >= 0
# th     : fraction of the least eigenvalue placed in pi
# v      : epigraph variable of w'Sw (F1) or of w'Qw + pi't (F1p)
# t      : perspective variables, w_i^2 <= t_i z_i
# tl, gp : time limit (s) and requested relative gap; ft : SCIP feasibility tolerance
# f      : objective recomputed at the returned w; c : certificate (pf/crt.py)
# F      : free set of an active-set solve; y : multipliers of the budget and return rows
# kt     : KKT residuals: stationarity, primal, dual sign

TH = 0.999


def obj(w, mu, S, lam):
    """lam w'Sw - (1 - lam) mu'w at w."""
    w = np.asarray(w, float)
    return float(lam * w @ S @ w - (1 - lam) * mu @ w)


def split(S, th=TH):
    """Frangioni-Gentile diagonal split S = Q + diag(pi), pi = th * l_min(S) * 1."""
    l = np.linalg.eigvalsh(S)[0]
    if l <= 0:
        raise ValueError(f"S is not positive definite (least eigenvalue {l:g})")
    pi = np.full(S.shape[0], th * l)
    return S - np.diag(pi), pi


def f1(mu, S, K, lam, e=0.0, d=1.0, sn="<=", rho=None, rs="==", p=False, tl=None, gp=0.0,
       ft=1e-9, sol=None):
    """Solve F1 (p=False) or F1p (p=True) with SCIP; returns w, z, f and the certificate."""
    from pyscipopt import Model, quicksum
    mu, S = np.asarray(mu, float), np.asarray(S, float)
    N = mu.size
    if not 0 <= lam <= 1:
        raise ValueError("lam must lie in [0, 1]")
    B = rw.X(N, K, e, d, sn)
    sg = 1.0 / np.mean(np.diag(S))      # objective scale: SCIP's tolerances are absolute below 1
    Ss, ms = sg * S, sg * mu
    m = Model()
    m.hideOutput()
    m.setParam("numerics/feastol", ft)
    m.setParam("limits/gap", gp)
    m.setParam("limits/absgap", 0.0)
    if tl is not None:
        m.setParam("limits/time", float(tl))
    w = [m.addVar(f"w{i}", lb=0.0, ub=float(B["ub"][i])) for i in range(N)]
    z = [m.addVar(f"z{i}", vtype="B") for i in range(N)]
    v = m.addVar("v", lb=0.0)
    rw.scip(m, B, w + z)
    if rho is not None:
        rw.scip(m, rw.R(mu, rho, rs), w + z)
    if p:
        Q, pi = split(Ss)
        t = [m.addVar(f"t{i}", lb=0.0) for i in range(N)]
        m.addCons(v >= quicksum(Q[i, j] * w[i] * w[j] for i in range(N) for j in range(N))
                  + quicksum(pi[i] * t[i] for i in range(N)))
        for i in range(N):
            m.addCons(w[i] * w[i] <= t[i] * z[i])
    else:
        m.addCons(v >= quicksum(Ss[i, j] * w[i] * w[j] for i in range(N) for j in range(N)))
    m.setObjective(lam * v - quicksum((1 - lam) * ms[i] * w[i] for i in range(N)), "minimize")
    if sol is not None:
        s0 = m.createSol()
        for i in range(N):
            m.setSolVal(s0, w[i], sol[0][i])
            m.setSolVal(s0, z[i], sol[1][i])
        m.addSol(s0)
    t0 = time.perf_counter()
    m.optimize()
    c = crt.un(crt.sc(m, "min", t=time.perf_counter() - t0), sg)
    if m.getNSols() == 0:
        return {"w": None, "z": None, "f": np.nan, "c": c}
    s = m.getBestSol()
    W = np.array([m.getSolVal(s, x) for x in w])
    Z = np.round([m.getSolVal(s, x) for x in z])
    return {"w": W, "z": Z, "f": obj(W, mu, S, lam), "c": c}


def qp(mu, S, r, w0, tol=1e-12, it=50):
    """Min w'Sw s.t. mu'w = r, 1'w = 1, 0 <= w <= 1, by an active-set solve from w0's support.

    Returns w, the variance, and the KKT residuals that certify it: for a convex QP a point
    satisfying stationarity, primal feasibility and the multiplier signs is optimal.
    """
    mu, S = np.asarray(mu, float), np.asarray(S, float)
    N = mu.size
    F = np.flatnonzero(np.asarray(w0) > 1e-7)
    A = np.vstack([np.ones(N), mu])
    b = np.array([1.0, r])
    for _ in range(it):
        n = F.size
        M = np.zeros((n + 2, n + 2))
        M[:n, :n] = 2 * S[np.ix_(F, F)]
        M[:n, n:] = -A[:, F].T
        M[n:, :n] = A[:, F]
        q = np.r_[np.zeros(n), b]
        s = np.linalg.lstsq(M, q, rcond=None)[0]
        w = np.zeros(N)
        w[F] = s[:n]
        y = s[n:]
        g = 2 * S @ w - A.T @ y             # reduced gradient; >= 0 off the free set
        neg = F[w[F] < -tol]
        if neg.size:
            F = np.setdiff1d(F, neg[np.argmin(w[neg])])
            continue
        off = np.setdiff1d(np.arange(N), F)
        if off.size and g[off].min() < -tol * max(1.0, np.abs(g).max()):
            F = np.union1d(F, off[np.argmin(g[off])])
            continue
        kt = {"st": float(np.abs(g[F]).max()) if F.size else 0.0,
                  "pr": float(np.abs(A @ w - b).max()),
                  "du": float(max(0.0, -g[off].min())) if off.size else 0.0,
                  "lo": float(max(0.0, -w.min())), "hi": float(max(0.0, w.max() - 1))}
        return {"w": w, "f": float(w @ S @ w), "y": y, "F": F, "kt": kt}
    raise RuntimeError("active-set solve did not settle")


def shrink(y):
    """Ledoit and Wolf (2004) shrinkage of the sample covariance toward constant correlation.

    Appendix A, eq. (3): target F; Appendix B, eq. (5): intensity max(0, min(k/T, 1)) with
    k = (pi - rho) / gamma. S is the sample covariance with divisor T, the normalisation the
    appendix's estimators pi_ij and theta use. Returns (shrunk matrix, intensity).
    """
    y = np.asarray(y, float)
    T, N = y.shape
    x = y - y.mean(axis=0)
    S = x.T @ x / T
    s = np.sqrt(np.diag(S))
    r = S / np.outer(s, s)
    rb = (r.sum() - N) / (N * (N - 1))
    F = rb * np.outer(s, s)
    np.fill_diagonal(F, np.diag(S))
    x2 = x * x
    P = (x2.T @ x2) / T - S * S                               # pi_ij
    pi = P.sum()
    th = ((x2 * x).T @ x) / T - np.diag(S)[:, None] * S       # theta_ii,ij in row i
    rho = np.trace(P) + rb / 2 * (np.sum((s[None, :] / s[:, None]) * th)
                                  + np.sum((s[:, None] / s[None, :]) * th.T)
                                  - 2 * np.trace(th))
    ga = np.sum((F - S) ** 2)
    dl = max(0.0, min((pi - rho) / ga / T, 1.0))
    return dl * F + (1 - dl) * S, dl


def pdev(ef, x, y):
    """Chang et al. (2000), sec. 5.2.2: percentage deviation of the point (std x, mean y) from the
    linearly interpolated frontier ef = (mean, variance) rows; the lesser of the horizontal and
    vertical deviations, nan where neither bracket exists.

    The source prints 100(y* - y**)/y** for the vertical direction, which is negative for a
    point below the frontier; both directions are taken in absolute value here.
    """
    fy, fx = ef[:, 0], np.sqrt(ef[:, 1])
    o = np.argsort(fy)
    fy, fx = fy[o], fx[o]
    d = []
    if fy[0] <= y <= fy[-1]:
        x2 = np.interp(y, fy, fx)
        d.append(abs(100 * (x - x2) / x2))
    if fx[0] <= x <= fx[-1]:
        y2 = np.interp(x, fx, fy)
        d.append(abs(100 * (y - y2) / y2))
    return min(d) if d else np.nan
