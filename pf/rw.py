"""The feasible set X(K) of a cardinality-constrained long-only portfolio, written once.

Chang, Meade, Beasley and Sharaiha (2000), eqs. (10)-(13), with the sense of (11) a parameter.

    X(K) = { (w, z) : 1'w = 1,  e_i z_i <= w_i <= d_i z_i,  1'z (<= or =) K,  z in {0,1}^N }

Columns are ordered x = [w, z, extra...]; a formulation appends its own columns after the 2N
here and reaches these rows through `pad`. The return row mu'w (>= or =) rho is the second
shared row block, `R`.

With e_i = 0 and sense "==", z_i = 1 marks an admitted name, not a positive holding: a solution
may carry w_i = 0 with z_i = 1. `ok` therefore checks the rows, and `nz` counts holdings.
"""

import numpy as np
from scipy import sparse

# Legend
# N      : number of assets
# K      : cardinality limit
# e, d   : floor eps_i and ceiling delta_i on a held weight (scalars broadcast to length N)
# sn     : sense of the cardinality row, "<=" or "=="; Chang et al. (2000) impose "=="
# mu     : mean return vector
# rho    : required mean return
# rs     : sense of the return row, ">=" (Konno-Yamazaki; Rockafellar-Uryasev) or "==" (Chang)
# A      : row matrix (scipy.sparse CSR) over the columns of x
# lo, hi : row bounds
# lb, ub : column bounds
# it     : integrality of each column (1 binary/integer, 0 continuous)
# m      : number of extra columns appended after [w, z]
# tol    : feasibility tolerance of a check

SN = ("<=", "==")
RS = (">=", "==")


def rr(r):
    """A nonempty finite scenario matrix with rows s and columns i."""
    r = np.asarray(r, float)
    if r.ndim != 2 or min(r.shape) == 0:
        raise ValueError("r must be a nonempty two-dimensional scenario matrix")
    if not np.isfinite(r).all():
        raise ValueError("r must contain only finite returns")
    return r


def ms(mu, S):
    """A finite mean vector and symmetric positive-semidefinite covariance matrix."""
    mu, S = np.asarray(mu, float), np.asarray(S, float)
    if mu.ndim != 1 or not mu.size or S.shape != (mu.size, mu.size):
        raise ValueError("mu and S must have shapes (N,) and (N,N), N >= 1")
    if not np.isfinite(mu).all() or not np.isfinite(S).all():
        raise ValueError("mu and S must be finite")
    a = max(float(np.abs(S).max()), np.finfo(float).tiny)
    if not np.allclose(S, S.T, rtol=0, atol=1e-12 * a):
        raise ValueError("S must be symmetric")
    if np.linalg.eigvalsh(S)[0] < -1e-12 * a:
        raise ValueError("S must be positive semidefinite")
    return mu, S


def scale(S):
    """Finite objective scale; deterministic returns retain scale one."""
    a = float(np.mean(np.diag(S)))
    return 1.0 / a if a > np.finfo(float).tiny else 1.0


def ed(N, e, d):
    """Floors and ceilings as length-N arrays; 0 <= e_i <= d_i <= 1 (Chang et al., sec. 2.4)."""
    e = np.broadcast_to(np.asarray(e, float), (N,)).copy()
    d = np.broadcast_to(np.asarray(d, float), (N,)).copy()
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(d))):
        raise ValueError("floors and ceilings must be finite")
    if np.any(e < 0) or np.any(d > 1) or np.any(e > d):
        raise ValueError("need 0 <= e_i <= d_i <= 1")
    return e, d


def X(N, K, e=0.0, d=1.0, sn="<="):
    """Rows, row bounds, column bounds and integrality of X(K) over x = [w, z]."""
    if sn not in SN:
        raise ValueError(f"cardinality sense {sn!r} not in {SN}")
    if not np.isfinite(N) or int(N) != N or N < 1:
        raise ValueError("N must be a positive integer")
    if not (1 <= K <= N) or int(K) != K:
        raise ValueError(f"need integer 1 <= K <= N, got K={K}, N={N}")
    N, K = int(N), int(K)
    e, d = ed(N, e, d)
    I = sparse.identity(N, format="csr")
    A = sparse.vstack([
        sparse.hstack([np.ones((1, N)), sparse.csr_matrix((1, N))]),      # 1'w = 1
        sparse.hstack([I, -sparse.diags(e)]),                                # w - e z >= 0
        sparse.hstack([I, -sparse.diags(d)]),                                # w - d z <= 0
        sparse.hstack([sparse.csr_matrix((1, N)), np.ones((1, N))]),        # 1'z  sn  K
    ], format="csr")
    lo = np.r_[1.0, np.zeros(N), np.full(N, -np.inf), K if sn == "==" else -np.inf]
    hi = np.r_[1.0, np.full(N, np.inf), np.zeros(N), K]
    lb = np.zeros(2 * N)
    ub = np.r_[d, np.ones(N)]
    it = np.r_[np.zeros(N), np.ones(N)]
    return {"A": A, "lo": lo, "hi": hi, "lb": lb, "ub": ub, "it": it, "N": N, "K": K, "e": e,
            "d": d, "sn": sn}


def R(mu, rho, rs=">="):
    """The return row mu'w (rs) rho over x = [w, z]."""
    if rs not in RS:
        raise ValueError(f"return sense {rs!r} not in {RS}")
    mu = np.asarray(mu, float)
    if mu.ndim != 1 or not mu.size or not np.isfinite(mu).all() or not np.isfinite(rho):
        raise ValueError("mu must be a finite vector and rho must be finite")
    N = mu.size
    A = sparse.csr_matrix(np.r_[mu, np.zeros(N)][None, :])
    return {"A": A, "lo": np.array([rho]), "hi": np.array([rho if rs == "==" else np.inf])}


def pad(B, m):
    """Row block B over [w, z] extended by m zero columns."""
    A = B["A"]
    return dict(B, A=sparse.hstack([A, sparse.csr_matrix((A.shape[0], m))], format="csr"))


def stack(*Bs):
    """One row block from several over the same columns; column data from the first."""
    return Bs[0] | {"A": sparse.vstack([b["A"] for b in Bs], format="csr"),
            "lo": np.concatenate([b["lo"] for b in Bs]),
            "hi": np.concatenate([b["hi"] for b in Bs])}


def scip(m, B, x):
    """Add the rows of block B to pyscipopt model m over the variable list x."""
    from pyscipopt import quicksum
    A = B["A"].tocsr()
    for r in range(A.shape[0]):
        s, t = A.indptr[r], A.indptr[r + 1]
        ex = quicksum(float(v) * x[j] for j, v in zip(A.indices[s:t], A.data[s:t]))
        lo, hi = B["lo"][r], B["hi"][r]
        if lo == hi:
            m.addCons(ex == float(lo))
        else:
            if np.isfinite(lo):
                m.addCons(ex >= float(lo))
            if np.isfinite(hi):
                m.addCons(ex <= float(hi))


def ok(w, z, K, e=0.0, d=1.0, sn="<=", tol=1e-7):
    """Violated conditions of (w, z) in X(K), checked from the definition; [] if none."""
    w, z = np.asarray(w, float), np.asarray(z, float)
    if w.ndim != 1 or z.ndim != 1 or not w.size or w.shape != z.shape:
        return ["w and z must be nonempty vectors of the same length"]
    if not np.isfinite(w).all() or not np.isfinite(z).all():
        return ["w and z must be finite"]
    if sn not in SN or not (1 <= K <= w.size) or int(K) != K:
        return ["invalid cardinality parameter"]
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    N = w.size
    e, d = ed(N, e, d)
    v = []
    if z.size != N:
        v.append("w and z differ in length")
        return v
    if np.any(np.abs(z - np.round(z)) > tol) or np.any((np.round(z) != 0) & (np.round(z) != 1)):
        v.append("z not binary")
    zb = np.round(z)
    if abs(w.sum() - 1) > tol:
        v.append(f"1'w = {w.sum():.12g}")
    if np.any(w < e * zb - tol):
        v.append("floor e_i z_i <= w_i violated")
    if np.any(w > d * zb + tol):
        v.append("ceiling w_i <= d_i z_i violated")
    k = zb.sum()
    if (sn == "<=" and k > K) or (sn == "==" and k != K):
        v.append(f"1'z = {k:g}, K = {K} ({sn})")
    return v


def nz(w, tol=1e-9):
    """Number of positive holdings."""
    return int(np.sum(np.asarray(w) > tol))
