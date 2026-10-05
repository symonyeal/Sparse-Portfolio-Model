"""F3: CVaR selection over X(K), a MILP solved by HiGHS.

Rockafellar and Uryasev (2000), eq. (17) with rows (15)-(16) and X(K) in place of their (11):

    min al + 1/((1 - b) S) sum_s u_s
    s.t. u_s >= -r_s'w - al,  u_s >= 0,  mu'w >= rho,  (w, z) in X(K)

The loss of scenario s is -r_s'w. `cvar` recomputes beta-CVaR at w by sorting the S losses, with
the fractional atom at the beta-quantile when (1 - b) S is not an integer; by their Theorem 1 it
equals the optimal value of the program at that w.
"""

import numpy as np
from scipy import sparse

from pf import crt, rw

# Legend
# r      : scenario returns (S, N), each scenario with probability 1/S
# b      : CVaR level beta in (0, 1)
# al     : VaR variable alpha (free)
# u      : tail excess u_s = max(0, loss_s - alpha)
# L      : scenario losses -r_s'w, sorted ascending in `cvar`
# tt     : sum of the k largest losses; atom : the fractional quantile atom
# K, e, dl, sn, rho, tl, gp : as in pf/mad.py


def cv(r, K, b=0.95, rho=None, e=0.0, dl=1.0, sn="<=", tl=None, gp=0.0):
    """Solve F3; returns w, z, alpha, the recomputed CVaR and the certificate."""
    r = rw.rr(r)
    S, N = r.shape
    if not 0 < b < 1:
        raise ValueError("b must lie in (0, 1)")
    mu = r.mean(axis=0)
    B = rw.pad(rw.X(N, K, e, dl, sn), 1 + S)
    if rho is not None:
        B = rw.stack(B, rw.pad(rw.R(mu, rho, ">="), 1 + S))
    U = sparse.hstack([r, sparse.csr_matrix((S, N)), np.ones((S, 1)), sparse.identity(S)],
                      format="csr")                                   # r_s'w + al + u_s >= 0
    A = sparse.vstack([B["A"], U], format="csr")
    lo = np.r_[B["lo"], np.zeros(S)]
    hi = np.r_[B["hi"], np.full(S, np.inf)]
    c = np.r_[np.zeros(2 * N), 1.0, np.full(S, 1.0 / ((1 - b) * S))]
    lb = np.r_[B["lb"], -np.inf, np.zeros(S)]
    ub = np.r_[B["ub"], np.inf, np.full(S, np.inf)]
    x, c0 = crt.mi(c, A, lo, hi, lb, ub, np.r_[B["it"], np.zeros(1 + S)], tl=tl, gp=gp)
    if x is None:
        return {"w": None, "z": None, "al": np.nan, "f": np.nan, "c": c0}
    w = x[:N]
    return {"w": w, "z": np.round(x[N:2 * N]), "al": float(x[2 * N]), "f": cvar(r, w, b),
            "c": c0}


def cvar(r, w, b):
    """beta-CVaR of the losses -r_s'w under equal weights 1/S, from the sorted losses."""
    if not 0 < b < 1:
        raise ValueError("b must lie in (0, 1)")
    L = np.sort(-(np.asarray(r, float) @ w))
    S = L.size
    m = (1 - b) * S                      # probability mass of the tail, in scenarios
    k = int(np.floor(m))
    tt = L[S - k:].sum() if k else 0.0
    fr = m - k                           # fractional atom at the beta-quantile
    atom = L[S - k - 1] * fr if fr > 0 else 0.0
    return float((tt + atom) / m)
