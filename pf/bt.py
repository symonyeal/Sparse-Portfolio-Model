"""The monthly panel and the out-of-sample engine: eligibility, selection at t, buy-and-hold.

Selection at t receives `P.upto(t)`, a copy holding only months <= t, so no weight can depend on
a later return (the no-look-ahead test perturbs every later return). Evaluation then holds the
weights on the full panel. Protocol: specification sec. 8; DeMiguel, Garlappi and Uppal (2009)
for the rolling estimation window M.

Holding: buy-and-hold for H months. A holding whose stock leaves the panel (delisting; its last
return already includes DLRET) moves to cash at the T-bill rate. A missing return while the
stock is still listed is refused: it is never read as zero.
"""

import numpy as np

from pf import wc

# Legend
# mo      : months YYYYMM (Tm,); ids : PERMNO (N,); pc : PERMCO (N,)
# R       : combined returns (Tm, N), nan where the stock has no return that month
# ME      : end-of-month market value |PRC| SHROUT (Tm, N), nan where unknown
# rf      : T-bill return per month (Tm,)
# ls      : index of each stock's last month with a return in the full panel
# t       : month index of a rebalance date; M : estimation window; m : required returns in it
# Nc      : eligible universe size, top Nc by ME at t; U : eligible column indices
# K       : names held; w : weights over U (or over all columns); H : holding months
# Wp      : wealth path of a holding, starting at 1 at the end of month t


class P:
    """A wide monthly panel; `upto` cuts it at a month."""

    def __init__(self, mo, ids, pc, R, ME, rf, ls=None):
        self.mo, self.ids, self.pc = np.asarray(mo), np.asarray(ids), np.asarray(pc)
        self.R, self.ME = np.asarray(R, float), np.asarray(ME, float)
        self.rf = np.asarray(rf, float)
        Tm, N = self.R.shape
        if self.ME.shape != (Tm, N) or self.rf.shape != (Tm,) or self.mo.shape != (Tm,):
            raise ValueError("panel arrays disagree in shape")
        if np.isnan(self.rf).any():
            raise ValueError("T-bill return missing in a panel month")
        if ls is None:
            f = np.isfinite(self.R)
            ls = np.where(f.any(axis=0), Tm - 1 - np.argmax(f[::-1], axis=0), -1)
        self.ls = np.asarray(ls)

    def upto(self, t):
        """The panel restricted to months 0..t; `ls` is recomputed from those months only."""
        return P(self.mo[:t + 1], self.ids, self.pc, self.R[:t + 1].copy(),
                 self.ME[:t + 1].copy(), self.rf[:t + 1].copy())


def panel(rows, rfd, fill=False):
    """A panel from pf.dat.crsp rows and pf.dat.rf; returns (P, number of Shumway fills)."""
    mo = np.unique(rows["mo"])
    ids, j = np.unique(rows["PERMNO"], return_inverse=True)
    i = np.searchsorted(mo, rows["mo"])
    R0, n = wc.comb(rows["RET"], rows["DLRET"], rows["dl"], rows["EXCHCD"], fill)
    R = np.full((mo.size, ids.size), np.nan)
    ME = np.full((mo.size, ids.size), np.nan)
    R[i, j] = R0
    ME[i, j] = np.abs(rows["PRC"]) * rows["SHROUT"]
    pc = np.zeros(ids.size, dtype=np.int64)
    pc[j] = rows["PERMCO"]
    k = np.searchsorted(rfd["mo"], mo)
    if np.any(k >= rfd["mo"].size) or np.any(rfd["mo"][np.minimum(k, rfd["mo"].size - 1)] != mo):
        raise ValueError("T-bill series does not cover every panel month")
    return P(mo, ids, pc, R, ME, rfd["rf"][k]), n


def elig(Pt, M, m, Nc):
    """Columns with a market value at the last month of Pt and >= m returns in its last M months,
    the Nc largest by that market value."""
    t = Pt.R.shape[0] - 1
    if t + 1 < M:
        return np.array([], dtype=int)
    v = Pt.ME[t]
    n = np.isfinite(Pt.R[t + 1 - M:t + 1]).sum(axis=0)
    U = np.flatnonzero(np.isfinite(v) & (v > 0) & (n >= m))
    return U[np.argsort(-v[U], kind="stable")][:Nc]


def hold(P0, t, cols, w, H):
    """Wealth path over months t+1..t+H of weights w on columns cols, bought at the end of t."""
    cols, w = np.asarray(cols), np.asarray(w, float)
    if abs(w.sum() - 1) > 1e-9 or np.any(w < -1e-12):
        raise ValueError("weights must be nonnegative and sum to one")
    v = w.copy()
    cash = 0.0
    Tm = P0.R.shape[0]
    Wp = []
    for s in range(t + 1, min(t + H, Tm - 1) + 1):
        cash *= 1 + P0.rf[s]
        r = P0.R[s, cols]
        for k in np.flatnonzero(v > 0):
            if np.isfinite(r[k]):
                v[k] *= 1 + r[k]
            elif s > P0.ls[cols[k]]:          # left the panel after its delisting month
                cash += v[k] * (1 + P0.rf[s])
                v[k] = 0.0
            else:
                raise ValueError(f"missing return for PERMNO {P0.ids[cols[k]]} in "
                                 f"{P0.mo[s]} while still listed")
        Wp.append(v.sum() + cash)
    return np.array(Wp)


def b1(P0, t, H):
    """B1: the value-weighted market of every stock with a return and a market value."""
    W, x = [], 1.0
    for s in range(t + 1, min(t + H, P0.R.shape[0] - 1) + 1):
        v, r = P0.ME[s - 1], P0.R[s]
        k = np.isfinite(v) & (v > 0) & np.isfinite(r)
        x *= 1 + float(v[k] @ r[k] / v[k].sum())
        W.append(x)
    return np.array(W)


def b5(P0, t, H):
    """B5: the one-month T-bill rolled."""
    return np.cumprod(1 + P0.rf[t + 1:min(t + H, P0.R.shape[0] - 1) + 1])


def b2(Pt, U, K=None):
    """B2: 1/N over the eligible universe (DeMiguel et al., 2009)."""
    return U, np.full(U.size, 1.0 / U.size)


def b3(Pt, U, K):
    """B3: the K largest eligible names, cap-weighted."""
    c = U[:K]
    v = Pt.ME[-1, c]
    return c, v / v.sum()


def b4(Pt, U, K, g):
    """B4 (extension of Bessembinder's bootstrap): K eligible names at random, equal-weighted."""
    c = g.choice(U, size=K, replace=False)
    return c, np.full(K, 1.0 / K)


def run(P0, sel, dates, M, m, Nc, H):
    """For each rebalance index t: eligibility and selection on P0.upto(t), holding on P0."""
    out = []
    for t in dates:
        Pt = P0.upto(t)
        U = elig(Pt, M, m, Nc)
        if U.size == 0:
            continue
        cols, w = sel(Pt, U)
        out.append({"t": int(t), "mo": int(P0.mo[t]), "cols": np.asarray(cols),
                    "w": np.asarray(w), "W": hold(P0, t, cols, w, H)})
    return out
