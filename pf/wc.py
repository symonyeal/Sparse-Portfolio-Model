"""Delisting-adjusted returns, Bessembinder's (2018) wealth creation, and the set T.

Wealth creation, Bessembinder (2018), eq. (3): for a stock with beginning-of-month market value
I_{t-1}, return R_t and T-bill return rf_t,

    W_T - W_0 FV_{0,T} = sum_t I_{t-1} (R_t - rf_t) FV_{t,T},
    FV_{t,T} = prod_{u=t+1..T} (1 + rf_u),

summed across a firm's share classes (PERMCO). T is the smallest set of top firms whose wealth
creation sums to the net wealth creation of all firms ("the 1,092 top-performing companies ...
account for all of the net wealth creation").

Delisting: R = (1 + RET)(1 + DLRET) - 1; DLRET alone when RET is missing in the delisting month.
A missing DLRET after a performance delisting is filled, when asked, with -30% on NYSE/AMEX
(Shumway, 1997: codes 500, 520-584) and -55% on Nasdaq (Shumway and Warther, 1999: codes 500,
505-588). Nothing missing is read as zero.
"""

import numpy as np

# Legend
# ret, dl  : CRSP RET and DLRET (nan when missing); dc : DLSTCD (0 if none); ex : EXCHCD
# fill     : apply the Shumway fill to a missing DLRET after a performance delisting
# R        : combined monthly return; nan when neither RET nor DLRET is known
# I        : beginning-of-month market value |PRC| SHROUT of the previous month
# rf       : T-bill return per month on the global calendar; FV : forward accumulation factors
# W        : wealth creation per stock or per firm
# pc       : PERMCO of each stock; T : the set of top firms

NA = (1, 2)                   # EXCHCD NYSE, AMEX (Shumway, 1997)
NQ = (3,)                     # EXCHCD Nasdaq (Shumway and Warther, 1999)


def perf(dc, ex):
    """True where DLSTCD dc is a performance delisting on exchange ex, by the source's ranges."""
    dc, ex = np.asarray(dc), np.asarray(ex)
    na = np.isin(ex, NA) & ((dc == 500) | ((dc >= 520) & (dc <= 584)))
    nq = np.isin(ex, NQ) & ((dc == 500) | ((dc >= 505) & (dc <= 588)))
    return na | nq


def sw(ex):
    """Shumway's replacement delisting return for exchange ex: -0.30 NYSE/AMEX, -0.55 Nasdaq."""
    ex = np.asarray(ex)
    return np.where(np.isin(ex, NQ), -0.55, np.where(np.isin(ex, NA), -0.30, np.nan))


def comb(ret, dl, dc, ex, fill=False):
    """Combined monthly return and the count of Shumway fills applied."""
    ret, dl = np.asarray(ret, float), np.asarray(dl, float).copy()
    dc = np.asarray(dc)
    if not (ret.shape == dl.shape == dc.shape == np.asarray(ex).shape):
        raise ValueError("delisting arrays disagree in shape")
    for x in (ret, dl):
        if np.isinf(x).any() or np.any(x[~np.isnan(x)] < -1):
            raise ValueError("return must be missing or finite and at least -1")
    n = 0
    if fill:
        m = np.isnan(dl) & perf(dc, ex)
        dl[m] = sw(np.asarray(ex))[m]
        n = int(np.sum(m & ~np.isnan(dl)))
    R = np.where(np.isnan(dl), ret, np.where(np.isnan(ret), dl, (1 + ret) * (1 + dl) - 1))
    return R, n


def FV(rf):
    """Forward factor of each month: the product of (1 + rf) over the later months."""
    g = np.r_[np.cumprod((1 + np.asarray(rf, float))[::-1])[::-1][1:], 1.0]
    return g


def wc(I, R, rf):
    """Bessembinder eq. (3) for one stock: I, R, rf aligned month by month to the horizon end."""
    I, R, rf = (np.asarray(x, float) for x in (I, R, rf))
    if I.ndim != 1 or not (I.shape == R.shape == rf.shape):
        raise ValueError("wealth-creation arrays disagree in shape")
    if not all(np.isfinite(x).all() for x in (I, R, rf)):
        raise ValueError("wealth creation needs every month's value and return; none is zero")
    return float(np.sum(I * (R - rf) * FV(rf)))


def firm(W, pc):
    """Wealth creation summed by PERMCO: (firms, totals)."""
    u, k = np.unique(np.asarray(pc), return_inverse=True)
    return u, np.bincount(k, weights=np.asarray(W, float))


def top(W):
    """Indices of the smallest set of largest W whose sum reaches sum(W) (Bessembinder's T)."""
    W = np.asarray(W, float)
    if W.ndim != 1 or not np.isfinite(W).all():
        raise ValueError("wealth creation must be a finite vector")
    if W.size == 0 or W.sum() <= 0:
        return np.array([], dtype=int)
    o = np.argsort(-W, kind="stable")
    c = np.cumsum(W[o])
    k = int(np.argmax(c >= W.sum() - 1e-9 * max(1.0, abs(W.sum())))) + 1
    return o[:k]


def bh(R):
    """Buy-and-hold return prod(1 + R) - 1."""
    return float(np.prod(1 + np.asarray(R, float)) - 1)
