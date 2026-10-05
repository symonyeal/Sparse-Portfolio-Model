"""Bessembinder's (2018) bootstrap of random K-stock portfolios (sec. 4, Table 4).

For each month, K stocks are drawn at random from those with a return that month; the month's
portfolio return is value-weighted by beginning-of-month market value; months are linked. The
procedure is repeated n times (20,000 in the source) and each linked return is compared with
zero, the T-bill and the value-weighted market over the same months.

This redraws the names every month. The specification's B4 holds K names bought once, equally
weighted (pf/bt.py, b4); that is an extension, not the source's protocol. Which months the
source links for its 1- and 10-year outcomes is not stated; the windows are an argument here,
fixed in docs/PLAN.md before any run on CRSP.
"""

import numpy as np

# Legend
# R      : returns (Tm, N), nan where a stock has no return
# I      : beginning-of-month market value (Tm, N), nan where unknown
# K      : stocks drawn per month (1 in the source's single-stock case)
# n      : replications; g : numpy Generator
# Rp     : portfolio returns (n, Tm)
# wd     : windows as (first month index, last month index) inclusive
# rf, vw : T-bill and value-weighted market returns (Tm,)


def draw(R, I, K, n, g):
    """Monthly value-weighted returns of K stocks drawn afresh each month, n replications."""
    R, I = arrays(R, I)
    if (not isinstance(K, (int, np.integer)) or K < 1
            or not isinstance(n, (int, np.integer)) or n < 1):
        raise ValueError("draw sizes must be positive integers")
    Tm = R.shape[0]
    Rp = np.empty((n, Tm))
    for t in range(Tm):
        a = np.flatnonzero(np.isfinite(R[t]) & np.isfinite(I[t]) & (I[t] > 0))
        if a.size < K:
            raise ValueError(f"month {t}: {a.size} stocks available, {K} drawn")
        c = np.array([g.choice(a, size=K, replace=False) for _ in range(n)])
        v = I[t][c]
        Rp[:, t] = np.sum(v * R[t][c], axis=1) / v.sum(axis=1)
    return Rp


def vw(R, I):
    """Value-weighted market return each month over stocks with a return and a value."""
    R, I = arrays(R, I)
    k = np.isfinite(R) & np.isfinite(I) & (I > 0)
    v = np.where(k, I, 0)
    total = v.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("market return unavailable in a month without eligible value")
    return (v * np.nan_to_num(R)).sum(axis=1) / total


def arrays(R, I):
    """Aligned return and beginning-value matrices; missing observations stay missing."""
    R, I = np.asarray(R, float), np.asarray(I, float)
    if R.ndim != 2 or R.shape != I.shape or 0 in R.shape:
        raise ValueError("return and value matrices disagree in shape")
    if np.isinf(R).any() or np.isinf(I).any() or np.any(R[~np.isnan(R)] < -1):
        raise ValueError("return or market value is invalid")
    return R, I


def link(Rp, wd):
    """Linked returns prod(1 + r) - 1 over each window, per replication: (n, len(wd))."""
    Rp = np.atleast_2d(np.asarray(Rp, float))
    if Rp.ndim != 2 or not np.isfinite(Rp).all() or np.any(Rp < -1) or not wd:
        raise ValueError("linked returns require finite returns and nonempty windows")
    if any(not (0 <= a <= b < Rp.shape[1]) for a, b in wd):
        raise ValueError("link window is outside the return series")
    return np.stack([np.prod(1 + Rp[:, a:b + 1], axis=1) - 1 for a, b in wd], axis=1)


def table(Rp, rf, vwm, wd):
    """Table 4 rows for the windows wd: mean, median, skewness, and shares > 0, > T-bill, > VW."""
    x = link(Rp, wd).ravel()
    f = np.repeat(link(rf, wd), Rp.shape[0], axis=0).ravel()
    m = np.repeat(link(vwm, wd), Rp.shape[0], axis=0).ravel()
    return {**stats(x, f, m), "n": int(x.size)}


def stats(x, f, m):
    """Linked-return statistics against paired T-bill and market outcomes."""
    x, f, m = (np.asarray(a, float) for a in (x, f, m))
    if (x.ndim != 1 or x.size == 0 or not (x.shape == f.shape == m.shape)
            or not all(np.isfinite(a).all() for a in (x, f, m))):
        raise ValueError("bootstrap outcomes must be nonempty finite paired vectors")
    sd = x.std()
    sk = float(np.mean(((x - x.mean()) / sd) ** 3)) if sd else None
    return {"mean": float(x.mean()), "med": float(np.median(x)), "skew": sk,
            "gt0": float(np.mean(x > 0)), "gtf": float(np.mean(x > f)),
            "gtvw": float(np.mean(x > m))}
